# Open-Set Clinician Identification via Radar Point-Cloud Gait, Calibrated Without Impostor Data

## 1. Introduction and Problem Statement

The objective of this work is to determine whether a person's walking pattern, captured as a
sparse point cloud by a millimeter-wave radar, is distinctive enough to identify hospital staff
in an open-set setting, and to do so using a rejection threshold that never requires a single
labelled intruder. A hospital can enrol its own clinicians, but it cannot recruit or record
labelled strangers to tune a decision boundary against, so any calibration method that assumes
access to real impostor scores is not something a ward can actually deploy.

Everything here runs on the public **mmGait10** dataset (Zenodo record 15784474, CC-BY 4.0)
[1], 10 subjects recorded with a TI MMWCAS-RF-EVM cascade radar at 77 to 81 GHz across three
walking conditions: free walking, walking with hands in pockets, and walking while using a
smartphone. 314 tracks, 178,372 frames at 10 Hz, roughly 5 hours of data in total.

The dataset's scale is a real constraint on this work and is stated here rather than left
implicit. 10 subjects is small for any identification task; with 6 enrolled and 4 held out as
strangers, the standard openness formula gives 13.4%, and splitting three random draws of which
6 are enrolled produces real seed-to-seed variance, documented throughout Section 5. Every
accuracy figure in this report should be read against that.

### 1.1 The Baseline: What Can Be Identified Without Any Gait Model At All

Before training anything, a nearest-centroid classifier was fit on seven cheap per-window
summary statistics computed directly from the raw point clouds: mean and standard deviation of
point count, mean and standard deviation of signal power in dB, mean absolute Doppler, and the
spread of the z and x coordinates. None of this is gait shape. Evaluated with 5-fold
cross-validation grouped by track (so no track contributes to both a fold's train and test side),
on all 10 subjects:

<p align="center"><img src="figures/report_shortcut_bar.png" width="700" alt="Bar chart of shortcut floor accuracy per feature"></p>

| Feature set | Accuracy | Std |
|---|---|---|
| Chance (10 subjects) | 0.100 | n/a |
| z-coordinate spread alone | 0.135 | 0.013 |
| Power in dB, mean alone | 0.144 | 0.007 |
| Power in dB, std alone | 0.150 | 0.018 |
| x-coordinate spread alone | 0.157 | 0.034 |
| Power (mean and std, 2 features) | 0.176 | 0.005 |
| Point-count spread (std) alone | 0.177 | 0.014 |
| Doppler magnitude alone | 0.183 | 0.037 |
| Point-count mean alone | 0.217 | 0.032 |
| Point count only (mean and std, 2 features) | 0.245 | 0.030 |
| All 7 metadata features | 0.293 | 0.034 |

Nearly a third of subject identity is recoverable from radar cross-section and body-size proxies
alone, with no model of walking whatsoever. This is body size and radar reflectivity, not gait,
and the finding that motivates the entire preprocessing design in Section 2: every frame is
forced to exactly 64 points before anything else happens, which removes point count as a
per-subject variable by construction. Any accuracy this project reports afterward should be read
against this 29.3% floor, not against the 10% random-chance baseline alone.

## 2. Methodology: Data Preparation and Preprocessing

### 2.1 Dataset and Split

| Channel | Role |
|---|---|
| Point cloud (x, y, z, Doppler per detection) | The identity signal. 10 Hz, forced to exactly 64 points per frame. |
| Micro-Doppler spectrogram | Present in the raw dataset (397 tracks) but not used in this pipeline. |
| Track-level subject and scenario labels | Ground truth for the split, never used as a model feature. |

Known and unknown subjects are drawn once per random seed (6 known, 4 unknown). Known subjects'
tracks are split 70/15/15 into train, validation, and test at the whole-track level, before any
windowing, so that no sliding window from one track can appear in two different splits. Unknown
subjects' tracks are used only at test time and never appear in training or in threshold
calibration.

| Seed | Known subjects | Unknown subjects | Train windows | Val windows | Test-known windows | Test-unknown windows |
|---|---|---|---|---|---|---|
| 0 | 2, 3, 4, 6, 7, 9 | 0, 1, 5, 8 | 7,195 | 1,522 | 1,583 | 6,714 |
| 1 | 1, 4, 5, 6, 7, 9 | 0, 2, 3, 8 | 7,201 | 1,468 | 1,650 | 6,695 |
| 2 | 0, 1, 5, 6, 7, 8 | 2, 3, 4, 9 | 7,106 | 1,523 | 1,568 | 6,817 |

A window is 30 frames (3 seconds at 10 Hz), with a 10-frame hop between consecutive windows.

### 2.2 Preprocessing Steps

| Step | Method | Why |
|---|---|---|
| Point count fix | Random subsample without replacement if a frame has more than 64 points, random resample with replacement if fewer | Removes the point-count shortcut in Section 1.1 by construction |
| Per-frame centring | Subtract each frame's own mean from x, y, z, and Doppler | Removes absolute position in the room and bulk walking speed; body extent (spread) survives, which is legitimate biometry, not a sensing artefact |
| Windowing | 30 frames, 10-frame hop | 3 seconds of continuous walking is the shortest span examined |
| Random-number seeding | A single seeded generator (`numpy.random.default_rng(0)`), created once and reused for every subsampling decision | Deterministic and reproducible; the original reference implementation for this dataset reseeds inconsistently between the sampling and padding code paths |

Training-time augmentation, applied only to the training split: random yaw rotation up to 15
degrees, a 50% chance of a left-right mirror flip, and 15% random point dropout with replacement
padding back to 64 points.

<p align="center"><img src="figures/radar_explainer.png" width="850" alt="Five-panel explainer of one radar frame, a walking trajectory, and the matching micro-Doppler spectrogram"></p>

## 3. Model Architecture

| Component | Structure |
|---|---|
| Frame encoder | Shared per-point MLP: Conv1d(4 to 32) to Conv1d(32 to 64) to Conv1d(64 to 128), each followed by BatchNorm and ReLU, then max-pool over the 64 points. Permutation-invariant by construction. |
| Temporal backbone | 3 dilated Conv1d blocks (dilation 1, 2, 4) over the 30-frame sequence, each with a residual connection |
| Pooling and head | Mean-pool and max-pool over time, concatenated, then Linear(256 to 256) to BatchNorm to ReLU to Dropout(0.2) to Linear(256 to 128), L2-normalised |
| Classification head | ArcFace additive angular margin, scale 16, margin 0.25, applied only over the known subjects during training |

**Parameter count, measured directly from the instantiated model:** the frame encoder plus
temporal backbone plus embedding head totals **258,240 parameters**. The ArcFace head adds one
small weight matrix per known class; for 6 known classes this is **768 parameters**, for a
combined total of **259,008 parameters**. The saved checkpoint (network weights only, seed 0)
is **1,058,373 bytes** on disk (float32, includes optimizer-irrelevant buffers such as BatchNorm
running statistics).

Training: AdamW, learning rate 2e-3 with a one-cycle schedule, weight decay 1e-4, batch size 32,
20 epochs, no early stopping or best-checkpoint selection (the final epoch's weights are used
regardless of intermediate validation accuracy).

## 4. Closed-Set Results

Closed-set accuracy answers a narrower question than open-set identification: given that a test
window belongs to one of the 6 enrolled subjects, does the model pick the right one. Averaged
over 3 independent random draws of which 6 subjects are known:

| Windows voted (k) | Seconds | Closed-set accuracy | Unknown-detection AUROC |
|---|---|---|---|
| 1 | 3 | 0.968 | 0.795 |
| 2 | 4 | 0.979 | 0.849 |
| 4 | 6 | 0.991 | 0.911 |
| 6 | 8 | 0.993 | 0.937 |
| 8 | 10 | 0.998 | 0.953 |

Closed-set accuracy is close to its ceiling after a single 3-second window. Unknown-detection
AUROC, which measures whether a genuine stranger's score is separable from an enrolled person's
score at all (no threshold involved), keeps improving through 10 seconds of voting. This
asymmetry, that identifying who someone is takes far less evidence than deciding whether they are
anyone you know at all, is the central empirical fact this report's open-set results are built
around.

## 5. Open-Set Identification Without Impostor Data

The harder and more practically relevant task: decide whether a person is one of the 6 enrolled
subjects, or should be rejected as a stranger, using a threshold set from enrolled subjects'
scores only. Five calibration rules were compared, one of which is disqualified from the start
because it cheats:

| Method | Uses impostor data? | Idea |
|---|---|---|
| Leave-one-known-out (LOKO) | No | Hold one enrolled subject out of the gallery, treat their score as a stand-in impostor |
| EVT Weibull tail | No | Peaks-over-threshold extreme value fit [2] to the low tail of each class's genuine scores |
| 5th-percentile | No | Plain empirical quantile of validation genuine scores |
| Split-conformal | No | Same as the percentile, but with a finite-sample correction [3][4] that gives a provable bound on the false-rejection rate instead of a point estimate |
| Oracle | Yes, cheats | Sets the threshold by Youden's J statistic [5] using real held-out unknown subjects; upper bound only, not deployable |

Macro-F1 over 7 classes (6 known plus "unknown"), the unknown test set subsampled to match the
known test count so the metric is not dominated by it, mean over 3 seeds:

<p align="center"><img src="figures/openset_results.png" width="850" alt="Open-set macro F1 versus temporal voting across five calibration methods"></p>

| k | Closed acc | AUROC | LOKO | EVT | 5th pct | Split-conformal | Oracle |
|---|---|---|---|---|---|---|---|
| 1 | 0.968 | 0.795 | 0.662 | 0.739 | 0.736 | 0.732 | 0.762 |
| 2 | 0.979 | 0.849 | 0.674 | 0.780 | 0.776 | 0.767 | 0.795 |
| 4 | 0.991 | 0.911 | 0.744 | 0.835 | 0.831 | 0.828 | 0.854 |
| 6 | 0.993 | 0.937 | 0.800 | 0.858 | 0.860 | 0.860 | 0.886 |
| 8 | 0.998 | 0.953 | 0.824 | 0.868 | 0.868 | 0.850 | 0.894 |

Operating points at k = 8 (staff correctly kept, intruders correctly rejected):

| Method | Staff kept | Intruders rejected |
|---|---|---|
| LOKO | 0.985 | 0.544 |
| EVT | 0.881 | 0.836 |
| 5th percentile | 0.881 | 0.836 |
| Split-conformal | 0.927 | 0.669 |
| Oracle | 0.893 | 0.894 |

**Impostor-free calibration costs little.** The gap between the 5th-percentile rule and the
cheating oracle is 0.026 macro-F1 at k = 8. The whole premise that open-set radar identification
requires labelled intruders to set a working threshold does not survive contact with this data.

## 6. Validation of the Calibration Methods

A calibration number is only useful once it has been checked against how it was supposed to
behave, not just reported. Three checks were run.

### 6.1 EVT Provides No Measurable Benefit Over the Plain Percentile

EVT and the 5th-percentile method matched to three decimal places at every value of k in Table 5,
both in the main experiment and in the cross-scenario experiment in Section 7. The Weibull
peaks-over-threshold machinery [2], the basis of the OpenMax method, adds a fitting procedure
with its own failure modes (both `evt_threshold` and `percentile_threshold` fall back to the
plain quantile when a class has fewer than 30 validation windows or fewer than 10 tail
exceedances) and earns nothing measurable on this data.

### 6.2 Leave-One-Known-Out Fails in the Dangerous Direction

LOKO looks the best of the impostor-free methods by staff-kept rate (0.985 at k = 8) and the
worst by intruder-rejection rate (0.544). The reason is structural: the ArcFace loss explicitly
trains the embedding to push enrolled subjects apart from one another, so a held-out enrolled
subject's score is not a stand-in for a real stranger, it is a stand-in for someone the network
was specifically taught to separate. The pseudo-impostor score lands lower than a genuine
stranger's would, so the threshold LOKO produces is too permissive. It also has the largest
seed-to-seed standard deviation of any method in Table 5 (0.032 to 0.037 versus 0.009 to 0.024
for the other impostor-free methods), consistent with it depending on whichever single subject
happens to be held out.

### 6.3 The Plain Percentile Overshoots Its Own Target; Split-Conformal Does Not

The percentile method targets a 5% false-rejection rate by construction, but that 5% is a point
estimate from a finite validation set. Measured directly on the seed-0 `free_walk`-holdout
validation set (1,048 windows, single 3-second windows, not the k = 8 voted operating point used
elsewhere in this report): the plain percentile threshold rejected **5.3%** of the genuine
validation scores it was itself fit on, overshooting its own 5% target before ever being tested on
new data. The split-conformal threshold, evaluated the same way, rejected **4.3%**, within its
provable bound.

On the fully held-out, genuinely unseen `known_shifted` set from the same `free_walk` run (3,493
single-window scores), both methods' false-rejection rates rise sharply under the walking-mode
shift: **25.5%** for the plain percentile and **23.9%** for split-conformal, consistent with
`free_walk` being the one holdout where calibration transfer fails outright (Section 7). On the
`smartphone` holdout, the easiest of the three to transfer to, the same measurement at single-
window resolution gives **4.76%** for the plain percentile and **3.80%** for split-conformal,
both still close to target. The two holdouts are reported separately here because averaging them
would hide exactly the finding Section 7 is built on: calibration transfer quality depends heavily
on which walking mode is held out, not on a fixed property of either impostor-free method.

## 7. Cross-Scenario Robustness

The main experiment calibrates and tests within the same walking condition. A more realistic
question for a hospital: does a threshold calibrated on how someone usually walks still work when
their gait changes because they are carrying a chart, wearing PPE, or on a phone call. The
cross-scenario experiment trains and calibrates on two of the three walking modes, then tests on
both those seen modes and the third, unseen mode, one held-out mode per run.

<p align="center"><img src="figures/xscenario_results.png" width="850" alt="Cross-scenario open-set F1, seen versus unseen walking mode, aggregated"></p>

<p align="center"><img src="figures/report_holdout_bar.png" width="700" alt="Bar chart of staff-kept rate by held-out walking mode, seen versus unseen, percentile versus conformal"></p>

Staff-kept rate (known recall) at k = 8, per held-out mode:

| Holdout | Seen modes, 5th pct | Seen modes, conformal | **Unseen mode, 5th pct** | **Unseen mode, conformal** |
|---|---|---|---|---|
| free_walk | 0.842 | 0.883 | **0.393** | **0.513** |
| hands_in_pockets | 0.941 | 0.983 | 0.863 | 0.948 |
| smartphone | 0.910 | 0.973 | 0.968 | 0.995 |

Holding out `free_walk`, that is, calibrating a threshold on people walking with hands in pockets
or on a phone and then testing on people walking normally, collapses the plain percentile
threshold to **39.3%** staff kept. The embedding underneath barely notices this shift: unknown
detection AUROC only drops from 0.936 (seen modes, aggregated) to 0.884 (unseen mode, aggregated)
across all three holdouts. The failure is in the frozen threshold, not in the model's ability to
represent gait.

The split-conformal threshold recovers a real fraction of this specific failure (39.3% to 51.3%
staff kept on the unseen `free_walk` mode) because it is built to not overshoot its own
false-rejection target on a finite calibration set, which makes it somewhat less brittle when
that calibration set's conditions do not match test time. It does not fix the underlying problem;
a threshold calibrated on two walking modes still has no direct evidence about a third.

Cross-scenario track and window counts, `free_walk` holdout (the case above):

| Split | Tracks | Windows |
|---|---|---|
| Train (seen modes) | 87 | 4,783 |
| Validation (seen modes) | 19 | 1,048 |
| Known, in-domain test | 18 | 976 |
| Known, shifted (unseen mode) test | 64 | 3,493 |
| Unknown, in-domain test | 82 | 4,419 |
| Unknown, shifted (unseen mode) test | 44 | 2,295 |

One seed was run per holdout, not the 3-seed repetition used for the main experiment; the
magnitude of the `free_walk` failure should be read as established but not precisely bounded.

## 8. Limitations

**Sample size.** 10 subjects, 6 to 8 enrolled depending on experiment, is the binding constraint
on every number in this report. The 3-seed standard deviations in Table 5 (up to 0.037 macro-F1)
and the single-seed cross-scenario results in Section 7 should both be read with this in mind.

**Not comparable to the original mmGait10 paper's reported result.** Different openness (13.4%
here versus 21.6% under the formula that paper uses), a 64-point-per-frame budget instead of 150,
a balanced unknown test set, and 3 splits instead of 5. This report's numbers demonstrate that the
four design choices in Section 1 hold together, not that they outperform a different protocol.

**One room, one radar, one day.** The cross-scenario experiment in Section 7 tests a change in
walking mode, not a change in room, day, or radar hardware. Cross-session and cross-room
generalisation, and transfer to a single-chip radar with far fewer points per frame than this
cascade unit, are both untested.

**No embedding-level confound probe was run.** Section 1.1 measures the shortcut on raw metadata
before any model is trained, and the point-count fix removes that specific shortcut by
construction. Whether the trained 128-dimensional embedding itself still separates subjects
partly by body-size proxies such as x or z spread, the way an anthropometric linear probe would
check, was not tested in this project. This is a concrete, specific gap, not a claim that the
issue does or does not exist.

**No model-selection or noise-robustness testing.** Training always uses the final epoch's
weights, with no early stopping on validation performance. No test-time signal degradation
(added noise, reduced point count at inference rather than at training time) was evaluated.

**Six to eight enrolled people is a tiny gallery.** A real ward has dozens of staff; cosine-margin
separation gets harder as the gallery grows, and the thresholds reported here were not tested at
that scale.

**One person at a time, clear line of sight.** Point cloud input was chosen partly because it is
the representation that could in principle support multiple simultaneous people in a room;
mmGait10 does not contain multi-person recordings, so that capability is unexercised here.

## References

Listed only where a specific method, dataset, or figure in this report was taken from or checked
against the cited work.

1. Mazzieri, R., Pegoraro, J., Rossi, M. Open-Set Gait Recognition from Sparse mmWave Radar Point
   Clouds. IEEE Sensors Journal (2025). DOI 10.1109/JSEN.2025.3587503. Source of the mmGait10
   dataset used throughout this report.
2. Bendale, A., Boult, T. E. Towards Open Set Deep Networks. Proceedings of the IEEE Conference
   on Computer Vision and Pattern Recognition, CVPR (2016). Basis of the OpenMax-style
   peaks-over-threshold Weibull tail fit used in Section 5 and evaluated in Section 6.1.
3. Vovk, V., Gammerman, A., Shafer, G. Algorithmic Learning in a Random World. Springer (2005).
   Foundational exchangeability argument behind the split-conformal threshold in Section 5.
4. Bates, S., Candes, E., Lei, L., Romano, Y., Sesia, M. Testing for outliers with conformal
   p-values. Annals of Statistics 51(1), 149 to 178 (2023). One-sided novelty-detection form of
   split-conformal calibration used in Section 5.
5. Youden, W. J. Index for rating diagnostic tests. Cancer 3(1), 32 to 35 (1950). Basis of the
   ROC threshold selection used by the leave-one-known-out and oracle methods in Section 5.
6. Deng, J., Guo, J., Xue, N., Zafeiriou, S. ArcFace: Additive Angular Margin Loss for Deep Face
   Recognition. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
   Recognition, CVPR (2019). Basis of the classification head in Section 3.
7. Qi, C. R., Su, H., Mo, K., Guibas, L. J. PointNet: Deep Learning on Point Sets for 3D
   Classification and Segmentation. Proceedings of the IEEE Conference on Computer Vision and
   Pattern Recognition, CVPR (2017). Basis of the per-frame point-cloud encoder in Section 3.
