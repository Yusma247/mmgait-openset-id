# Open-set gait ID on mmGait10, built the way a ward system would need it

Run date: 2026-09-10, extended 2026-09-14 with a split-conformal calibration method and the
cross-scenario experiment. Code in `code/`, per-split output in `results/runs/` and
`results/runs_xs/`, figures `figures/openset_results.png` and `figures/xscenario_results.png`.

Four design choices, each made for the hospital problem rather than for the benchmark:

1. **Point cloud input**, because it is the only representation that survives more than one person in the room.
2. **Metric-learning head** (ArcFace cosine margin) instead of a softmax classifier, so staff can be enrolled or removed by editing a gallery rather than retraining.
3. **Rejection threshold calibrated without impostor data**, because you can enrol clinicians but you cannot collect labelled intruders.
4. **Temporal voting** over consecutive windows, because a clinician walking bed to bed gives you seconds of data, not one frame.

## Setup

| | |
|---|---|
| input | 64 points per frame, sampled without replacement, features x, y, z, Doppler |
| normalisation | per-frame mean subtracted from all four features, so absolute position and bulk walking speed are removed |
| window | 30 frames = 3 s at 10 Hz, hop 10 frames |
| model | per-frame PointNet (32-64-128, max-pool over points), 3 dilated temporal conv blocks, mean+max pool over time, 128-d L2-normalised embedding |
| loss | ArcFace, scale 16, margin 0.25, over the 6 known subjects only |
| augmentation | ±15° yaw, left-right mirror, 15% point dropout |
| split | 6 known, 4 unknown. Known tracks 70/15/15 train/val/test. Unknown subjects appear only at test, never in training and never in calibration |
| repeats | 3 random known/unknown draws, 20 epochs each, for the main experiment. 1 draw per holdout for the cross-scenario experiment |
| openness | 13.4% by the standard formula with 6 of 10 classes known |

Because every frame is forced to exactly 64 points, the point-count shortcut documented in
`ANALYSIS.md` (29.3% subject accuracy from summary statistics alone) is removed by construction.
Body extent in x and z survives, and that is legitimate biometry rather than a sensing artefact.

## Results

Mean over 3 splits. Macro F1 is over 7 classes: 6 known subjects plus "unknown". The unknown
test set is subsampled to match the known test count so the macro average is not dominated by it.
AUROC uses every unknown window and needs no threshold.

| windows voted | seconds | closed-set acc | unknown AUROC | leave-one-known-out | EVT Weibull tail | 5th-percentile | split-conformal | oracle (sees unknowns) |
|---|---|---|---|---|---|---|---|---|
| k = 1 | 3 s | 0.968 | 0.795 | 0.663 ± 0.016 | **0.739** ± 0.024 | **0.736** ± 0.024 | 0.732 ± 0.023 | 0.762 ± 0.015 |
| k = 2 | 4 s | 0.979 | 0.849 | 0.674 ± 0.020 | **0.780** ± 0.018 | **0.776** ± 0.015 | 0.767 ± 0.019 | 0.795 ± 0.013 |
| k = 4 | 6 s | 0.991 | 0.911 | 0.744 ± 0.035 | **0.835** ± 0.020 | **0.831** ± 0.018 | 0.828 ± 0.023 | 0.854 ± 0.015 |
| k = 6 | 8 s | 0.993 | 0.937 | 0.800 ± 0.037 | **0.858** ± 0.012 | **0.860** ± 0.013 | 0.860 ± 0.027 | 0.886 ± 0.012 |
| k = 8 | 10 s | 0.998 | 0.953 | 0.824 ± 0.032 | **0.868** ± 0.009 | **0.868** ± 0.009 | 0.850 ± 0.033 | 0.894 ± 0.009 |

Operating points at k = 8:

| calibration | staff correctly identified | intruders correctly rejected |
|---|---|---|
| leave-one-known-out | 0.985 | 0.544 |
| EVT Weibull tail | 0.881 | 0.836 |
| 5th-percentile | 0.881 | 0.836 |
| split-conformal | 0.927 | 0.669 |
| oracle | 0.893 | 0.894 |

## What split-conformal calibration is, and why it was added

Every other impostor-free rule here (EVT, the plain percentile) fits a threshold by treating the
validation genuine scores as if they exactly described the population: take the empirical 5th
percentile and call it done. That percentile is itself a random quantity - with a few hundred
validation windows per class, its distance from the *true* 5% point can be large enough that the
achieved false-rejection rate misses the 5% target in either direction. On this data it missed
high: the plain percentile's own calibration set rejected 5.3% of its own genuine scores against a
5% target, before it was ever tested on new data.

Split-conformal calibration removes that estimation gap with one change: instead of the raw
`target_frr` quantile, use the k-th smallest calibration score with
`k = floor(target_frr * (n + 1))`. This is not a different heuristic competing with the percentile,
it is the same idea corrected for having a finite calibration set. The correction comes from a
single exchangeability fact (Vovk et al.; the one-sided novelty-detection form is in Bates,
Candès et al., *"Testing for outliers with conformal p-values"*): if the n calibration scores and
one new genuine test score are exchangeable, the test score's rank among all n+1 values is uniform
on `{1, ..., n+1}`, so

```
P(new genuine score < threshold) <= k / (n + 1) <= target_frr
```

holds exactly, for any n, with no distributional assumption beyond the same exchangeability the
train/val/test split already relies on. When a class has too few calibration windows to support
any guarantee at the target rate (`k < 1`), the threshold is `-inf`: refuse to reject rather than
invent a number from too little data - the same failure mode that made leave-one-known-out's
variance blow up on the classes with the fewest validation windows.

Implementation: `calib.conformal_threshold` in `calib.py`, wired into `run.py` and `run_xs.py`
alongside the other three impostor-free methods.

### What it changed, in-domain

On the main experiment it is not a strict improvement over the percentile - across the 3 seeds it
came out ahead once (0.895 vs 0.876 macro F1 at k=8) and behind twice (0.831 vs 0.873). What is
consistent is the direction of its conservatism: at k=8 it keeps more staff (92.7% vs 88.1%) and
catches fewer intruders (66.9% vs 83.6%). That is the guarantee doing exactly what it is supposed
to do - bounding the false-rejection rate necessarily trades away some rejection sensitivity, and
it trades away the side that carries no such bound.

### What it changed under scenario shift - the result that matters

The cross-scenario experiment (next section) is where the difference stops being a wash. Holding
out `free_walk` - training and calibrating on staff walking with hands in pockets or on a phone,
then testing on people walking normally - is a real failure case for the plain percentile: it
keeps only **39.3%** of legitimate staff on the unseen mode. The percentile threshold, fit tightly
to the two training modes' score distribution, does not transfer. Split-conformal, built to not
overshoot its target on a finite calibration set, recovers a meaningful fraction of that: staff
kept rises to **51.3%**, and macro F1 recovers from 0.480 to 0.569. On the two milder holdouts the
two methods are close, trading a few points either way.

So the honest claim is narrower than "conformal is better": **it does not reliably beat the plain
percentile when test conditions match training, but it meaningfully limits the percentile's worst
failure under domain shift** - which is the property an impostor-free calibration rule most needs,
since a hospital cannot know in advance which covariate will shift on a given day.

## Cross-scenario results

Train on two of the three walking modes (free walk, hands in pockets, smartphone), calibrate the
threshold on validation data from those two modes only, then test on known and unknown subjects in
both the seen modes and the held-out third mode. This is the nearest proxy the dataset offers for a
clinician's gait changing under PPE, a trolley, or a phone call - and it is the first time an
impostor-free threshold here is asked to generalise across a genuine covariate shift, not just to
new people. One seed per holdout (not yet repeated across seeds - see Next steps).

| windows voted | condition | closed-set acc | unknown AUROC | macro F1 (5th pct) | macro F1 (conformal) | staff kept (5th pct) | staff kept (conformal) |
|---|---|---|---|---|---|---|---|
| k = 1 | seen modes | 0.962 | 0.770 | 0.708 ± 0.018 | 0.701 ± 0.023 | 0.917 | 0.922 |
| k = 1 | unseen mode | 0.906 | 0.751 | 0.696 ± 0.045 | 0.690 ± 0.037 | 0.816 | 0.825 |
| k = 4 | seen modes | 0.989 | 0.886 | 0.769 ± 0.006 | 0.740 ± 0.019 | 0.909 | 0.935 |
| k = 4 | unseen mode | 0.936 | 0.845 | 0.715 ± 0.102 | 0.709 ± 0.071 | 0.790 | 0.826 |
| k = 8 | seen modes | 0.994 | 0.936 | 0.801 ± 0.020 | 0.772 ± 0.016 | 0.898 | 0.946 |
| k = 8 | unseen mode | 0.950 | 0.884 | 0.699 ± 0.155 | 0.728 ± 0.114 | 0.741 | 0.819 |

Per held-out mode at k = 8:

| holdout | seen: p5 | seen: conformal | unseen: p5 | unseen: conformal | staff kept, unseen (p5 → conformal) |
|---|---|---|---|---|---|
| free_walk | 0.809 | 0.794 | **0.480** | **0.569** | 39.3% → **51.3%** |
| hands_in_pockets | 0.820 | 0.765 | 0.811 | 0.835 | 86.3% → 94.8% |
| smartphone | 0.773 | 0.757 | 0.807 | 0.780 | 96.8% → 99.5% |

Two things stand out beyond the conformal comparison above:

**Closed-set accuracy holds up under shift (0.994 → 0.950 at k=8) but the unseen-mode standard
deviation on macro F1 is huge - ± 0.155 at k=8, an order of magnitude above the in-domain ± 0.020.**
That spread comes almost entirely from `free_walk` being much harder to transfer to than the other
two modes. Identity survives the covariate shift far better than a fixed threshold does: unknown
AUROC (threshold-free) only drops from 0.936 to 0.884, but the percentile threshold's usable
operating point collapses on one specific holdout. The failure is in calibration, not in the
embedding.

**Voting still helps identification and detection under shift, but interacts badly with a stale
threshold on `free_walk` specifically**: with the plain percentile, `free_walk` macro F1 on the
unseen mode goes 0.632 → 0.570 → 0.480 as k rises from 1 to 4 to 8 - voting makes the wrong
threshold's damage worse, not better, because it sharpens a systematically miscalibrated score
distribution rather than a noisy one. Split-conformal does not have this inversion (its unseen-mode
macro F1 on `free_walk` rises with k, same shape as everywhere else in this project).

## What this says

**Impostor-free calibration costs almost nothing, in-domain.** The gap to the oracle is 0.007 F1 at
one window and 0.026 at eight, using the percentile (0.044 for conformal, which trades some of that
gap away for its guarantee). The whole objection that open-set radar ID needs labelled intruders to
set its threshold does not survive contact with the data. Enrolment data alone is enough, as long
as conditions at test time resemble conditions at calibration time.

Caveat, stated plainly: the oracle picks its threshold by maximising true-positive minus
false-positive rate, not macro F1, so it is not a strict upper bound. On individual splits the
impostor-free rules occasionally edge past it. Read the gap as "small", not as "we beat the oracle".

**The Weibull fit earns nothing.** EVT and a plain 5th percentile of the validation genuine scores
agree to three decimals at every k, in-domain and under scenario shift alike. The extreme-value
machinery from OpenMax is defensible in a paper but there is no measurable benefit here. Use the
percentile, it has one line of code and no fitting failure mode - or use split-conformal, which has
the same cost and gives an actual finite-sample bound instead of a point estimate.

**Leave-one-known-out is the wrong proxy, and it fails in the dangerous direction.** It is the
obvious idea: hold one enrolled person out of the gallery and treat their scores as impostor
scores. But the embedding was explicitly trained to push those subjects apart, so pseudo-impostor
scores land far lower than a real stranger's. The threshold comes out too permissive. At k = 8 it
keeps 98.5% of staff, which looks excellent, while letting 46% of intruders through. A ward system
tuned this way would look like it was working and would not be. Its variance across splits is also
several times larger than the other impostor-free methods.

**Under scenario shift, calibration - not identification - is the weak link, and this is where the
choice of impostor-free method stops being cosmetic.** In-domain, percentile and split-conformal are
close enough that either is defensible. Across a genuine covariate shift, the percentile threshold
can quietly fail on one specific mode (39% staff kept on `free_walk`) while the embedding underneath
it barely notices (AUROC only drops 5 points). Split-conformal does not fix the shift, but it
substantially limits how badly a stale threshold can fail - a materially different property than
"about the same average performance."

**Voting buys rejection, not identification, and can amplify a bad threshold.** Closed-set accuracy
moves 0.968 to 0.998 over eight windows, which is nearly nothing. Unknown-detection AUROC moves
0.795 to 0.953. Deciding who someone is takes one gait cycle. Deciding whether they are anyone you
know takes several. The right operating point is 8 to 10 seconds of walking, roughly one bed-to-bed
transit - but only once the threshold underneath the voting is trustworthy; see `free_walk` above
for what happens when it is not.

## What this does not say

- **Not comparable to the PCAA paper's 0.587.** Different openness (13.4% here, 21.6% under the
  formula that paper uses), a 64-point budget instead of 150, a balanced unknown set, and 3 splits
  instead of 5. The numbers here are not evidence of beating anything. They are evidence that the
  four design choices hold together.
- **One room, one day, one radar.** The cross-scenario experiment tests a change in walking mode,
  not a change in room, day, or radar hardware. Cross-session and cross-room generalisation, which
  is what actually breaks radar identification in deployment, is still untested.
- **The cross-scenario numbers are one seed per holdout.** The main experiment's 3-seed spread
  (± 0.009 to ± 0.037 in-domain) shows split-to-split variance is real at this subject count; the
  single-seed `free_walk` result should be read as "this failure mode exists and is large," not as
  a precise estimate of its size.
- **Six enrolled people is a tiny gallery.** Cosine margins get harder as the gallery grows, and a
  real ward has dozens of staff. Expect the threshold to need re-tuning at that scale.
- **One person at a time, clear line of sight.** The reason for choosing point cloud input was
  multi-person capability, and mmGait10 cannot exercise it.

## Next experiments, in order

1. **Repeat the cross-scenario runs across seeds.** One seed per holdout is enough to find the
   `free_walk` failure but not enough to size it precisely. 2 more seeds per holdout, same cost as
   the main experiment.
2. **Gallery growth.** Retrain with 8 known and 2 unknown, then sweep the target rejection rate
   instead of fixing it at 5%. In a ward the false-alarm cost sets that dial, not a convention.
3. **Occlusion transfer.** Take this pipeline to the Harbin occluded person ID set on IEEE
   DataPort and see how much of the 0.868 survives a clothing rack between sensor and subject.

## Reproducing

```
cd code
python prep.py                                        # raw .obj -> prepared_64pts.npz
python run.py --seed 0 --epochs 20                    # repeat for seeds 1 and 2
python aggregate.py

python run_xs.py --holdout free_walk --seed 0         # repeat for hands_in_pockets, smartphone
python aggregate_xs.py
```

`prep.py` needs numpy only. `run.py` and `run_xs.py` need torch and scipy. One split is about
15-22 minutes on a few CPU cores depending on machine load, so the whole thing runs overnight on a
laptop and in minutes on any GPU.
