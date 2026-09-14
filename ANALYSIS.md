# mmGait10: data profile, what the paper did, and where the gaps are

Analysis date: 2026-09-08, corrected 2026-09-10
Source: Zenodo record 15784474, `mmGait10_v2.zip`, MD5 `78e71838c5f0b437e79c038e66bfbcc8` (verified)
Reference code: https://github.com/rmazzier/OpenSetGaitRecognition_PCAA
Paper: Mazzieri, Pegoraro, Rossi, "Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds", IEEE Sensors Journal 2025, DOI 10.1109/JSEN.2025.3587503

## 1. What is in the folder

```
mmGait10/
  point_clouds/target{0..9}/{free_walk,hands_in_pockets,smartphone}/pc_<subj>_<track>.obj   314 files
  spectrograms/target{0..9}/{free_walk,hands_in_pockets,smartphone}/md_<subj>_<track>.pt    397 files
mmGait10_v2.zip          715.7 MB, kept for reference
mmGait10_profile.png     three summary plots
pc_profile.json          per track statistics
```

The `.obj` files are Python pickles, not meshes. Each is a list of per frame dicts with keys
`cardinality, elements (x,y), z_coord, dopplers, powers, center, velocity, birth_step`.
The `.pt` files are single float32 tensors of shape 128 x T, values scaled to [0,1].

## 2. Data profile

Point clouds, all 10 subjects:

| quantity | value |
|---|---|
| tracks | 314 (all readable) |
| frames | 178,372 at 10 Hz = 297.3 min = 4.95 h |
| points total | 31.7 M |
| points per frame | mean 178, median 173, p5 103, p95 269, min 1, max 558 |
| frames below the NMAX=150 cap used by PCAA | 31.1% |
| frames below 50 points | 0.46% |
| track length | median 597 frames (59.7 s), min 101, most tracks are exactly 597 |
| 30-frame crops at step 6 | 28,313 |
| x range | -3.93 to 3.76 m |
| y range | 0 to 6.27 m |
| z range | -2.26 to 2.36 m |
| Doppler | -2.68 to 2.64 m/s |
| power | 0 to 9.74e6 linear, mean 5809, hence the dB conversion in the loader |

Per subject the set is well balanced in frames and in the three walking modes:

| subject | tracks | frames | minutes | mean points/frame |
|---|---|---|---|---|
| 0 | 30 | 15,747 | 26.2 | 183.2 |
| 1 | 30 | 17,007 | 28.3 | 182.4 |
| 2 | 31 | 17,695 | 29.5 | 176.5 |
| 3 | 32 | 18,161 | 30.3 | 174.5 |
| 4 | 30 | 16,753 | 27.9 | 156.0 |
| 5 | 34 | 19,056 | 31.8 | 189.6 |
| 6 | 32 | 18,637 | 31.1 | 135.2 |
| 7 | 30 | 17,839 | 29.7 | 187.8 |
| 8 | 32 | 18,609 | 31.0 | 214.9 |
| 9 | 33 | 18,868 | 31.4 | 177.0 |

Spectrograms: 397 files, 181,973 time bins = 303.3 min, 128 Doppler bins, values in [0,1],
median value 0 and mean 0.025, so they are very sparse. 64 files are shorter than 60 bins
and 36 are shorter than 30 bins.

Modality alignment: every one of the 314 point cloud tracks has a spectrogram with the same
subject and track id. There are 83 spectrogram-only tracks with no point cloud, mostly the
very short ones. So the two modalities are pairable on the point cloud subset, and 21% of the
spectrogram data is never used by the released point cloud pipeline.

## 3. What the paper did and what it achieved

Method, PCAA (Point Cloud Adversarial Autoencoder):

- Per frame PointNet encoder, then temporal dilated convolutions over a 30 frame window (3 s).
- Three heads: a supervised identity classifier, a decoder that reconstructs the point cloud
  sequence, and a discriminator that pushes the 32-dim latent onto a mixture of Gaussians whose
  means are fixed, one per known subject, placed deterministically rather than learned.
- Input features are x, y, z, Doppler. Power is computed then dropped (`NFEATURES = 4`).
- Each frame is padded or subsampled to exactly 150 points and standardised by subtracting the
  per frame mean, so absolute position is removed.

Open set protocol:

- Openness swept from about 3% to 40% by varying how many of the 10 subjects are known.
- 5 random known/unknown splits per openness level, macro F1 as the metric.
- At inference, the joint likelihood under the Gaussian mixture is thresholded, and k consecutive
  windows (k in 1, 2, 4, 6) vote by majority. The threshold is chosen by Youden's J on an ROC
  computed against a held out unknown subject.

Results: about 0.587 macro F1 at 21.6% openness with k=6, against roughly 0.25 to 0.30 for the
adapted OR-CED and L-GM baselines, reported as a 24% average F1 improvement. Ablations show
performance flattens above about 130 points per frame, and that the smartphone walking mode is
slightly easier to identify than free walk or hands in pockets.

Split hygiene is sound. Splits are made at track level before cropping, so overlapping windows
never straddle train and test.

## 4. Problems found in the data

1. **No corrupt files, but reads over a mounted folder are flaky.** An earlier pass through this
   data reported `pc_9_038.obj` as truncated. It is not. Re-reading it succeeds, and the bytes on
   disk match the zip entry exactly. Three files in `target9` needed one retry each on a later pass,
   and it was a different file each time. If your loader reads the data from a network drive, a
   synced folder or a mounted volume, wrap `pickle.load` in a retry rather than skipping the file,
   or you will silently drop tracks and never know. All 314 tracks are intact.
2. **The data is not sparse in the way the title suggests.** Median 173 points per frame is dense
   for a single walking person. The 150 point cap actually throws points away in 69% of frames and
   pads with duplicates in the other 31%. Calling this sparse invites a wrong comparison against
   IWR6843-class single chip radars, which typically give 20 to 60 points per frame on a person.
3. **Point density is subject correlated.** Mean points per frame runs from 135 (subject 6) to 215
   (subject 8). A nearest centroid classifier trained on nothing but seven cheap summary statistics
   per 3 s window (point count mean and std, mean and std power in dB, mean absolute Doppler, and
   the spread of z and x) reaches **29.3% accuracy across 10 subjects with 5-fold grouping by
   track, against 10% chance**. Point count alone gives 24.4%. None of that is gait. It is body
   size, radar cross section and range geometry. The paper reports no such floor.
4. **Single room, single radar, single session.** All data is one 7.8 x 7.3 m room with one cascade
   radar. There is no cross-room, cross-day or cross-radar split, so nothing in the reported numbers
   speaks to generalisation, which is what actually breaks radar identification in deployment.
5. **Ten subjects.** At 40% openness that is 6 known and 4 unknown. Split-to-split variance will be
   large, and 5 random splits is thin for a claim about openness scaling.
6. **Cascade radar, not a deployable sensor.** MMWCAS-RF-EVM is a 4-chip cascade with far better
   angular resolution than an IWR6843. Results do not transfer directly to a single chip device.

## 5. Problems found in the code

1. **The subsampling ablation does not subsample randomly.** In `datasets.py`,
   `MSRadarDataset.process_track`:

   ```python
   if force_pc_subsampling > 0 and force_pc_subsampling < frame_cardinality:
       frame_cardinality = force_pc_subsampling                     # overwritten first
       choices = rng.choice(frame_cardinality, force_pc_subsampling, replace=False)
   ```

   `frame_cardinality` is reassigned before it is used, so `choices` is a permutation of
   `0..force_pc_subsampling-1`. The code keeps the first N points in file order rather than a random
   subset. Figure 4 of the paper therefore measures robustness to "keep the first N detections",
   not to genuine point sparsity. Since detection order out of the radar processing chain is usually
   sorted by power or by range, this is a biased subset. Fixing this will very likely lower the
   flat part of that curve.
2. **Seeding is inconsistent.** `process_track` takes an `rng` argument and then immediately
   overwrites it with `np.random.default_rng(0)`, while the padding path uses the unseeded global
   `np.random.choice`. Dataset generation is not reproducible for the padding step.
3. **The threshold needs a real unknown subject.** `inference_PCAA.py` picks the operating point by
   maximising `tpr - fpr` on an ROC built from a held out unknown class. That is a strong assumption:
   it requires labelled impostor data from the deployment site.
4. **Hardcoded paths and a `wandb` dependency.** `DATA_PATH` points at `../../radar_reid_pytorch/...`
   and logging defaults to `WANDB_MODE = "online"`. Both need changing before a first run.
5. **Absolute power is dropped.** Reasonable for avoiding an RCS shortcut, but it is not stated as a
   design choice and it is never ablated.

## 6. Gaps worth attacking, in order

**A. Publish the shortcut floor before any model number.** Report the metadata-only baseline
(29.3% here) alongside the open set F1. Any paper on this dataset that does not show this floor is
overstating how much of its accuracy comes from gait. This is a cheap, publishable contribution on
its own and it directly protects a clinician ID claim from the reviewer question "did it just learn
who is tall".

**B. Remove the density and RCS shortcut, then re-measure.** Two concrete controls: match the point
count per frame across subjects by fixed subsampling to a common floor, and normalise per frame by
range. If open set F1 holds up, the gait claim is real. If it drops, that is the finding.

**C. Fix the subsampling bug and redo Figure 4 with true random subsampling down to 20, 30 and 60
points.** That range is where a single chip IWR6843 actually lives. Right now nobody knows whether
PCAA works on a deployable sensor, and the published curve cannot answer it. This is the single most
useful experiment for a hospital deployment, since a ward radar will be a cheap single chip unit,
not a cascade.

**D. Threshold calibration without impostor data.** Replace Youden's J on a held out unknown with a
calibration that only uses known subjects: leave-one-known-out as pseudo-unknowns, or an extreme
value fit on the tail of known-class likelihoods (the Weibull approach from OpenMax). In a hospital
you can enrol staff but you cannot collect labelled intruders, so this is the difference between a
method that deploys and one that does not.

**E. Use the 83 spectrogram-only tracks and the pairing that already exists.** All 314 point cloud
tracks have a matched spectrogram. Nobody has published point cloud plus micro-Doppler fusion on
this dataset, and the paper's own ablation shows the two modalities carry different information
(the walking-mode ranking differs). A two-branch model with the spectrogram-only tracks used for
semi-supervised pretraining is a clean, low-risk contribution.

**F. Cross-condition splits.** The dataset has three walking modes. Train on two, test on the third,
and report open set F1. That is a domain shift the data can actually support and the paper never
runs it. It is the nearest available proxy for the gown, PPE and trolley variation you will hit in
a ward.

**G. Occlusion and multi-person, which this dataset cannot do.** mmGait10 is one person at a time in
clear line of sight. For the ward work, pair it with the Harbin occluded person ID set on IEEE
DataPort (22 subjects, three occluder types, 600k frames) or with people-gait for co-existing
walkers. Use mmGait10 to build and validate the open set machinery, then transfer.

## 7. Practical notes for a first run

- Point the loader at `Documents/mmGait10/mmGait10/point_clouds` and set `WANDB_MODE = "disabled"`.
- Wrap `pickle.load` in a short retry loop if the data sits on a mounted or synced folder. Do not
  use a bare `try/except ... continue`, which hides dropped tracks.
- Set `safe_mode=False` in `generate_splits` or it blocks on `input()`.
- The code targets Python 3.8 and an older PyTorch. Expect to pin versions.
- Full crop generation writes about 28k `.npy` files per split configuration, so keep the generated
  dataset on a local disk, not on a synced folder.
