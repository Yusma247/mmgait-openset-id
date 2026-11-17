"""Zero-shot check: do mmGait10-trained embeddings separate same- from different-
person people-gait volunteers, people the ArcFace head was never trained to name?

The trained head only has 6 classes (the known mmGait10 subjects), so it cannot
score people-gait's volunteers directly. What transfers is the embedding space
itself: GaitNet(x) before the ArcFace layer is a generic L2-normalised gait
embedding. This computes one embedding per people-gait track (mean-pooled over its
3 s windows), then checks whether cosine similarity separates same-volunteer track
pairs from different-volunteer track pairs - the same signal the open-set
thresholds in calib.py are calibrated against, just on a dataset the model never
saw a single frame of.

Reads  : <project>/prepared_peoplegait_64pts.npz, results/runs/model_seed{SEED}_k6.pt
Writes : results/pg_verify_seed{SEED}.json
"""
import argparse, json, os
import numpy as np
import torch

import calib
import data as D_
import paths
from model import GaitNet

P = argparse.ArgumentParser()
P.add_argument("--npz", default=str(paths.PG_PREPARED))
P.add_argument("--seed", type=int, default=0)
P.add_argument("--emb", type=int, default=128)
A = P.parse_args()

ckpt_path = paths.RUNS / f"model_seed{A.seed}_k6.pt"
paths.require(A.npz, "prepared_peoplegait_64pts.npz (run pg_prep.py first)")
paths.require(ckpt_path, "a trained mmGait10 checkpoint (run run.py first)")

D = D_.load(A.npz)
net = GaitNet(emb_dim=A.emb)
net.load_state_dict(torch.load(ckpt_path, map_location="cpu")["net"])
net.eval()

n_tracks = len(D["offsets"]) - 1
win = D_.build_windows(D, np.arange(n_tracks))
print(f"[pg_verify] {n_tracks} tracks, {len(win)} windows of {D_.NSTEPS} frames")

loader = torch.utils.data.DataLoader(
    D_.WindowSet(D, win, label_map=None, train=False), batch_size=64, shuffle=False)

sums = np.zeros((n_tracks, A.emb), np.float64)
counts = np.zeros(n_tracks, np.int64)
with torch.no_grad():
    for x, _, t, _ in loader:
        z = net(x).numpy()
        for zi, ti in zip(z, t.numpy()):
            sums[ti] += zi
            counts[ti] += 1

has_windows = counts > 0
track_emb = sums[has_windows] / counts[has_windows, None]
track_emb /= np.linalg.norm(track_emb, axis=1, keepdims=True)
track_lab = D["labels"][has_windows]
track_meta = D["tracks"][has_windows]
print(f"[pg_verify] {has_windows.sum()}/{n_tracks} tracks had >=1 full 3 s window "
      f"(skipped: {list(D['tracks'][~has_windows])})")

S = track_emb @ track_emb.T
iu, ju = np.triu_indices(len(track_emb), k=1)
same = track_lab[iu] == track_lab[ju]
pos, neg = S[iu, ju][same], S[iu, ju][~same]

auroc = calib.auroc(pos, neg)
print(f"[pg_verify] track pairs: {same.sum()} same-person, {(~same).sum()} different-person")
print(f"[pg_verify] same-person cosine mean {pos.mean():.3f}  "
      f"different-person cosine mean {neg.mean():.3f}")
print(f"[pg_verify] same- vs different-person AUROC on {len(set(track_lab))} "
      f"unseen people-gait volunteers: {auroc:.3f}")

out = dict(seed=A.seed, n_tracks=int(has_windows.sum()), n_volunteers=int(len(set(track_lab))),
          n_same_pairs=int(same.sum()), n_diff_pairs=int((~same).sum()),
          same_cos_mean=float(pos.mean()), diff_cos_mean=float(neg.mean()), auroc=auroc)
json.dump(out, open(paths.RESULTS / f"pg_verify_seed{A.seed}.json", "w"), indent=1)
