"""Turn the raw mmGait10 point clouds into one compact array for training.

Each frame is forced to exactly NP points, sampled without replacement when the
frame has more and padded with repeats when it has fewer, then centred on its own
mean in all four features. Fixing the point count removes the body-size shortcut
documented in ANALYSIS.md, and centring removes absolute position and bulk speed.

Reads  : <project>/mmGait10/point_clouds/target*/<scenario>/pc_*.obj
Writes : <project>/prepared_64pts.npz
"""
import argparse, glob, os, pickle
import numpy as np

import paths

P = argparse.ArgumentParser()
P.add_argument("--points", type=int, default=64,
               help="points kept per frame (64 is closer to a single-chip ward sensor)")
P.add_argument("--root", default=None, help="override the point_clouds folder")
P.add_argument("--out", default=None, help="override the output .npz")
A = P.parse_args()

ROOT = paths.require(A.root or paths.POINT_CLOUDS, "the point_clouds folder")
OUT = A.out or (paths.PROJECT / f"prepared_{A.points}pts.npz")
NP = A.points
rng = np.random.default_rng(0)

frames, meta, offs = [], [], [0]
bad, retried = [], []
for sub in sorted(os.listdir(ROOT), key=lambda s: int(s[6:])):
    lab = int(sub[6:])
    for sc in sorted(os.listdir(os.path.join(ROOT, sub))):
        for f in sorted(glob.glob(os.path.join(ROOT, sub, sc, "*.obj"))):
            # Reads over a network drive, synced folder or mounted volume can come
            # back short. Retry rather than skip, or tracks vanish silently.
            d = None
            for attempt in range(6):
                try:
                    d = pickle.load(open(f, "rb")); break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            if d is None:
                bad.append((os.path.relpath(f, ROOT), err)); continue
            if attempt:
                retried.append((os.path.relpath(f, ROOT), attempt))

            arr = np.empty((len(d), NP, 4), np.float32)
            for i, fr in enumerate(d):
                n = int(fr["cardinality"][0])
                M = np.concatenate([fr["elements"],
                                    fr["z_coord"][:, None],
                                    fr["dopplers"][:, None]], 1)[:n]
                if n >= NP:
                    idx = rng.choice(n, NP, replace=False)
                else:
                    idx = np.concatenate([np.arange(n), rng.choice(n, NP - n, replace=True)])
                F = M[idx]
                arr[i] = F - F.mean(0, keepdims=True)
            frames.append(arr.astype(np.float16))
            offs.append(offs[-1] + len(d))
            meta.append((lab, sc, os.path.basename(f)))

X = np.concatenate(frames, 0)
np.savez_compressed(OUT, X=X, offsets=np.array(offs, np.int64),
                    labels=np.array([m[0] for m in meta], np.int64),
                    scenarios=np.array([m[1] for m in meta]),
                    tracks=np.array([m[2] for m in meta]))
print("files needing a retry:", retried)
print("unreadable after 6 tries:", bad)
print(f"tracks {len(meta)}  frames {X.shape}  -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
