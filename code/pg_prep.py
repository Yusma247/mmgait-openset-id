"""Convert the single-person people-gait CSVs into the mmGait10 track format.

people-gait (Meng et al., AAAI 2020, github.com/mmGait/people-gait) is organised as
room{1,2}/<n_people>/{fixed_route,free_route}/{60Ghz,77Ghz}_radar/<id>.csv, where
<n_people> is how many volunteers co-walk in that recording and <id> is the global
volunteer number (multi-person files concatenate ids, e.g. 101_105_106_107.csv).
Only <n_people> == 1 is read here - multi-person separation needs DBSCAN + tracking,
which is a separate step.

Each csv is one continuous recording for one volunteer: columns are
Frame #, # Obj, X, Y, Z, Doppler, Intensity, y, m, d, h, m, s (header row present).
Measured from the timestamps: ~10 Hz native frame rate (median inter-frame gap
0.098-0.102 s across both radar bands and both rooms), close enough to the mmGait10
target rate that frames are used directly, in Frame # order, without resampling.
Points per frame range 1-64 (same 64-point hardware ceiling as mmGait10), so the
same fixed-size sampling scheme applies.

Only the 77Ghz_radar band is kept by default - closer to mmGait10's 77-81GHz cascade
than 60Ghz, and keeping both would double-count every walk (both bands record the
same simultaneous walk; issue #5 on the upstream repo notes the inter-radar offset
needed to merge them isn't known, so they are kept as separate, unmerged devices
rather than combined into one point cloud).

Reads  : <project>/people-gait-master/room{1,2}/1/*/<device>/*.csv
Writes : <project>/prepared_peoplegait_64pts.npz (X, offsets, labels, scenarios, tracks)
"""
import argparse, os
import numpy as np

import paths

P = argparse.ArgumentParser()
P.add_argument("--points", type=int, default=64)
P.add_argument("--root", default=None, help="override the people-gait-master folder")
P.add_argument("--out", default=None, help="override the output .npz")
P.add_argument("--devices", default="77Ghz_radar",
               help="comma-separated radar folders to read, e.g. 77Ghz_radar,60Ghz_radar")
A = P.parse_args()

ROOT = paths.require(A.root or paths.PEOPLE_GAIT, "the people-gait-master folder")
OUT = A.out or paths.PG_PREPARED
NP = A.points
DEVICES = A.devices.split(",")
ROOMS = ["room1", "room2"]
ROUTES = ["fixed_route", "free_route"]
rng = np.random.default_rng(0)


MAX_RANGE_M = 10.0    # room is a few metres across; the TI radar's own max range is ~10 m
MAX_SPEED_MS = 8.0    # a walking/turning human stays well under this radial speed


def load_track(path):
    """One csv -> (n_frames, NP, 4) array of [x, y, z, doppler], per-frame centred.

    A small fraction of rows (~0.002% of points, counted across the single-person
    77Ghz files) are sensor glitches: coordinates tens of metres out or Doppler in
    the thousands of m/s, impossible for a walking person. One such point in a frame
    dominates that frame's mean, so it is dropped before centring rather than after.
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    if raw.ndim == 1:
        raw = raw[None, :]
    raw = raw[~np.isnan(raw).any(1)]
    ok = (np.abs(raw[:, 2:5]) < MAX_RANGE_M).all(1) & (np.abs(raw[:, 5]) < MAX_SPEED_MS)
    raw = raw[ok]
    frame_no = raw[:, 0]
    feat = raw[:, 2:6]  # X, Y, Z, Doppler

    order = np.argsort(frame_no, kind="stable")
    frame_no, feat = frame_no[order], feat[order]
    _, starts = np.unique(frame_no, return_index=True)

    out = []
    for grp in np.split(np.arange(len(frame_no)), starts[1:]):
        pts = feat[grp]
        n = len(pts)
        if n == 0:
            continue  # every point in this frame was a filtered-out glitch
        if n >= NP:
            idx = rng.choice(n, NP, replace=False)
        else:
            idx = np.concatenate([np.arange(n), rng.choice(n, NP - n, replace=True)])
        F = pts[idx]
        out.append(F - F.mean(0, keepdims=True))
    return np.stack(out).astype(np.float16)


frames, meta = [], []
bad = []
for room in ROOMS:
    for route in ROUTES:
        for device in DEVICES:
            folder = ROOT / room / "1" / route / device
            if not folder.is_dir():
                continue
            for f in sorted(folder.glob("*.csv")):
                lab = int(f.stem)  # filename is the global volunteer id
                try:
                    arr = load_track(f)
                except Exception as e:
                    bad.append((str(f.relative_to(ROOT)), f"{type(e).__name__}: {e}"))
                    continue
                frames.append(arr)
                meta.append((lab, f"{room}_{route}_{device}", str(f.relative_to(ROOT))))

X = np.concatenate(frames, 0)
offs = np.array([0] + list(np.cumsum([len(a) for a in frames])), np.int64)
np.savez_compressed(OUT, X=X, offsets=offs,
                    labels=np.array([m[0] for m in meta], np.int64),
                    scenarios=np.array([m[1] for m in meta]),
                    tracks=np.array([m[2] for m in meta]))
print("unreadable:", bad)
print(f"tracks {len(meta)}  unique volunteers {len(set(m[0] for m in meta))}  "
      f"frames {X.shape} -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
