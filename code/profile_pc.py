"""Per-track and per-subject statistics for the raw mmGait10 point clouds.

Writes results/pc_profile.json and results/card.npy. Numpy only.
"""
import pickle, os, json, glob
import numpy as np
from collections import defaultdict

import paths

ROOT = paths.require(paths.POINT_CLOUDS, "the point_clouds folder")
FPS = 10.0
NSTEPS, CROP_STEP = 30, 6

rows = []
BAD, RETRIED = [], []
allcard = []
per_sub = defaultdict(lambda: defaultdict(float))
gx=[];gy=[];gz=[];gd=[];gp=[]

for sub in sorted(os.listdir(ROOT)):
    for sc in sorted(os.listdir(os.path.join(ROOT, sub))):
        for f in sorted(glob.glob(os.path.join(ROOT, sub, sc, "*.obj"))):
            d = None
            for attempt in range(6):
                try:
                    d = pickle.load(open(f, "rb")); break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:50]}"
            if d is None:
                BAD.append((os.path.relpath(f, ROOT), err)); continue
            if attempt: RETRIED.append((os.path.relpath(f, ROOT), attempt))
            card = np.array([fr["cardinality"][0] for fr in d])
            xs = np.concatenate([fr["elements"][:,0] for fr in d]) if len(d) else np.array([])
            ys = np.concatenate([fr["elements"][:,1] for fr in d])
            zs = np.concatenate([fr["z_coord"] for fr in d])
            dp = np.concatenate([fr["dopplers"] for fr in d])
            pw = np.concatenate([fr["powers"] for fr in d])
            ncrops = max(0, len(np.arange(len(d)-NSTEPS, step=CROP_STEP)))
            rows.append(dict(subject=sub, scenario=sc, track=os.path.basename(f),
                frames=len(d), dur_s=len(d)/FPS, pts_total=int(card.sum()),
                pts_mean=float(card.mean()), pts_med=float(np.median(card)),
                pts_min=int(card.min()), pts_max=int(card.max()),
                frac_lt50=float((card<50).mean()), frac_lt20=float((card<20).mean()),
                ncrops=int(ncrops)))
            allcard.append(card)
            per_sub[sub]["frames"] += len(d); per_sub[sub]["crops"] += ncrops
            per_sub[sub]["pts"] += int(card.sum())
            per_sub[sub][sc+"_frames"] += len(d)
            for g,v in ((gx,xs),(gy,ys),(gz,zs),(gd,dp),(gp,pw)):
                g.append(np.array([v.min(), v.max(), v.mean()]))

card = np.concatenate(allcard)
def q(a,p): return float(np.percentile(a,p))
summary = dict(
    n_tracks=len(rows), n_subjects=len(per_sub),
    total_frames=int(card.size), total_minutes=card.size/FPS/60,
    total_points=int(card.sum()),
    pts_per_frame=dict(mean=float(card.mean()), std=float(card.std()), median=q(card,50),
                       p5=q(card,5), p25=q(card,25), p75=q(card,75), p95=q(card,95),
                       min=int(card.min()), max=int(card.max())),
    frac_frames_lt150=float((card<150).mean()),
    frac_frames_lt50=float((card<50).mean()),
    frac_frames_lt20=float((card<20).mean()),
    total_crops=int(sum(r["ncrops"] for r in rows)),
)
G = {k: dict(min=float(np.min([a[0] for a in g])), max=float(np.max([a[1] for a in g])),
             mean=float(np.mean([a[2] for a in g])))
     for k,g in (("x",gx),("y",gy),("z",gz),("doppler",gd),("power",gp))}
summary["ranges"] = G

summary["unreadable_files"]=BAD
summary["files_needing_retry"]=RETRIED
out = dict(summary=summary, per_subject={k: dict(v) for k,v in per_sub.items()}, tracks=rows)
json.dump(out, open(paths.RESULTS / "pc_profile.json", "w"), indent=1)
np.save(paths.RESULTS / "card.npy", card)

print(json.dumps(summary, indent=1))
print("\nper subject:")
print(f"{'subj':8} {'tracks':>6} {'frames':>7} {'min':>6} {'crops':>6} {'pts/frame':>9}  scenario frame split")
for s in sorted(per_sub, key=lambda x:int(x[6:])):
    v=per_sub[s]; tr=[r for r in rows if r['subject']==s]
    sc = {c: int(v.get(c+'_frames',0)) for c in ['free_walk','hands_in_pockets','smartphone']}
    print(f"{s:8} {len(tr):6d} {int(v['frames']):7d} {v['frames']/FPS/60:6.1f} {int(v['crops']):6d} {v['pts']/v['frames']:9.1f}  {sc}")
