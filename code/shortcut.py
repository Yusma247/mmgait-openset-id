"""Shortcut floor: how well can a subject be identified from cheap summary
statistics alone, with no gait shape at all? Any model score should be read
against this number. Numpy only.
"""
import pickle, os, glob, json
import numpy as np

import paths
ROOT = paths.require(paths.POINT_CLOUDS, "the point_clouds folder")
NST,STEP=30,6
X=[];Y=[];G=[];S=[]
for sub in sorted(os.listdir(ROOT), key=lambda s:int(s[6:])):
    lab=int(sub[6:])
    for sc in sorted(os.listdir(os.path.join(ROOT,sub))):
        for f in sorted(glob.glob(os.path.join(ROOT,sub,sc,"*.obj"))):
            try: d=pickle.load(open(f,"rb"))
            except Exception: continue
            card=np.array([fr["cardinality"][0] for fr in d],float)
            pw=np.array([10*np.log10(fr["powers"].mean()+1e-8) for fr in d])
            dop=np.array([np.abs(fr["dopplers"]).mean() for fr in d])
            zsd=np.array([fr["z_coord"].std() for fr in d])
            xsd=np.array([fr["elements"][:,0].std() for fr in d])
            for i in np.arange(len(d)-NST, step=STEP):
                sl=slice(i,i+NST)
                X.append([card[sl].mean(),card[sl].std(),pw[sl].mean(),pw[sl].std(),
                          dop[sl].mean(),zsd[sl].mean(),xsd[sl].mean()])
                Y.append(lab); G.append(f); S.append(sc)
X=np.array(X);Y=np.array(Y);G=np.array(G);S=np.array(S)
np.savez(paths.RESULTS / "shortcut_feats.npz",X=X,Y=Y,G=G,S=S)
print("crops:",X.shape,"classes:",len(set(Y)))

names=["card_mean","card_std","powdB_mean","powdB_std","dop_mean","z_std","x_std"]
tracks=np.unique(G); rng=np.random.default_rng(0); rng.shuffle(tracks)
folds=np.array_split(tracks,5)
def run(cols):
    acc=[]
    for fo in folds:
        te=np.isin(G,fo); tr=~te
        mu=X[tr][:,cols].mean(0); sd=X[tr][:,cols].std(0)+1e-9
        A=(X[tr][:,cols]-mu)/sd; B=(X[te][:,cols]-mu)/sd
        C=np.stack([A[Y[tr]==c].mean(0) for c in range(10)])
        pred=np.argmin(((B[:,None,:]-C[None])**2).sum(-1),1)
        acc.append((pred==Y[te]).mean())
    return float(np.mean(acc)), float(np.std(acc))
print(f"{'feature set':28} {'acc':>7} {'sd':>6}")
print(f"{'chance':28} {0.10:7.3f}")
for i,n in enumerate(names):
    a,s=run([i]); print(f"{n:28} {a:7.3f} {s:6.3f}")
a,s=run(list(range(len(names)))); print(f"{'ALL 7 metadata features':28} {a:7.3f} {s:6.3f}")
a,s=run([0,1]); print(f"{'cardinality only (2)':28} {a:7.3f} {s:6.3f}")
a,s=run([2,3]); print(f"{'power only (2)':28} {a:7.3f} {s:6.3f}")
