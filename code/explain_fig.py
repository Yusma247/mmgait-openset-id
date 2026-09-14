"""Five-panel explainer built from one real track: a single frame from above and
from the side, the walking trajectory, the micro-Doppler spectrogram, and the same
gait signal seen inside the point cloud. Numpy and matplotlib only, no torch.
"""
SUBJ, SCEN, TRACK = "target0", "free_walk", "000"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pickle, zipfile, os
import paths
B = paths.RAW
pc=pickle.load(open(B / "point_clouds" / SUBJ / SCEN / f"pc_{SUBJ[6:]}_{TRACK}.obj", "rb"))
with zipfile.ZipFile(B / "spectrograms" / SUBJ / SCEN / f"md_{SUBJ[6:]}_{TRACK}.pt") as z:
    rec=[x for x in z.namelist() if "/data/" in x][0]
    sp=np.frombuffer(z.read(rec),dtype="<f4")
sp=sp.reshape(128,-1)
print("frames",len(pc),"spec",sp.shape)
card=np.array([int(f["cardinality"][0]) for f in pc])
F0=int(np.argmax(card[150:400]))+150
fr=pc[F0]; n=int(fr["cardinality"][0])
x,y=fr["elements"][:n,0],fr["elements"][:n,1]; z_=fr["z_coord"][:n]; dop=fr["dopplers"][:n]

fig=plt.figure(figsize=(16,8.8))
gs=fig.add_gridspec(2,3,hspace=0.42,wspace=0.30)
ax=fig.add_subplot(gs[0,0])
cx,cy=x.mean(),y.mean()
s=ax.scatter(x-cx,y-cy,c=dop,cmap="coolwarm",s=18,vmin=-1.5,vmax=1.5)
ax.set_xlabel("x relative to body centre (m)"); ax.set_ylabel("y relative to body centre (m)")
ax.set_title(f"A. One frame from above\n{n} detections, colour = radial speed (m/s)",fontsize=10)
ax.set_xlim(-0.8,0.8); ax.set_ylim(-0.8,0.8); ax.set_aspect("equal"); plt.colorbar(s,ax=ax,shrink=.85)

ax=fig.add_subplot(gs[0,1])
ax.scatter(x-cx,z_,c=dop,cmap="coolwarm",s=18,vmin=-1.5,vmax=1.5)
ax.set_xlabel("x (m)"); ax.set_ylabel("z, height around torso centre (m)")
ax.set_title("B. Same frame from the side\nthe silhouette is a smear, not a skeleton",fontsize=10)
ax.set_xlim(-0.8,0.8); ax.set_ylim(-1.2,1.2); ax.set_aspect("equal")

ax=fig.add_subplot(gs[0,2])
seg=pc[100:400:6]
for i,f in enumerate(seg):
    m=int(f["cardinality"][0])
    ax.scatter(f["elements"][:m,0],f["elements"][:m,1],s=3,color=plt.cm.viridis(i/len(seg)),alpha=.5)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("C. 30 s of walking, colour = time\nthe person is a moving blob of dots",fontsize=10)

ax=fig.add_subplot(gs[1,:2])
t0,t1=F0-90,F0+90
ax.imshow(sp[:,t0:t1],aspect="auto",origin="lower",cmap="magma",
          extent=[t0/10,t1/10,-2.5,2.5],vmax=np.percentile(sp,99.5))
ax.set_xlabel("time (s)"); ax.set_ylabel("radial velocity (m/s)")
ax.set_title("D. Micro-Doppler spectrogram of the same 18 s. Bright horizontal band = torso. "
             "The arcs above it = feet and arms swinging faster than the body.",fontsize=10)

ax=fig.add_subplot(gs[1,2])
for i,f in enumerate(pc[t0:t1]):
    m=int(f["cardinality"][0]); t=(t0+i)/10
    ax.scatter(np.full(m,t),f["dopplers"][:m],s=1.4,alpha=.22,color="#333333")
    ax.scatter([t],[f["dopplers"][:m].mean()],s=5,color="crimson")
ax.set_xlabel("time (s)"); ax.set_ylabel("Doppler (m/s)")
ax.set_title("E. The same gait signal inside the point cloud\nred = torso speed, grey spread = limbs",fontsize=10)
plt.savefig(paths.FIGURES / "radar_explainer.png", dpi=118, bbox_inches="tight")
print("saved", paths.FIGURES / "radar_explainer.png")
