import json, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" prints on a Windows console codepage

RUNS = str(paths.RUNS)
res = [json.load(open(f)) for f in sorted(glob.glob(f"{RUNS}/result_seed*.json"))]
KS = [1, 2, 4, 6, 8]
METHODS = ["loko", "evt", "p5", "conformal", "oracle"]
NICE = {"loko": "leave-one-known-out", "evt": "EVT Weibull tail",
        "p5": "5th-percentile", "conformal": "split-conformal",
        "oracle": "oracle (sees unknowns)"}

def col(k, path):
    out = []
    for r in res:
        d = r["results"][f"k{k}"]
        for p in path.split("."):
            d = d[p]
        out.append(d)
    return np.array(out, float)

lines = []
lines.append("| windows voted | seconds | closed-set acc | unknown AUROC | "
             + " | ".join(NICE[m] for m in METHODS) + " |")
lines.append("|---|---|---|---|" + "---|" * len(METHODS))
for k in KS:
    secs = (30 + 10 * (k - 1)) / 10
    row = [f"k = {k}", f"{secs:.0f} s",
           f"{col(k,'closed_set_acc').mean():.3f}",
           f"{col(k,'unknown_auroc').mean():.3f}"]
    for m in METHODS:
        v = col(k, f"{m}.macro_f1")
        row.append(f"**{v.mean():.3f}** ± {v.std():.3f}" if m in ("evt", "p5", "conformal") else f"{v.mean():.3f} ± {v.std():.3f}")
    lines.append("| " + " | ".join(row) + " |")
table = "\n".join(lines)
print(table)

print("\nknown / unknown recall at k=8")
for m in METHODS:
    print(f"  {NICE[m]:24} known {col(8,f'{m}.known_recall').mean():.3f}  "
          f"unknown {col(8,f'{m}.unknown_recall').mean():.3f}")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
cols = {"loko": "#C44E52", "evt": "#4C72B0", "p5": "#55A868", "conformal": "#DD8452", "oracle": "#8172B2"}
for m in METHODS:
    mu = np.array([col(k, f"{m}.macro_f1").mean() for k in KS])
    sd = np.array([col(k, f"{m}.macro_f1").std() for k in KS])
    ls = "--" if m == "oracle" else "-"
    ax[0].plot(KS, mu, ls, marker="o", color=cols[m], label=NICE[m])
    ax[0].fill_between(KS, mu - sd, mu + sd, color=cols[m], alpha=0.12)
ax[0].set_xlabel("windows voted (k)"); ax[0].set_ylabel("macro F1 over 6 known + unknown")
ax[0].set_title("Open-set F1 vs temporal voting\nmean of 3 splits, shaded = 1 sd")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

a = np.array([col(k, "unknown_auroc") for k in KS])
c = np.array([col(k, "closed_set_acc") for k in KS])
ax[1].errorbar(KS, a.mean(1), a.std(1), marker="o", color="#4C72B0", label="unknown detection AUROC")
ax[1].errorbar(KS, c.mean(1), c.std(1), marker="s", color="#937860", label="closed-set accuracy")
ax[1].set_xlabel("windows voted (k)"); ax[1].set_ylim(0.6, 1.02)
ax[1].set_title("Voting helps rejection far more\nthan it helps identification")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

d = np.load(f"{RUNS}/scores_seed0_k6.npz")
def vote(cos, trk, st, k):
    if k == 1: return cos
    out = []
    for t in np.unique(trk):
        m = np.where(trk == t)[0]; m = m[np.argsort(st[m])]
        for i in range(0, len(m) - k + 1, k): out.append(cos[m[i:i+k]].mean(0))
    return np.array(out)
bins = np.linspace(0.25, 1.0, 60)
for k, style in [(1, dict(histtype="step", lw=1.6, ls=":")),
                 (8, dict(histtype="stepfilled", alpha=0.45, lw=1.4))]:
    kk = vote(d["cos_k"], d["tk"], d["sk"], k).max(1)
    uu = vote(d["cos_u"], d["tu"], d["su"], k).max(1)
    ax[2].hist(kk, bins=bins, density=True, color="#4C72B0",
               edgecolor="#4C72B0", label=f"known, k={k}", **style)
    ax[2].hist(uu, bins=bins, density=True, color="#C44E52",
               edgecolor="#C44E52", label=f"unknown, k={k}", **style)
ax[2].set_xlabel("max cosine to a known prototype"); ax[2].set_ylabel("density")
ax[2].set_title("Score separation, split 0\ndotted = 1 window, filled = 8 windows")
ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(paths.FIGURES / "openset_results.png", dpi=130)
json.dump(dict(table=table,
               summary={f"k{k}": {m: [float(col(k, f"{m}.macro_f1").mean()),
                                      float(col(k, f"{m}.macro_f1").std())] for m in METHODS}
                        | {"closed_set_acc": float(col(k, "closed_set_acc").mean()),
                           "unknown_auroc": float(col(k, "unknown_auroc").mean())}
                        for k in KS},
               splits=[dict(known=r["known"], unknown=r["unknown"]) for r in res]),
          open(paths.RESULTS / "summary.json", "w"), indent=1)
print("\nwrote", paths.FIGURES / "openset_results.png", "and", paths.RESULTS / "summary.json")
