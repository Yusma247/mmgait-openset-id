"""Two extra figures for REPORT.md, not part of the main pipeline outputs.
Both are built directly from already-saved results, no new experiments.

  figures/report_shortcut_bar.png    per-feature accuracy of the metadata-only shortcut
  figures/report_holdout_bar.png     staff-kept, seen vs unseen mode, all 3 holdouts, k=8
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths

# ---------------------------------------------------------------- shortcut bar
d = np.load(paths.RESULTS / "shortcut_feats.npz")
X, Y, G = d["X"], d["Y"], d["G"]
names = ["card_mean", "card_std", "powdB_mean", "powdB_std", "dop_mean", "z_std", "x_std"]
tracks = np.unique(G)
rng = np.random.default_rng(0)
rng.shuffle(tracks)
folds = np.array_split(tracks, 5)


def run(cols):
    acc = []
    for fo in folds:
        te = np.isin(G, fo); tr = ~te
        mu = X[tr][:, cols].mean(0); sd = X[tr][:, cols].std(0) + 1e-9
        A = (X[tr][:, cols] - mu) / sd; B = (X[te][:, cols] - mu) / sd
        C = np.stack([A[Y[tr] == c].mean(0) for c in range(10)])
        pred = np.argmin(((B[:, None, :] - C[None]) ** 2).sum(-1), 1)
        acc.append((pred == Y[te]).mean())
    return float(np.mean(acc)), float(np.std(acc))


rows = [(n, *run([i])) for i, n in enumerate(names)]
rows.append(("ALL 7 features", *run(list(range(7)))))
rows.append(("cardinality only", *run([0, 1])))
rows.append(("power only", *run([2, 3])))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(8.5, 5.2))
labels = [r[0] for r in rows]
accs = [r[1] for r in rows]
sds = [r[2] for r in rows]
colors = ["#8172B2" if "ALL" in l else "#C44E52" if l in ("cardinality only", "power only") else "#4C72B0"
          for l in labels]
y = np.arange(len(labels))
ax.barh(y, accs, xerr=sds, color=colors, height=0.6, capsize=3)
ax.axvline(0.10, color="k", ls="--", lw=1, label="chance (10%, 10 subjects)")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("5-fold nearest-centroid accuracy, grouped by track")
ax.set_title("The metadata-only shortcut floor\nno gait shape used, only cheap per-window summary statistics")
ax.legend(loc="lower right", fontsize=9)
for yi, a in zip(y, accs):
    ax.text(a + 0.01, yi, f"{a:.3f}", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(paths.FIGURES / "report_shortcut_bar.png", dpi=140)
print("wrote", paths.FIGURES / "report_shortcut_bar.png")

# ---------------------------------------------------------------- per-holdout bar
xs = json.load(open(paths.RESULTS / "summary_xs.json"))
holds = ["free_walk", "hands_in_pockets", "smartphone"]
seen = [xs["per_holdout"][h]["k8"]["in_domain"]["p5"]["known_recall"] for h in holds]
seen_cf = [xs["per_holdout"][h]["k8"]["in_domain"]["conformal"]["known_recall"] for h in holds]
unseen_p5 = [xs["per_holdout"][h]["k8"]["shifted"]["p5"]["known_recall"] for h in holds]
unseen_cf = [xs["per_holdout"][h]["k8"]["shifted"]["conformal"]["known_recall"] for h in holds]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = np.arange(len(holds))
w = 0.2
ax.bar(x - 1.5 * w, seen, w, label="seen modes, 5th pct", color="#4C72B0", alpha=0.55)
ax.bar(x - 0.5 * w, seen_cf, w, label="seen modes, conformal", color="#4C72B0")
ax.bar(x + 0.5 * w, unseen_p5, w, label="unseen mode, 5th pct", color="#C44E52", alpha=0.55)
ax.bar(x + 1.5 * w, unseen_cf, w, label="unseen mode, conformal", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels([h.replace("_", " ") for h in holds])
ax.set_ylabel("staff kept (known recall), k = 8 windows")
ax.set_ylim(0, 1.05)
ax.set_title("Staff-kept rate by held-out walking mode\nthe free_walk holdout is where a frozen threshold actually breaks")
ax.legend(fontsize=8.5, loc="lower center", ncol=2)
ax.axhline(0.95, color="k", ls=":", lw=0.8)
for xi, vals in zip(x, zip(seen, seen_cf, unseen_p5, unseen_cf)):
    for off, v in zip([-1.5, -0.5, 0.5, 1.5], vals):
        ax.text(xi + off * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
plt.tight_layout()
plt.savefig(paths.FIGURES / "report_holdout_bar.png", dpi=140)
print("wrote", paths.FIGURES / "report_holdout_bar.png")
