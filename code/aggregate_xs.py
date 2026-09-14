"""Aggregate the cross-scenario runs: train on two walking modes, test on the third."""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" prints on a Windows console codepage

res = [json.load(open(f)) for f in sorted(glob.glob(str(paths.RUNS_XS / "result_seed*.json")))]
if not res:
    raise SystemExit(f"No results in {paths.RUNS_XS}. Run run_xs.py first.")
KS = [1, 4, 8]
CONDS = ["in_domain", "shifted"]
NICE = {"in_domain": "seen modes", "shifted": "unseen mode"}


def col(k, cond, path):
    out = []
    for r in res:
        d = r["results"].get(f"k{k}")
        if d is None:
            continue
        d = d[cond]
        for p in path.split("."):
            d = d[p]
        out.append(d)
    return np.array(out, float)


rows = ["| windows voted | condition | closed-set acc | unknown AUROC | macro F1 (5th pct) | macro F1 (conformal) | staff kept (5th pct) | staff kept (conformal) |",
        "|---|---|---|---|---|---|---|---|"]
for k in KS:
    for c in CONDS:
        rows.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} |".format(
            f"k = {k}", NICE[c], col(k, c, "closed_set_acc").mean(), col(k, c, "unknown_auroc").mean(),
            col(k, c, "p5.macro_f1").mean(), col(k, c, "p5.macro_f1").std(),
            col(k, c, "conformal.macro_f1").mean(), col(k, c, "conformal.macro_f1").std(),
            col(k, c, "p5.known_recall").mean(), col(k, c, "conformal.known_recall").mean()))
table = "\n".join(rows)
print(table)

print("\nper held-out mode, k=8, macro F1: 5th percentile vs split-conformal")
for r in res:
    d = r["results"]["k8"]
    print(f"  hold out {r['holdout']:17} "
          f"seen  p5 {d['in_domain']['p5']['macro_f1']:.3f} conformal {d['in_domain']['conformal']['macro_f1']:.3f}   "
          f"unseen p5 {d['shifted']['p5']['macro_f1']:.3f} conformal {d['shifted']['conformal']['macro_f1']:.3f}   "
          f"(staff kept: p5 {d['shifted']['p5']['known_recall']:.3f} "
          f"conformal {d['shifted']['conformal']['known_recall']:.3f})")

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
C = {"in_domain": "#4C72B0", "shifted": "#C44E52"}
for c in CONDS:
    a = ax[1]
    mu = np.array([col(k, c, "unknown_auroc").mean() for k in KS])
    sd = np.array([col(k, c, "unknown_auroc").std() for k in KS])
    a.errorbar(KS, mu, sd, marker="o", color=C[c], label=NICE[c], ls="-" if c == "in_domain" else "--")
    a.set_xlabel("windows voted (k)"); a.grid(alpha=0.3)
    for a, base, name in [(ax[0], "macro_f1", "macro F1"), (ax[2], "known_recall", "staff kept")]:
        for method, mk, alpha in [("p5", "o", 1.0), ("conformal", "^", 0.6)]:
            metric = f"{method}.{base}"
            mu = np.array([col(k, c, metric).mean() for k in KS])
            sd = np.array([col(k, c, metric).std() for k in KS])
            a.errorbar(KS, mu, sd, marker=mk, color=C[c], alpha=alpha,
                       label=f"{NICE[c]}, {method}", ls="-" if c == "in_domain" else "--")
        a.set_xlabel("windows voted (k)"); a.set_title(name); a.grid(alpha=0.3)
ax[0].set_ylabel("macro F1 over known + unknown")
ax[0].set_title("Open-set F1: 5th pct (o) vs split-conformal (^)\nthreshold fitted on the seen modes only")
ax[1].set_title("Unknown detection\nthreshold-free, so it isolates the embedding")
ax[2].set_title("Staff still accepted: 5th pct (o) vs conformal (^)\nthis is where a stale threshold shows up")
ax[0].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(paths.FIGURES / "xscenario_results.png", dpi=130)

json.dump(dict(table=table,
               per_holdout={r["holdout"]: r["results"] for r in res}),
          open(paths.RESULTS / "summary_xs.json", "w"), indent=1)
print("\nwrote", paths.FIGURES / "xscenario_results.png")
