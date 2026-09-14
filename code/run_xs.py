"""Cross-scenario open-set gait ID on mmGait10.

Train on two walking modes, then test on the third. The rejection threshold is
calibrated on the SEEN modes only, which is the realistic hospital case: you enrol
staff walking normally and they later turn up carrying a chart or with hands in
pockets. Every number is produced by one trained model evaluated twice, so the
in-domain and shifted columns differ only by the walking mode, not by training data.
"""
import argparse, json, os, time
import numpy as np
import torch

import data as D_
import calib
import paths
from model import GaitNet, ArcFace

SCEN = ["free_walk", "hands_in_pockets", "smartphone"]

P = argparse.ArgumentParser()
P.add_argument("--npz", default=str(paths.PREPARED))
P.add_argument("--out", default=str(paths.RUNS_XS))
P.add_argument("--seed", type=int, default=0)
P.add_argument("--holdout", required=True, choices=SCEN)
P.add_argument("--n-known", type=int, default=6)
P.add_argument("--epochs", type=int, default=20)
P.add_argument("--batch", type=int, default=32)
P.add_argument("--lr", type=float, default=2e-3)
P.add_argument("--emb", type=int, default=128)
# Windows starts dataloader workers with spawn, which re-imports this script and
# re-runs the whole training body. These scripts are flat by design, so no workers there.
P.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
A = P.parse_args()
if os.name == "nt" and A.workers:
    print("[warn] --workers ignored on Windows (spawn re-imports this script); using 0", flush=True)
    A.workers = 0

torch.manual_seed(A.seed); np.random.seed(A.seed); torch.set_num_threads(2)
os.makedirs(A.out, exist_ok=True)
tag = f"seed{A.seed}_hold-{A.holdout}"

paths.require(A.npz, "prepared_64pts.npz (run prep.py first)")
D = D_.load(A.npz)
# identical subject draw to the in-domain experiment, so the two are comparable
rng = np.random.default_rng(1000 + A.seed)
subs = rng.permutation(10)
known, unknown = sorted(subs[:A.n_known].tolist()), sorted(subs[A.n_known:].tolist())
lmap = {g: i for i, g in enumerate(known)}
print(f"[{tag}] known={known} unknown={unknown} holdout={A.holdout}", flush=True)

lab, scn = D["labels"], D["scenarios"]
seen_m = scn != A.holdout
hold_m = scn == A.holdout
kn_m = np.isin(lab, known)
un_m = np.isin(lab, unknown)

# known + seen -> train / val / in-domain test, split at track level
srng = np.random.default_rng(A.seed)
tr, va, te = [], [], []
for k in known:
    idx = np.where((lab == k) & seen_m)[0]
    srng.shuffle(idx)
    n = len(idx); a, b = int(round(.70 * n)), int(round(.85 * n))
    tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])

groups = dict(
    train           = np.array(tr),
    val             = np.array(va),                       # calibration, seen modes only
    known_indomain  = np.array(te),
    known_shifted   = np.where(kn_m & hold_m)[0],
    unknown_indomain= np.where(un_m & seen_m)[0],
    unknown_shifted = np.where(un_m & hold_m)[0],
)
sets = {k: D_.build_windows(D, v) for k, v in groups.items()}
print("[tracks]", {k: len(v) for k, v in groups.items()}, flush=True)
print("[windows]", {k: len(v) for k, v in sets.items()}, flush=True)

mk = lambda n, t: torch.utils.data.DataLoader(
    D_.WindowSet(D, sets[n], lmap, train=t), batch_size=A.batch, shuffle=t,
    num_workers=A.workers, drop_last=t, persistent_workers=A.workers > 0)
loaders = {n: mk(n, n == "train") for n in sets}

net, head = GaitNet(emb_dim=A.emb), ArcFace(A.emb, len(known))
opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=A.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, A.lr, epochs=A.epochs,
                                            steps_per_epoch=max(len(loaders["train"]), 1))


@torch.no_grad()
def embed(loader):
    net.eval()
    Z, Y, T, S = [], [], [], []
    for x, y, t, s in loader:
        Z.append(net(x)); Y.append(y); T.append(t); S.append(s)
    return torch.cat(Z), torch.cat(Y).numpy(), torch.cat(T).numpy(), torch.cat(S).numpy()


log = []
for ep in range(A.epochs):
    net.train(); head.train(); t0 = time.time(); tot = nb = 0
    for x, y, _, _ in loaders["train"]:
        opt.zero_grad(); loss = head(net(x), y); loss.backward(); opt.step(); sched.step()
        tot += loss.item(); nb += 1
    zv, yv, _, _ = embed(loaders["val"])
    acc = float((head.cosine(zv).argmax(1).numpy() == yv).mean())
    log.append(dict(epoch=ep, loss=tot / max(nb, 1), val_closed_acc=acc, secs=time.time() - t0))
    print(f"[{tag}] ep{ep:02d} loss {tot/max(nb,1):.3f} val_acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

head.eval()
E = {}
with torch.no_grad():
    for n in sets:
        z, y, t, s = embed(loaders[n])
        E[n] = dict(cos=head.cosine(z).numpy(), y=y, t=t, s=s)

K = len(known)
np.savez(os.path.join(A.out, f"scores_{tag}.npz"),
         **{f"{n}_{k}": v for n, d in E.items() for k, v in d.items()},
         known=np.array(known), unknown=np.array(unknown))


def vote(cos, trk, st, k):
    if k == 1:
        return cos
    out = []
    for t in np.unique(trk):
        m = np.where(trk == t)[0]; m = m[np.argsort(st[m])]
        for i in range(0, len(m) - k + 1, k):
            out.append(cos[m[i:i + k]].mean(0))
    return np.array(out) if out else np.zeros((0, cos.shape[1]))


def votey(y, trk, st, k, K):
    return vote(np.eye(K)[y], trk, st, k).argmax(1) if k > 1 else y


def macro_f1(pred, true, K):
    f1 = []
    for c in range(K + 1):
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def decide(cos, taus):
    a = cos.argmax(1)
    return np.where(cos.max(1) >= taus[a], a, len(taus))


results = {}
for kv in [1, 4, 8]:
    V = {n: vote(E[n]["cos"], E[n]["t"], E[n]["s"], kv) for n in sets}
    Yv = {n: votey(E[n]["y"], E[n]["t"], E[n]["s"], kv, K) for n in ["val", "known_indomain", "known_shifted"]}
    if min(len(V[n]) for n in V) == 0:
        continue

    # thresholds fitted on seen-mode validation only, then reused unchanged
    taus = {"evt": calib.evt_threshold(V["val"], Yv["val"], K)[0],
            "p5": calib.percentile_threshold(V["val"], Yv["val"], K)[0],
            "conformal": calib.conformal_threshold(V["val"], Yv["val"], K)[0]}

    row = {}
    for cond, kn, un in [("in_domain", "known_indomain", "unknown_indomain"),
                         ("shifted", "known_shifted", "unknown_shifted")]:
        ck, cu = V[kn], V[un]
        auroc = calib.auroc(ck.max(1), cu.max(1))
        if len(cu) > len(ck):
            cu = cu[np.random.default_rng(7).choice(len(cu), len(ck), replace=False)]
        true = np.concatenate([Yv[kn], np.full(len(cu), K)])
        r = dict(closed_set_acc=float((ck.argmax(1) == Yv[kn]).mean()), unknown_auroc=auroc,
                 n_known=int(len(ck)), n_unknown=int(len(cu)))
        # oracle for this condition, for reference only
        tau_o = calib.oracle_threshold(ck, cu, K)[0]
        for name, tt in list(taus.items()) + [("oracle", tau_o)]:
            pred = np.concatenate([decide(ck, tt), decide(cu, tt)])
            r[name] = dict(macro_f1=macro_f1(pred, true, K),
                           known_recall=float(np.mean(decide(ck, tt) == Yv[kn])),
                           unknown_recall=float(np.mean(decide(cu, tt) == K)))
        row[cond] = r
    results[f"k{kv}"] = row
    print(f"[{tag}] k={kv} "
          f"in-domain: acc {row['in_domain']['closed_set_acc']:.3f} auroc {row['in_domain']['unknown_auroc']:.3f} "
          f"p5 {row['in_domain']['p5']['macro_f1']:.3f} | "
          f"shifted: acc {row['shifted']['closed_set_acc']:.3f} auroc {row['shifted']['unknown_auroc']:.3f} "
          f"p5 {row['shifted']['p5']['macro_f1']:.3f}", flush=True)

json.dump(dict(known=known, unknown=unknown, holdout=A.holdout,
               tracks={k: int(len(v)) for k, v in groups.items()},
               windows={k: int(len(v)) for k, v in sets.items()},
               train_log=log, results=results),
          open(os.path.join(A.out, f"result_{tag}.json"), "w"), indent=1)
print(f"[{tag}] done", flush=True)
