"""Open-set gait recognition on mmGait10.

Point cloud input, metric-learning head, rejection thresholds calibrated without
impostor data, and temporal voting over consecutive windows.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

import data as D_
import calib
import paths
from model import GaitNet, ArcFace

P = argparse.ArgumentParser()
P.add_argument("--npz", default=str(paths.PREPARED))
P.add_argument("--out", default=str(paths.RUNS))
P.add_argument("--seed", type=int, default=0)
P.add_argument("--n-known", type=int, default=6)
P.add_argument("--epochs", type=int, default=20)
P.add_argument("--batch", type=int, default=32)
P.add_argument("--lr", type=float, default=2e-3)
P.add_argument("--emb", type=int, default=128)
# Windows starts dataloader workers with spawn, which re-imports this script and
# re-runs the whole training body. These scripts are flat by design, so no workers there.
P.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
P.add_argument("--bench", action="store_true")
A = P.parse_args()
if os.name == "nt" and A.workers:
    print("[warn] --workers ignored on Windows (spawn re-imports this script); using 0", flush=True)
    A.workers = 0

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.set_num_threads(2)
os.makedirs(A.out, exist_ok=True)
tag = f"seed{A.seed}_k{A.n_known}"

paths.require(A.npz, "prepared_64pts.npz (run prep.py first)")
D = D_.load(A.npz)
rng = np.random.default_rng(1000 + A.seed)
subs = rng.permutation(10)
known, unknown = sorted(subs[:A.n_known].tolist()), sorted(subs[A.n_known:].tolist())
lmap = {g: i for i, g in enumerate(known)}
openness = 1 - np.sqrt(2 * len(known) / (len(known) + 10))
print(f"[{tag}] known={known} unknown={unknown} openness={openness*100:.2f}%", flush=True)

tr, va, te_k, te_u = D_.split_tracks(D, known, unknown, A.seed)
sets = {k: D_.build_windows(D, v) for k, v in
        dict(train=tr, val=va, test_known=te_k, test_unknown=te_u).items()}
print("[windows]", {k: len(v) for k, v in sets.items()}, flush=True)

mk = lambda name, train: torch.utils.data.DataLoader(
    D_.WindowSet(D, sets[name], lmap, train=train), batch_size=A.batch,
    shuffle=train, num_workers=A.workers, drop_last=train, persistent_workers=A.workers > 0)
loaders = {n: mk(n, n == "train") for n in sets}

net = GaitNet(emb_dim=A.emb)
head = ArcFace(A.emb, len(known))
opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=A.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, A.lr, epochs=A.epochs,
                                            steps_per_epoch=max(len(loaders["train"]), 1))

if A.bench:
    net.train(); t0 = time.time(); n = 0
    for x, y, _, _ in loaders["train"]:
        opt.zero_grad(); head(net(x), y).backward(); opt.step(); n += 1
        if n == 20: break
    print(f"[bench] {(time.time()-t0)/n:.3f} s/batch -> "
          f"{(time.time()-t0)/n*len(loaders['train'])/60:.1f} min/epoch"); raise SystemExit


@torch.no_grad()
def embed(loader):
    net.eval()
    Z, Y, T, S = [], [], [], []
    for x, y, t, s in loader:
        Z.append(net(x)); Y.append(y); T.append(t); S.append(s)
    return (torch.cat(Z), torch.cat(Y).numpy(), torch.cat(T).numpy(), torch.cat(S).numpy())


log = []
for ep in range(A.epochs):
    net.train(); head.train(); t0 = time.time(); tot = nb = 0
    for x, y, _, _ in loaders["train"]:
        opt.zero_grad()
        loss = head(net(x), y)
        loss.backward(); opt.step(); sched.step()
        tot += loss.item(); nb += 1
    zv, yv, _, _ = embed(loaders["val"])
    acc = (head.cosine(zv).argmax(1).numpy() == yv).mean()
    log.append(dict(epoch=ep, loss=tot / max(nb, 1), val_closed_acc=float(acc),
                    secs=time.time() - t0))
    print(f"[{tag}] ep{ep:02d} loss {tot/max(nb,1):.3f} val_closed_acc {acc:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

# ---------------- scores ----------------
with torch.no_grad():
    head.eval()
    zv, yv, tv, sv = embed(loaders["val"])
    zk, yk, tk, sk = embed(loaders["test_known"])
    zu, yu, tu, su = embed(loaders["test_unknown"])
    cos_v = head.cosine(zv).numpy()
    cos_k = head.cosine(zk).numpy()
    cos_u = head.cosine(zu).numpy()

K = len(known)
np.savez(os.path.join(A.out, f"scores_{tag}.npz"), cos_v=cos_v, yv=yv, tv=tv, sv=sv,
         cos_k=cos_k, yk=yk, tk=tk, sk=sk, cos_u=cos_u, tu=tu, su=su,
         known=np.array(known), unknown=np.array(unknown))


def vote(cos, trk, st, k):
    """Average the cosine vector over k consecutive windows of the same track."""
    if k == 1:
        return cos, trk
    out, ot = [], []
    for t in np.unique(trk):
        m = np.where(trk == t)[0]
        m = m[np.argsort(st[m])]
        for i in range(0, len(m) - k + 1, k):
            out.append(cos[m[i:i + k]].mean(0)); ot.append(t)
    return (np.array(out), np.array(ot)) if out else (np.zeros((0, cos.shape[1])), np.array([]))


def macro_f1(pred, true, n_cls):
    """n_cls known classes plus index n_cls for 'unknown'."""
    f1 = []
    for c in range(n_cls + 1):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)), f1


def decide(cos, taus):
    a = cos.argmax(1)
    return np.where(cos.max(1) >= taus[a], a, len(taus))


results = {}
for kvote in [1, 2, 4, 6, 8]:
    ck, tkk = vote(cos_k, tk, sk, kvote)
    cu, _ = vote(cos_u, tu, su, kvote)
    cv, _ = vote(cos_v, tv, sv, kvote)
    yv_v = vote(np.eye(K)[yv], tv, sv, kvote)[0].argmax(1) if kvote > 1 else yv
    yk_v = vote(np.eye(K)[yk], tk, sk, kvote)[0].argmax(1) if kvote > 1 else yk
    if len(ck) == 0 or len(cu) == 0:
        continue
    auroc_full = calib.auroc(ck.max(1), cu.max(1))     # uses every unknown window
    # balance the unknown side for F1 so the macro average is not dominated by it
    if len(cu) > len(ck):
        sel = np.random.default_rng(7).choice(len(cu), len(ck), replace=False)
        cu = cu[sel]

    methods = {
        "loko":      calib.loko_threshold(cv, yv_v, K),
        "evt":       calib.evt_threshold(cv, yv_v, K),
        "p5":        calib.percentile_threshold(cv, yv_v, K),
        "conformal": calib.conformal_threshold(cv, yv_v, K),
        "oracle":    calib.oracle_threshold(ck, cu, K),
    }
    true = np.concatenate([yk_v, np.full(len(cu), K)])
    row = dict(n_known_windows=int(len(ck)), n_unknown_windows=int(len(cu)),
               closed_set_acc=float((ck.argmax(1) == yk_v).mean()),
               unknown_auroc=auroc_full)
    for name, (taus, extra) in methods.items():
        pred = np.concatenate([decide(ck, taus), decide(cu, taus)])
        mf1, per = macro_f1(pred, true, K)
        row[name] = dict(macro_f1=mf1, tau=[float(t) for t in taus],
                         known_recall=float(np.mean(decide(ck, taus) == yk_v)),
                         unknown_recall=float(np.mean(decide(cu, taus) == K)))
    results[f"k{kvote}"] = row
    print(f"[{tag}] vote k={kvote} closed {row['closed_set_acc']:.3f} auroc {row['unknown_auroc']:.3f} "
          + " ".join(f"{m}={row[m]['macro_f1']:.3f}" for m in methods), flush=True)

json.dump(dict(known=known, unknown=unknown, openness=float(openness * 100),
               windows={k: int(len(v)) for k, v in sets.items()},
               train_log=log, results=results),
          open(os.path.join(A.out, f"result_{tag}.json"), "w"), indent=1)
torch.save(dict(net=net.state_dict(), head=head.state_dict()),
           os.path.join(A.out, f"model_{tag}.pt"))
print(f"[{tag}] done", flush=True)
