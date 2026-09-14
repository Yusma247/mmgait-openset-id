"""Open-set rejection thresholds calibrated WITHOUT impostor data, plus an oracle."""
import numpy as np
from scipy.stats import weibull_min


def roc_youden(pos, neg):
    """Global threshold maximising tpr - fpr. pos = genuine scores, neg = impostor scores."""
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s)
    s, y = s[o], y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(len(pos), 1); fpr = fp / max(len(neg), 1)
    j = np.argmax(tpr - fpr)
    return float(s[j]), float(np.trapezoid(tpr[np.argsort(fpr)], np.sort(fpr)))


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    r = np.argsort(np.argsort(s)) + 1
    n1, n0 = len(pos), len(neg)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def loko_threshold(cos_val, y_val, n_cls):
    """Leave-one-known-out. For each validation window of class c, the best cosine to
    a gallery with c removed is a stand-in for an impostor score. Uses known data only."""
    genuine, pseudo = [], []
    for c in range(n_cls):
        m = y_val == c
        if m.sum() == 0:
            continue
        genuine.append(cos_val[m].max(1))
        others = [j for j in range(n_cls) if j != c]
        pseudo.append(cos_val[m][:, others].max(1))
    g, p = np.concatenate(genuine), np.concatenate(pseudo)
    tau, _ = roc_youden(g, p)
    return np.full(n_cls, tau), dict(genuine_mean=float(g.mean()), pseudo_mean=float(p.mean()),
                                     separation_auroc=auroc(g, p))


def evt_threshold(cos_val, y_val, n_cls, target_frr=0.05, tail_frac=0.25):
    """Peaks-over-threshold Weibull fit to the LOW tail of each class's genuine scores.
    Sets a per-class threshold that should reject target_frr of genuine samples.
    Uses known data only, and never looks at a second class."""
    taus, info = np.zeros(n_cls), []
    for c in range(n_cls):
        s = cos_val[y_val == c].max(1)
        if len(s) < 30:
            taus[c] = np.quantile(s, target_frr) if len(s) else 0.0
            info.append(dict(cls=c, n=int(len(s)), fitted=False)); continue
        anchor = s.max() + 1e-6
        d = anchor - s                                   # low score -> large d
        u = np.quantile(d, 1 - tail_frac)                # POT threshold
        exc = d[d > u] - u
        if len(exc) < 10:
            taus[c] = np.quantile(s, target_frr)
            info.append(dict(cls=c, n=int(len(s)), fitted=False)); continue
        shape, loc, scale = weibull_min.fit(exc, floc=0)
        # P(D > d*) = target_frr, with P(D > u) = tail_frac
        q = 1.0 - target_frr / tail_frac                 # conditional CDF level
        d_star = u + weibull_min.ppf(q, shape, loc=0, scale=scale)
        taus[c] = anchor - d_star
        info.append(dict(cls=c, n=int(len(s)), fitted=True,
                         shape=float(shape), scale=float(scale), tau=float(taus[c])))
    return taus, dict(per_class=info)


def percentile_threshold(cos_val, y_val, n_cls, target_frr=0.05):
    """Plain empirical quantile of the genuine scores. The control for EVT:
    if the Weibull fit is not earning its place, this matches it."""
    taus = np.zeros(n_cls)
    for c in range(n_cls):
        s = cos_val[y_val == c].max(1)
        taus[c] = np.quantile(s, target_frr) if len(s) else 0.0
    return taus, {}


def conformal_threshold(cos_val, y_val, n_cls, target_frr=0.05):
    """Split-conformal threshold. Same job as percentile_threshold, but with the
    finite-sample correction that makes the false-rejection rate a guarantee
    instead of an estimate.

    For n exchangeable calibration scores and one new exchangeable genuine test
    score, the test score's rank among all n+1 values (n calibration + itself)
    is uniform on {1, ..., n+1} -- this is the basic conformal exchangeability
    lemma (Vovk et al.; see Bates, Candes et al. "Testing for outliers with
    conformal p-values" for the one-sided novelty-detection form used here).
    So if tau is the k-th smallest calibration score with
    k = floor(target_frr * (n + 1)), then

        P(new genuine score < tau) <= k / (n + 1) <= target_frr

    exactly, for any n, with no distributional assumption beyond exchangeability
    (which sits on the same footing as splitting val/test iid, already assumed
    everywhere else in this file). Plain np.quantile(s, target_frr), what
    percentile_threshold uses, has neither property: it is a point estimate of
    a population quantile with no correction for n being finite, so for a small
    per-class validation count it can be arbitrarily optimistic (or, via linear
    interpolation past the sample minimum, produce a threshold below every
    calibration score it was fit on).

    When k < 1 (too few calibration windows to make any guarantee at this
    target_frr), the threshold is -inf: refuse to reject rather than fabricate
    a number, which is the same failure mode that made leave-one-known-out's
    variance blow up on the classes with the fewest validation windows.

    Uses known data only, same as percentile_threshold and evt_threshold.
    """
    taus, info = np.zeros(n_cls), []
    for c in range(n_cls):
        s = np.sort(cos_val[y_val == c].max(1))
        n = len(s)
        tau, k = order_stat_tau(s, target_frr)
        taus[c] = tau
        info.append(dict(cls=c, n=n, k=k, guaranteed_frr=(float(k / (n + 1)) if k >= 1 else None)))
    return taus, dict(per_class=info)


def order_stat_tau(sorted_genuine, alpha):
    """Core of split-conformal calibration, factored out of conformal_threshold for
    readability. sorted_genuine must already be sorted ascending. Returns (tau, k);
    tau is -inf when k < 1 (not enough calibration points to guarantee anything at
    this alpha).
    """
    n = len(sorted_genuine)
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return -np.inf, k
    k = min(k, n)
    return float(sorted_genuine[k - 1]), k


def oracle_threshold(cos_known, cos_unknown, n_cls):
    """Cheats: uses real unknown subjects. Upper bound only."""
    tau, _ = roc_youden(cos_known.max(1), cos_unknown.max(1))
    return np.full(n_cls, tau), {}
