"""mmGait10 windowed point-cloud dataset for open-set gait recognition."""
import numpy as np
import torch

NSTEPS = 30      # 3 s at 10 Hz
STEP = 10        # window hop in frames


def load(path):
    d = np.load(path, allow_pickle=False)
    return dict(X=d["X"], offsets=d["offsets"], labels=d["labels"],
                scenarios=d["scenarios"], tracks=d["tracks"])


def build_windows(D, track_ids):
    """Return (track_idx, start_frame) pairs for the given tracks."""
    out = []
    for t in track_ids:
        a, b = D["offsets"][t], D["offsets"][t + 1]
        n = b - a
        for s in range(0, n - NSTEPS + 1, STEP):
            out.append((t, a + s))
    return np.array(out, np.int64)


class WindowSet(torch.utils.data.Dataset):
    def __init__(self, D, win, label_map=None, train=False):
        self.X = D["X"]
        self.win = win
        self.raw_lab = D["labels"]
        self.label_map = label_map          # global subject id -> contiguous known-class id
        self.train = train

    def __len__(self):
        return len(self.win)

    def __getitem__(self, i):
        t, s = self.win[i]
        x = self.X[s:s + NSTEPS].astype(np.float32)          # (T, N, 4)
        if self.train:
            # light augmentation: random yaw about the vertical axis and point dropout
            th = np.random.uniform(-np.pi / 12, np.pi / 12)
            c, sn = np.cos(th), np.sin(th)
            xy = x[..., :2].copy()
            x[..., 0] = xy[..., 0] * c - xy[..., 1] * sn
            x[..., 1] = xy[..., 0] * sn + xy[..., 1] * c
            if np.random.rand() < 0.5:                        # mirror left/right
                x[..., 0] *= -1
            keep = np.random.rand(x.shape[1]) > 0.15          # drop 15% of points
            if keep.sum() >= 8:
                idx = np.where(keep)[0]
                idx = np.concatenate([idx, np.random.choice(idx, x.shape[1] - len(idx))])
                x = x[:, idx]
        g = int(self.raw_lab[t])
        y = self.label_map.get(g, -1) if self.label_map is not None else g
        return torch.from_numpy(x), y, int(t), int(s)


def split_tracks(D, known, unknown, seed):
    """Track-level split. Unknown subjects appear only in the test set."""
    rng = np.random.default_rng(seed)
    lab = D["labels"]
    tr, va, te_k = [], [], []
    for k in known:
        idx = np.where(lab == k)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr, nva = int(round(0.70 * n)), int(round(0.15 * n))
        tr += list(idx[:ntr]); va += list(idx[ntr:ntr + nva]); te_k += list(idx[ntr + nva:])
    te_u = [i for i in range(len(lab)) if lab[i] in unknown]
    return np.array(tr), np.array(va), np.array(te_k), np.array(te_u)
