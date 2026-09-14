"""PointNet frame encoder + dilated temporal CNN + L2-normalised embedding, ArcFace head."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    """Shared per-point MLP then max-pool over points. Permutation invariant."""

    def __init__(self, cin=4, widths=(32, 64, 128)):
        super().__init__()
        layers, c = [], cin
        for w in widths:
            layers += [nn.Conv1d(c, w, 1, bias=False), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        self.net = nn.Sequential(*layers)
        self.out_dim = c

    def forward(self, x):                # x: (B*T, N, C)
        x = x.transpose(1, 2)            # (B*T, C, N)
        x = self.net(x)
        return x.max(dim=2).values       # (B*T, F)


class TemporalBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return F.relu(x + self.bn(self.conv(x)))


class GaitNet(nn.Module):
    def __init__(self, emb_dim=128, widths=(32, 64, 128), dilations=(1, 2, 4)):
        super().__init__()
        self.frame = FrameEncoder(4, widths)
        c = self.frame.out_dim
        self.temporal = nn.Sequential(*[TemporalBlock(c, d) for d in dilations])
        self.head = nn.Sequential(nn.Linear(2 * c, 256), nn.BatchNorm1d(256),
                                  nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(256, emb_dim))

    def forward(self, x):                # x: (B, T, N, C)
        B, T, N, C = x.shape
        f = self.frame(x.reshape(B * T, N, C)).reshape(B, T, -1).transpose(1, 2)
        f = self.temporal(f)             # (B, F, T)
        z = torch.cat([f.mean(2), f.max(2).values], 1)
        return F.normalize(self.head(z), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin. Gives a metric space where cosine to a class
    prototype is directly usable as an open-set score."""

    def __init__(self, emb_dim, n_classes, scale=16.0, margin=0.25):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def cosine(self, z):
        return z @ F.normalize(self.W, dim=1).t()

    def forward(self, z, y):
        cos = self.cosine(z).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        oh = F.one_hot(y, self.W.shape[0]).float()
        return F.cross_entropy(self.scale * torch.cos(theta + self.margin * oh), y)
