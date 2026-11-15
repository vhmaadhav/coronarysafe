"""Small 3D CNN for M4: corrupted 96^3 (2 mm) lumen mask in, critical logit + log-burden out."""

from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class PlausibilityNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(1, w, 2),          # 48
            _block(w, 2 * w, 2),      # 24
            _block(2 * w, 4 * w, 2),  # 12
            _block(4 * w, 8 * w, 2),  # 6
            _block(8 * w, 8 * w, 1),
        )
        self.head = nn.Sequential(nn.Linear(16 * w, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x):
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=(2, 3, 4)), f.amax(dim=(2, 3, 4))], dim=1)
        out = self.head(pooled)
        return out[:, 0], out[:, 1]  # critical logit, log1p(burden)
