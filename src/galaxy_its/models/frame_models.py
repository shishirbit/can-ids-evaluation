"""Frame-sequence baselines, re-implemented in the form used by the CAN IDS literature.

Both consume a sliding window of F consecutive CAN frames (ID + 8 payload bytes) rather than the
aggregated per-ID window features STTF uses, so they are independent of our feature engineering.

    ConvGRU   supervised: ID embedding + byte channels -> 1D convolutions over the frame axis
              -> GRU -> window logit. Stands in for shallow CNN+GRU in-vehicle IDS
              (ConvGRU / CANGuard family).
    FrameAE   unsupervised: convolutional autoencoder over the same frame window, trained on
              attack-free frames only; anomaly score = reconstruction MSE. Stands in for
              reconstruction-based signal-level IDS (CANShield family).

These are re-implementations adapted to a common evaluation protocol, not the original authors'
code; they are reported as such.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, n_ids: int, d_id: int = 32, d: int = 64, dropout: float = 0.1, n_extra: int = 0,
                 n_channels: int = 8):
        """n_ids <= 0 disables the ID embedding entirely (ID-agnostic mode); n_channels is the
        number of per-frame value channels (8 payload bytes, or len(relative.FEATURES))."""
        super().__init__()
        self.emb = nn.Embedding(n_ids, d_id) if n_ids > 0 else None
        cin = (d_id if self.emb is not None else 0) + n_channels + n_extra
        self.conv = nn.Sequential(
            nn.Conv1d(cin, d, 5, padding=2), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Dropout(dropout))
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, ids: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        # ids [B, F] long; data [B, F, 8 (+8 novelty)] float
        x = (torch.cat([self.emb(ids), data], -1) if self.emb is not None else data).transpose(1, 2)
        h = self.conv(x).transpose(1, 2)
        o, _ = self.gru(h)
        return self.head(o.mean(1)).squeeze(-1)


class FrameAE(nn.Module):
    def __init__(self, n_ids: int, d_id: int = 16, d: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_ids, d_id)
        cin = d_id + 8
        self.enc = nn.Sequential(
            nn.Conv1d(cin, d, 5, padding=2, stride=2), nn.ReLU(),
            nn.Conv1d(d, d, 5, padding=2, stride=2), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(d, d, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(d, cin, 4, stride=2, padding=1))

    def forward(self, ids: torch.Tensor, data: torch.Tensor):
        x = torch.cat([self.emb(ids), data], -1).transpose(1, 2)
        z = self.dec(self.enc(x))
        return ((z - x) ** 2).mean((1, 2)), x                        # per-sample reconstruction MSE
