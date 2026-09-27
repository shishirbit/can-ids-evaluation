"""Non-deep detection baselines, one per level of the detectability hierarchy.

    RateDetector      per-ID frame-count exceedance (unsupervised)   -> rate attacks
    SurpriseDetector  per-ID, per-byte payload surprisal exceedance  -> value attacks
                      (unsupervised)
    WindowGBDT        gradient-boosted trees on the current window's -> supervised per-window
                      flattened node features (no graph, no history)    upper bound without
                                                                        relational/temporal model
All operate on the same window features as STTF: x [T, N, F] with F = NODE_FEATURES + PAYLOAD_FEATURES.
Scores are per window; higher = more anomalous.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

LOG_COUNT = 0            # index of log_count in NODE_FEATURES
N_TIMING = 8             # len(NODE_FEATURES)
SURPRISE = slice(N_TIMING + 16, N_TIMING + 24)


class _Exceedance:
    """Per-channel exceedance beyond the normal range, max over channels.

    score = max_c (v_c - q_hi_c) / s_c  (and (q_lo_c - v_c) / s_c if two-sided), where q are
    the 0.1 / 99.9 percentiles on attack-free training windows and s_c is the channel's
    inter-percentile range floored at `floor`. A plain max z-score over hundreds of channels is
    dominated by near-constant channels (tiny std -> huge z for any wobble).
    """
    two_sided = False
    floor = 0.05

    def _channels(self, x):
        raise NotImplementedError

    def fit(self, x_normal: np.ndarray):
        c = self._channels(x_normal).reshape(len(x_normal), -1)
        self.lo, self.hi = np.percentile(c, 0.1, 0), np.percentile(c, 99.9, 0)
        self.s = np.maximum(self.hi - self.lo, self.floor)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        c = self._channels(x).reshape(len(x), -1)
        e = (c - self.hi) / self.s
        if self.two_sided:
            e = np.maximum(e, (self.lo - c) / self.s)
        return e.max(1)


class RateDetector(_Exceedance):
    """Per-ID frame count outside its normal range (flooding, injection, suspension)."""
    two_sided = True

    def _channels(self, x):
        return x[:, :, LOG_COUNT]


class SurpriseDetector(_Exceedance):
    """Per-(ID, byte) payload surprisal above its normal range (never-seen values)."""

    def _channels(self, x):
        return x[:, :, SURPRISE]


class WindowGBDT:
    def fit(self, x: np.ndarray, y: np.ndarray, max_rows: int = 60000, seed: int = 0):
        rng = np.random.default_rng(seed)
        if len(x) > max_rows:
            keep = rng.choice(len(x), max_rows, replace=False)
            x, y = x[keep], y[keep]
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                  class_weight="balanced", random_state=seed)
        self.clf.fit(x.reshape(len(x), -1), y)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(x.reshape(len(x), -1))[:, 1]
