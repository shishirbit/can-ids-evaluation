"""Sanity baselines every forecasting result must beat.

persistence   p(attack within H) = 1 if the current window is attacked, else 0.
              Beating it on `all` but not on `pre_onset` means the model only detects.
timing        gradient-boosted trees on two clocks only: time since the last attacked window and
              time since the recording started. No traffic content at all. If this forecasts
              onsets well, the dataset's injection schedule (Car-Hacking: fixed 3 s gaps) or
              capture protocol (ROAD: attacks start some seconds into each capture), not traffic
              precursors, is enough to solve the task.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def persistence(inj_now: np.ndarray, n_horizons: int) -> np.ndarray:
    return np.repeat(inj_now.astype(np.float32)[:, None], n_horizons, 1)


def _features(tsla, elapsed):
    return np.stack([np.minimum(tsla, 600.0), np.minimum(elapsed, 600.0)], 1)


def timing_fit_predict(tsla_tr, el_tr, y_tr, tsla_te, el_te) -> np.ndarray:
    Xtr, Xte = _features(tsla_tr, el_tr), _features(tsla_te, el_te)
    out = np.zeros((len(Xte), y_tr.shape[1]), np.float32)
    for k in range(y_tr.shape[1]):
        ok = y_tr[:, k] >= 0
        if len(np.unique(y_tr[ok, k])) < 2:
            out[:, k] = y_tr[ok, k].mean() if ok.any() else 0
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr[ok], y_tr[ok, k])
        out[:, k] = clf.predict_proba(Xte)[:, 1]
    return out
