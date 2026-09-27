"""Evaluation metrics from the proposal (Section 4.3), computed per horizon.

Every horizon is reported on two subsets:
    all        every test window (mixes detection of ongoing attacks with forecasting)
    pre_onset  windows whose current window is attack-free -> genuine forecasting ability
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y = y.astype(int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "pos_rate": float(y.mean()) if len(y) else 0.0,
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
        "mcc": float(matthews_corrcoef(y, pred)) if both else float("nan"),
        "auc_roc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auc_pr": float(average_precision_score(y, p)) if both else float("nan"),
        # Brier only defined for probabilities (unsupervised detectors output raw scores)
        "brier": float(brier_score_loss(y, p)) if len(y) and p.min() >= 0 and p.max() <= 1 else float("nan"),
        "threshold": thr,
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Exact F1-maximising threshold over every distinct score (quantile grids miss it when
    scores saturate near 0/1)."""
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate(y: np.ndarray, p: np.ndarray, clean_now: np.ndarray, horizons_s: list[float],
             thresholds: list[float] | None = None) -> dict:
    """y, p: [n, K]; clean_now: [n] bool. Returns {horizon: {subset: metrics}}."""
    res = {}
    for k, h in enumerate(horizons_s):
        thr = thresholds[k] if thresholds else 0.5
        ok = y[:, k] >= 0
        res[f"{h:g}s"] = {
            "all": binary_metrics(y[ok, k], p[ok, k], thr),
            "pre_onset": binary_metrics(y[ok & clean_now, k], p[ok & clean_now, k], thr),
        }
    return res


def onset_latency(inj: np.ndarray, seq_of: np.ndarray, gidx: np.ndarray, p0: np.ndarray,
                  thr: float, dt: float) -> dict:
    """Detection latency per attack episode in the evaluated windows.

    inj, seq_of: per global window; gidx: evaluated global windows; p0: detection-head
    probabilities for gidx. Latency = first alarm inside the episode minus episode onset.
    Latency is quantised to the window size dt.
    """
    alarm = np.zeros(len(inj), bool)
    alarm[gidx] = p0 >= thr
    evaluated = np.zeros(len(inj), bool); evaluated[gidx] = True
    lat, missed = [], 0
    starts = np.where((inj[1:] == 1) & ((inj[:-1] == 0) | (seq_of[1:] != seq_of[:-1])))[0] + 1
    if len(inj) and inj[0] == 1:
        starts = np.concatenate([[0], starts])
    for s in starts:
        if not evaluated[s]:
            continue
        e = s
        while e + 1 < len(inj) and inj[e + 1] == 1 and seq_of[e + 1] == seq_of[s]:
            e += 1
        hit = np.where(alarm[s:e + 1])[0]
        if len(hit):
            lat.append(hit[0] * dt)
        else:
            missed += 1
    lat = np.asarray(lat)
    return {"episodes": len(lat) + missed, "detected": len(lat), "missed": missed,
            "latency_ms_median": float(np.median(lat) * 1000) if len(lat) else float("nan"),
            "latency_ms_mean": float(lat.mean() * 1000) if len(lat) else float("nan"),
            "latency_ms_max": float(lat.max() * 1000) if len(lat) else float("nan")}


def format_table(res: dict, keys=("f1", "auc_roc", "auc_pr", "brier", "fpr", "fnr", "mcc")) -> str:
    lines = [f"{'horizon':>8} {'subset':>10} {'pos%':>6} " + " ".join(f"{k:>8}" for k in keys)]
    for h, sub in res.items():
        for name, m in sub.items():
            lines.append(f"{h:>8} {name:>10} {100 * m['pos_rate']:6.2f} " +
                         " ".join(f"{m[k]:8.4f}" for k in keys))
    return "\n".join(lines)
