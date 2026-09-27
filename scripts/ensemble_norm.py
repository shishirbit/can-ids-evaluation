"""Two post-hoc fixes for the tight-false-alarm-budget regime, evaluated on saved predictions.

1. seed ensembling  - average the per-window probabilities of the 3 seeds (no retraining)
2. causal per-capture normalisation - divide each window's score by a running robust scale of the
   *preceding* windows of the same drive:  s'_t = (s_t - med_{t-W..t-1}) / (IQR_{t-W..t-1} + eps).
   Causal (uses only the past), online, and removes per-drive score offsets that break threshold
   transfer between recordings.

    python scripts/ensemble_norm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from event_curve import load
from galaxy_its.eval.operating import event_curve

DT, RULE, BUDGETS = 0.1, (3, 5), (5, 10, 30, 60)


def causal_norm(s: np.ndarray, window: int = 600, eps: float = 1e-3) -> np.ndarray:
    """Running median/IQR of the preceding `window` windows (default 60 s at dt=0.1 s)."""
    out = np.zeros_like(s, dtype=np.float64)
    for t in range(len(s)):
        past = s[max(0, t - window):t]
        if len(past) < 50:                      # warm-up: leave the raw score
            out[t] = s[t]
            continue
        q1, med, q3 = np.percentile(past, [25, 50, 75])
        out[t] = (s[t] - med) / (q3 - q1 + eps)
    return out


def detected_at(seqs, names, budgets=BUDGETS):
    c = event_curve(seqs, names, *RULE, DT)
    return {b: max((p["detected"] for p in c if p["fa_per_h"] <= b and p["duty"] <= 0.02), default=0)
            for b in budgets}, c[0]["events"]


def main():
    variants = {"graph": "experiments/road_v3_graph{s}_fold*", "nograph": "experiments/road_v3_nograph{s}_fold*"}
    print(f"{'model':34s} " + " ".join(f"@{b}/h" for b in BUDGETS))
    for v, pat in variants.items():
        per_seed = [load(pat.format(s=s), "sttf") for s in ("", "_s1", "_s2")]
        names = per_seed[0][1]
        for tag, seqs in (("seed0", per_seed[0][0]), ("seed1", per_seed[1][0]), ("seed2", per_seed[2][0])):
            got, n = detected_at(seqs, names)
            print(f"{v + ' ' + tag:34s} " + " ".join(f"{got[b]:5d}" for b in BUDGETS))
        # 1. ensemble: mean probability per window across seeds (same folds/order)
        ens = [(np.mean([per_seed[k][0][i][0] for k in range(3)], 0), per_seed[0][0][i][1])
               for i in range(len(names))]
        got, n = detected_at(ens, names)
        print(f"{v + ' ENSEMBLE(3 seeds)':34s} " + " ".join(f"{got[b]:5d}" for b in BUDGETS))
        # 2. causal per-capture normalisation, on the ensemble and on seed 0
        for tag, src in (("seed0 + causal-norm", per_seed[0][0]), ("ENSEMBLE + causal-norm", ens)):
            norm = [(causal_norm(s), inj) for s, inj in src]
            got, n = detected_at(norm, names)
            print(f"{v + ' ' + tag:34s} " + " ".join(f"{got[b]:5d}" for b in BUDGETS))
    print(f"(events = {n})")


if __name__ == "__main__":
    main()
