"""Compare models at matched false-alarm budgets, with significance tests.

    python scripts/compare_models.py

Per model: events detected at each budget, per seed (mean ± std) and for the seed ensemble.
Tests:
  * across seeds  - Mann-Whitney U on per-seed detected counts (independent runs)
  * paired events - McNemar exact test on *which* of the 29 events each ensemble detects at the
                    same budget (the events are the same, so the comparison is paired)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from event_curve import load
from galaxy_its.eval.operating import event_curve, event_detection, false_alarms_per_hour

RULE, DT, BUDGETS, MAX_DUTY = (3, 5), 0.1, (5, 10, 30), 0.02

MODELS = {                                    # label -> (pattern with {s}, kind)
    "STTF graph": ("experiments/road_v3_graph{s}_fold*", "sttf"),
    "ConvGRU": ("experiments/baseline_frames/convgru{s}_fold*/test_seqs.npz", "frames"),
    "ConvGRU+payload": ("experiments/baseline_frames/convgru_payload{s}_fold*/test_seqs.npz", "frames"),
}
SEEDS = ("", "_s1", "_s2", "_s3", "_s4")


def detected_at(seqs, names, budget):
    """Events detected at the most sensitive threshold meeting the budget and duty cap."""
    cand = np.unique(np.concatenate([s for s, _ in seqs]))
    cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), 200)).astype(int)]
    best, mask = 0, None
    for t in cand:
        fa, _, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
        if fa <= budget and duty <= MAX_DUTY:
            rows = event_detection(seqs, names, t, *RULE, DT)
            if sum(r["detected"] for r in rows) > best:
                best = sum(r["detected"] for r in rows)
                mask = np.array([r["detected"] for r in rows])
    return best, (mask if mask is not None else np.zeros(1, bool))


def main():
    per_seed, ens, masks = {}, {}, {}
    for label, (pat, kind) in MODELS.items():
        runs = []
        for s in SEEDS:
            seqs, names = load(pat.format(s=s), kind)
            if seqs:
                runs.append(seqs)
        if not runs:
            print(f"skip {label}: no runs"); continue
        per_seed[label] = {b: [detected_at(r, names, b)[0] for r in runs] for b in BUDGETS}
        e = [(np.mean([r[i][0] for r in runs], 0), runs[0][i][1]) for i in range(len(names))]
        ens[label] = {}; masks[label] = {}
        for b in BUDGETS:
            ens[label][b], masks[label][b] = detected_at(e, names, b)
        print(f"{label:18s} seeds={len(runs)} " +
              " ".join(f"@{b}/h {np.mean(per_seed[label][b]):5.1f}±{np.std(per_seed[label][b]):4.1f} "
                       f"(ens {ens[label][b]:2d})" for b in BUDGETS))

    print("\nMann-Whitney U across seeds (per-seed detected counts):")
    for a, b in combinations(per_seed, 2):
        for bud in BUDGETS:
            x, y = per_seed[a][bud], per_seed[b][bud]
            if len(x) > 1 and len(y) > 1:
                u = mannwhitneyu(x, y, alternative="two-sided")
                print(f"  @{bud:2d}/h {a} vs {b}: {np.mean(x):.1f} vs {np.mean(y):.1f}, p={u.pvalue:.3f}")

    print("\nMcNemar exact test on paired events (ensembles, same budget):")
    for a, b in combinations(masks, 2):
        for bud in BUDGETS:
            ma, mb = masks[a][bud], masks[b][bud]
            if ma.shape != mb.shape:
                continue
            n01 = int((~ma & mb).sum()); n10 = int((ma & ~mb).sum())
            if n01 + n10 == 0:
                print(f"  @{bud:2d}/h {a} vs {b}: identical event sets"); continue
            p = binomtest(n10, n01 + n10, 0.5).pvalue
            print(f"  @{bud:2d}/h {a} vs {b}: {a} only {n10}, {b} only {n01}, p={p:.3f}")


if __name__ == "__main__":
    main()
