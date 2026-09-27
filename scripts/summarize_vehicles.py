"""Summarise cross-vehicle results over all can-train-and-test vehicle pairs.

    python scripts/summarize_vehicles.py

Reads experiments/id_agnostic/set_XX_s*.json (ID-agnostic, 3 seeds) and
experiments/cross_vehicle/set_XX_convgru*_s0/results.json (ID-based) and prints, per pair, the
events detected at a false-alarm budget with the budget honoured on test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"set_01": "Impala -> Silverado", "set_02": "Traverse -> Forester",
         "set_03": "Silverado -> Forester", "set_04": "Forester -> Traverse"}
KEYS = {"known": "test_01_known_vehicle_known_attack", "unknown": "test_02_unknown_vehicle_known_attack",
        "masq": "test_06_masquerade"}


def ida(setname, features="relative"):
    out = {}
    files = sorted((ROOT / "experiments/id_agnostic").glob(f"{setname}_{features}_s*.json"))
    if not files:
        return None
    runs = [json.load(open(f))["results"] for f in files]
    for label, sub in KEYS.items():
        for budget in (5, 30):
            k = f"{sub}|target-own|{budget}"
            det = [r[k]["detected"] for r in runs if k in r]
            fa = [r[k]["fa_per_h"] for r in runs if k in r]
            ev = [r[k]["events"] for r in runs if k in r]
            valid = [r[k].get("valid", True) for r in runs if k in r]
            if det:
                out[(label, budget)] = (np.mean(det), np.std(det), ev[0], np.mean(fa), sum(valid), len(det))
    return out


def idbased(setname):
    out = {}
    for tag in (f"{setname}_convgru_s0", f"{setname}_convgru_payload_s0"):
        f = ROOT / "experiments/cross_vehicle" / tag / "results.json"
        if not f.exists():
            continue
        r = json.load(open(f))["results"]
        for label, sub in KEYS.items():
            for budget in (5, 30):
                k = f"{sub}|target-own|{budget}"
                if k in r:
                    out[(label, budget)] = (r[k]["detected"], r[k]["events"], r[k]["fa_per_h"], r[k]["duty"])
        break
    return out


def main():
    print(f"{'pair':24s} {'test':8s} {'ID-based':>10s} {'residualisation':>17s} {'ours (relative)':>17s}")
    rows = []
    for s, name in PAIRS.items():
        a, b, c = ida(s, "relative"), idbased(s), ida(s, "residual")
        if not a:
            print(f"{name:26s} (no ID-agnostic results yet)"); continue
        for label in ("known", "unknown", "masq"):
            base = b.get((label, 5))
            g5 = a.get((label, 5))
            r5 = (c or {}).get((label, 5))
            if not g5:
                continue
            bs = f"{base[0]}/{base[1]}" if base else "-"
            rs = f"{r5[0]:.1f}+-{r5[1]:.1f}/{r5[2]}" if r5 else "-"
            print(f"{name:24s} {label:8s} {bs:>10s} {rs:>17s} "
                  f"{f'{g5[0]:.1f}+-{g5[1]:.1f}/{g5[2]}':>17s}")
            rows.append({"pair": name, "test": label, "id_based_at5": bs,
                         "residual_at5": r5[:3] if r5 else None, "relative_at5": g5[:3],
                         "relative_fa_per_h": g5[3], "residual_fa_per_h": r5[3] if r5 else None})
    (ROOT / "experiments/vehicle_pairs.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\nsaved -> experiments/vehicle_pairs.json")


if __name__ == "__main__":
    main()
