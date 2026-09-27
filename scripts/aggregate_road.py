"""Pool ROAD cross-validation folds and report detection/forecasting results.

    python scripts/aggregate_road.py experiments/road_graph_fold0 experiments/road_graph_fold1 ...

For each run dir the latest timestamped sub-dir is used. Every attack event is tested exactly
once across the 3 folds, so pooling gives all 15 injection events (+ masquerade copies).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_its.eval.metrics import evaluate, format_table

DT = 0.1


def latest(run: Path) -> Path:
    return sorted(p for p in run.iterdir() if p.is_dir())[-1]


def per_capture(z, p0, thr):
    """Latency / false alarms per test sequence."""
    rows = []
    for s in np.unique(z["seq"]):
        m = z["seq"] == s
        order = np.argsort(z["gidx"][m])
        inj = z["y"][m][order, 0]
        alarm = p0[m][order] >= thr
        name = str(z["seq_names"][s])
        if inj.any():
            on = int(np.argmax(inj))
            end = on + int(np.argmax(inj[on:] == 0)) if (inj[on:] == 0).any() else len(inj)
            hit = np.where(alarm[on:end])[0]
            pre_alarms = int(alarm[:on].sum())
            rows.append({"name": name, "attack": True, "detected": bool(len(hit)),
                         "latency_ms": float(hit[0] * DT * 1000) if len(hit) else np.nan,
                         "false_alarm_windows_before_onset": pre_alarms})
        else:
            rows.append({"name": name, "attack": False, "hours": len(inj) * DT / 3600,
                         "false_alarm_windows": int(alarm.sum())})
    return rows


def family(name: str) -> str:
    base = name.removesuffix("_masquerade")
    return base.rsplit("_", 1)[0] if base[-1].isdigit() else base


def report(runs: list[Path], label: str):
    zs = [np.load(latest(r) / "test_preds.npz") for r in runs]
    horizons = json.load(open(latest(runs[0]) / "results.json"))["args"]["horizons"]
    y = np.concatenate([z["y"] for z in zs]); clean = np.concatenate([z["clean"] for z in zs])
    out = {}
    for model, key, tkey in (("STTF", "p", "thr"), ("timing", "p_timing", "thr_timing")):
        p = np.concatenate([z[key] for z in zs])
        # thresholds were tuned per fold on that fold's validation set; apply them per fold
        pred_bin = np.concatenate([(z[key] >= z[tkey][None]) for z in zs]).astype(float)
        res = evaluate(y, p, clean, horizons)
        # replace threshold-dependent metrics with the per-fold-threshold versions
        res_bin = evaluate(y, pred_bin, clean, horizons, [0.5] * len(horizons))
        for h in res:
            for sub in res[h]:
                for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "mcc"):
                    res[h][sub][k] = res_bin[h][sub][k]
        out[model] = res
        print(f"\n=== {label}: {model} (pooled over {len(zs)} folds) ===\n{format_table(res)}")

    rows = [r for z in zs for r in per_capture(z, z["p"][:, 0], float(z["thr"][0]))]
    att = [r for r in rows if r["attack"]]
    amb = [r for r in rows if not r["attack"]]
    hours = sum(r["hours"] for r in amb); fa = sum(r["false_alarm_windows"] for r in amb)
    print(f"\n--- {label}: detection per attack family (0 s head, per-fold val threshold) ---")
    fam = defaultdict(list)
    for r in att:
        fam[family(r["name"])].append(r)
    print(f"{'family':32s} {'events':>6} {'detected':>8} {'median_ms':>9} {'max_ms':>7}")
    for f, rs in sorted(fam.items()):
        lat = [r["latency_ms"] for r in rs if r["detected"]]
        print(f"{f:32s} {len(rs):6d} {sum(r['detected'] for r in rs):8d} "
              f"{np.median(lat) if lat else float('nan'):9.0f} {max(lat) if lat else float('nan'):7.0f}")
    lat_all = [r["latency_ms"] for r in att if r["detected"]]
    print(f"ALL: {sum(r['detected'] for r in att)}/{len(att)} detected, median latency "
          f"{np.median(lat_all):.0f} ms; ambient false-alarm windows: {fa} in {hours:.2f} h "
          f"({fa / max(hours, 1e-9):.1f} per hour)")
    return {"metrics": out, "captures": rows}


def operating(runs: list[Path], label: str):
    """Event detection at false-alarm budgets, thresholds calibrated on each fold's validation."""
    from detect_road import by_sequence, print_operating
    from galaxy_its.eval.operating import operating_report
    folds = []
    for r in runs:
        d = latest(r)
        if not (d / "val_preds.npz").exists():
            print(f"{label}: no val_preds.npz in {d} (re-run eval with --ckpt)"); return
        v, t = np.load(d / "val_preds.npz"), np.load(d / "test_preds.npz")
        names = t["seq_names"]
        vs = by_sequence(v["p"][:, 0], v["y"][:, 0], v["seq"], v["gidx"])
        ts = by_sequence(t["p"][:, 0], t["y"][:, 0], t["seq"], t["gidx"])
        folds.append(operating_report([s for _, s in vs], [s for _, s in ts],
                                      [str(names[i]) for i, _ in ts], DT))
    print_operating({label: folds})


def main():
    groups = defaultdict(list)
    for a in sys.argv[1:]:
        p = Path(a)
        groups[p.name.rsplit("_fold", 1)[0]].append(p)
    summary = {}
    for g, r in groups.items():
        summary[g] = report(r, g)
        operating(r, g)
    Path("experiments/road_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nsaved -> experiments/road_summary.json")


if __name__ == "__main__":
    main()
