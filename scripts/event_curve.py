"""Event-level trade-off curves: attacks detected vs false alarms/h, pooled over ROAD folds.

    python scripts/event_curve.py

Reads STTF test predictions (experiments/road_v*_{graph,nograph}_fold*/…/test_preds.npz) and
non-deep detector scores (experiments/road_detectors_v4/scores_<det>_fold*.npz), sweeps the alarm
threshold on test for a given k-of-n rule, and writes a markdown table plus a PNG figure.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from detect_road import by_sequence
from galaxy_its.eval.operating import event_curve

DT, RULE = 0.1, (3, 5)
MODELS = {
    "STTF v1 graph (timing only)": ("experiments/road_graph_fold*", "sttf"),
    "STTF v3 no-graph": ("experiments/road_v3_nograph_fold*", "sttf"),
    "STTF v3 graph": ("experiments/road_v3_graph_fold*", "sttf"),
    "rate exceedance": ("experiments/road_detectors_v4/scores_rate_fold*.npz", "det"),
    "payload novelty": ("experiments/road_detectors_v4/scores_surprise_fold*.npz", "det"),
    "window GBDT": ("experiments/road_detectors_v4/scores_window_gbdt_fold*.npz", "det"),
    "ConvGRU (frames, sup.)": ("experiments/baseline_frames/convgru_fold*/test_seqs.npz", "frames"),
    "FrameAE (frames, unsup.)": ("experiments/baseline_frames/frameae_fold*/test_seqs.npz", "frames"),
}


def load(pattern: str, kind: str):
    seqs, names = [], []
    for p in sorted(ROOT.glob(pattern)):
        if kind == "frames":                 # scripts/baseline_frames.py output
            z = np.load(p, allow_pickle=True)
            for nm, sc, inj in zip(z["names"], z["p"], z["inj"]):
                seqs.append((np.asarray(sc, float), np.asarray(inj, np.int8))); names.append(str(nm))
            continue
        if kind == "sttf":
            preds = sorted(p.glob("*/test_preds.npz")) if p.is_dir() else []
            if not preds:
                continue                     # .log files and crashed runs
            z = np.load(preds[-1])
        else:
            z = np.load(p)
        p0 = z["p"][:, 0] if z["p"].ndim > 1 else z["p"]
        y0 = z["y"][:, 0] if z["y"].ndim > 1 else z["y"]
        for i, s in by_sequence(p0, y0, z["seq"], z["gidx"]):
            seqs.append(s); names.append(str(z["seq_names"][i]))
    return seqs, names


def main():
    rows, curves = [], {}
    for label, (pattern, kind) in MODELS.items():
        try:
            seqs, names = load(pattern, kind)
        except (IndexError, ValueError) as e:
            print(f"skip {label}: {e}"); continue
        if not seqs:
            print(f"skip {label}: no predictions found ({pattern})"); continue
        c = event_curve(seqs, names, *RULE, DT)
        curves[label] = c
        n_events = c[0]["events"]
        best = {}
        for budget in (10, 30, 60):
            ok = [p for p in c if p["fa_per_h"] <= budget and p["duty"] <= 0.02]
            best[budget] = max((p["detected"] for p in ok), default=0)
        rows.append((label, n_events, best))
        print(f"{label:30s} events={n_events}  detected@10/h {best[10]:2d}  @30/h {best[30]:2d}  @60/h {best[60]:2d}")

    md = ["| model | events | detected @10 FA/h | @30 FA/h | @60 FA/h |", "|---|---|---|---|---|"]
    md += [f"| {l} | {n} | {b[10]} | {b[30]} | {b[60]} |" for l, n, b in rows]
    out = ROOT / "experiments" / "event_curves.md"
    out.write_text("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, c in curves.items():
            c = sorted(c, key=lambda p: p["fa_per_h"])
            ax.plot([p["fa_per_h"] for p in c], [p["detected"] for p in c], marker=".", label=label)
        ax.set_xscale("symlog"); ax.set_xlabel("false alarms per hour (held-out normal driving)")
        ax.set_ylabel(f"attack events detected (of {rows[0][1]})")
        ax.set_title(f"ROAD, {RULE[0]}-of-{RULE[1]} alarm rule, 3-fold pooled")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(ROOT / "experiments" / "event_curves.png", dpi=150)
        print("wrote experiments/event_curves.png")
    except ImportError:
        pass
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
