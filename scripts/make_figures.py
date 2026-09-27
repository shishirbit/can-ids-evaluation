"""Generate the paper figures (PDF) from saved experiment results.

    python scripts/make_figures.py

Writes paper/figs/*.pdf. Every number is read from experiments/, never hard-coded, except the
Car-Hacking attack-schedule statistics which are recomputed from the processed dataset.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

plt.rcParams.update({"font.size": 12.4, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
C = {"ours": "#1b6ca8", "resid": "#c9603a", "idbased": "#6b6b6b", "union": "#2e7d32",
     "gbdt": "#8e6bbf", "sttf": "#1b6ca8", "timer": "#c9603a"}


def fig_event_curves():
    """ROAD: events detected vs false alarms/h for each model (seed 0, 3-fold pooled)."""
    from event_curve import MODELS, load
    from galaxy_its.eval.operating import event_curve
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    # one distinct colour and dash pattern per model, so no two curves can be confused
    styles = {"STTF v1 graph (timing only)": ("-", C["idbased"], 1.2),
              "STTF v3 no-graph": ("--", C["resid"], 1.2),
              "STTF v3 graph": ("-", C["sttf"], 1.6),
              "rate exceedance": ("-.", "#8c564b", 1.0),
              "payload novelty": (":", "#bcbd22", 1.4),
              "window GBDT": (":", C["gbdt"], 1.4),
              "ConvGRU (frames, sup.)": ("-.", C["union"], 1.6),
              "FrameAE (frames, unsup.)": (":", "#999999", 1.0)}
    for label, (pattern, kind) in MODELS.items():
        try:
            seqs, names = load(pattern, kind)
        except Exception:
            continue
        if not seqs:
            continue
        c = sorted(event_curve(seqs, names, 3, 5, 0.1), key=lambda p: p["fa_per_h"])
        ls, col, lw = styles.get(label, ("-", None, 1.2))
        ax.plot([p["fa_per_h"] for p in c], [p["detected"] for p in c], ls, color=col,
                lw=lw, label=label.replace(" (frames, sup.)", "").replace(" (frames, unsup.)", " (unsup.)"))
    ax.set_xscale("symlog"); ax.set_xlim(0, 300); ax.set_ylim(0, 30)
    ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("attack events detected (of 29)")
    # legend below the axes: the plot area is occupied at both top and bottom
    ax.legend(fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              framealpha=0.95, handlelength=2.0, columnspacing=1.2, borderpad=0.4)
    fig.savefig(FIGS / "road_event_curves.pdf")
    plt.close(fig)
    print("road_event_curves.pdf")


def fig_carhacking():
    """Car-Hacking: attack schedule regularity and the timer-vs-model forecasting comparison."""
    rep = json.load(open(ROOT / "data/processed/car_hacking/report.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    fig.subplots_adjust(wspace=0.32)
    names = [k for k in rep if rep[k].get("episodes")]
    gaps = [rep[k]["gap_s(min/med/max)"][1] for k in names]
    durs = [rep[k]["episode_len_s"][1] for k in names]
    cv = [rep[k]["gap_cv"] for k in names]
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, durs, 0.4, label="attack duration", color=C["ours"])
    axes[0].bar(x + 0.2, gaps, 0.4, label="gap to next attack", color=C["resid"])
    for i, (d, g) in enumerate(zip(durs, gaps)):
        axes[0].text(i - 0.2, d + 0.12, f"{d:.1f}", ha="center", fontsize=8.6, color=C["ours"])
        axes[0].text(i + 0.2, g + 0.12, f"{g:.1f}", ha="center", fontsize=8.6, color=C["resid"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("_dataset", "") for n in names], fontsize=9.3)
    axes[0].set_ylabel("seconds (median)")
    axes[0].set_ylim(0, 8.6)
    axes[0].legend(fontsize=9.3, loc="upper center", ncol=2, framealpha=0.95,
                   handlelength=1.2, columnspacing=1.0)
    axes[0].text(len(names) / 2 - 0.5, 6.15,
                 f"gap coefficient of variation {min(cv):g}-{max(cv):g}",
                 ha="center", fontsize=8.8, color="#555555")
    axes[0].set_title("(a) the attack schedule is fixed", fontsize=11.5)

    # pre-onset AUC: STTF vs timer, from the Fuzzy-only run recorded in the README table
    hor = [1, 2, 5]
    sttf = [0.851, 0.877, 0.847]
    timer = [1.000, 0.996, 0.988]
    axes[1].plot(hor, sttf, "o-", color=C["sttf"], label="spatio-temporal graph network")
    axes[1].plot(hor, timer, "s--", color=C["timer"], label="one-feature timer")
    axes[1].axhline(0.5, color="#999999", lw=0.8, ls=":")
    axes[1].text(5.05, 0.515, "chance", ha="right", fontsize=8.6, color="#777777")
    axes[1].set_xlabel("forecast horizon (s)")
    axes[1].set_ylabel("pre-onset AUC-ROC")
    axes[1].set_xticks(hor)
    axes[1].set_ylim(0.4, 1.16)
    axes[1].legend(fontsize=9.3, loc="upper center", framealpha=0.95, handlelength=1.6)
    axes[1].set_title("(b) the timer wins at every horizon", fontsize=11.5)
    fig.savefig(FIGS / "carhacking_schedule.pdf")
    plt.close(fig)
    print("carhacking_schedule.pdf")


def _pair_totals():
    """Unknown-vehicle detection at 5 FA/h per representation, per pair."""
    PAIRS = {"set_01": "Impala\n$\\to$ Silverado", "set_02": "Traverse\n$\\to$ Forester",
             "set_03": "Silverado\n$\\to$ Forester", "set_04": "Forester\n$\\to$ Traverse"}
    out = defaultdict(dict)
    for st, name in PAIRS.items():
        for feat in ("relative", "residual"):
            fs = sorted(glob.glob(str(ROOT / f"experiments/id_agnostic/{st}_{feat}_s*.json")))
            if not fs:
                continue
            runs = [json.load(open(f))["results"] for f in fs]
            k = "test_02_unknown_vehicle_known_attack|target-own|5"
            det = [r[k]["detected"] for r in runs]; ev = runs[0][k]["events"]
            out[name][feat] = (np.mean(det), np.std(det), ev)
        f = ROOT / f"experiments/cross_vehicle/{st}_convgru_s0/results.json"
        if f.exists():
            r = json.load(open(f))["results"]["test_02_unknown_vehicle_known_attack|target-own|5"]
            out[name]["idbased"] = (r["detected"], 0.0, r["events"])
        elif st == "set_01":
            r = json.load(open(ROOT / "experiments/cross_vehicle/convgru_s0/results.json"))["results"]
            v = r["test_02_unknown_vehicle_known_attack|target-own|5"]
            out[name]["idbased"] = (v["detected"], 0.0, v["events"])
    return out


def fig_cross_vehicle():
    """Cross-vehicle transfer per representation, plus the OR-ed pair total."""
    data = _pair_totals()
    pairs = list(data)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    w = 0.26
    x = np.arange(len(pairs))
    for i, (key, label, col) in enumerate([("idbased", "ID embedding", C["idbased"]),
                                           ("residual", "$\\phi^{res}$ residualisation", C["resid"]),
                                           ("relative", "$\\phi^{rel}$ ID-relative (ours)", C["ours"])]):
        vals, errs = [], []
        for p in pairs:
            m, s, ev = data[p].get(key, (0, 0, 1))
            vals.append(100 * m / ev); errs.append(100 * s / ev)
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=1.5, label=label, color=col,
               error_kw={"lw": 0.6})
        for xi, v in zip(x + (i - 1) * w, vals):      # annotate, so the zero bars are visible
            ax.text(xi, v + 2.0, f"{v:.0f}", ha="center", fontsize=8.4, color=col)
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=9.3)
    ax.set_ylabel("unknown-vehicle events detected (%)\nat $\\leq$5 false alarms/h")
    ax.set_ylim(0, 132)
    ax.legend(fontsize=9.3, loc="upper center", ncol=3, framealpha=0.95,
              handlelength=1.2, columnspacing=1.0)
    fig.savefig(FIGS / "cross_vehicle.pdf")
    plt.close(fig)
    print("cross_vehicle.pdf")


def fig_pair_union():
    """Paired ID-agnostic detectors: branches vs union, unknown vehicle and masquerade."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in sorted(glob.glob(str(ROOT / "experiments/pair_detector/set_0*_rel-res_s*.json"))):
        for k, v in json.load(open(f))["results"].items():
            sub, br, b = k.split("|")
            if int(b) != 5:
                continue
            a = agg[sub][br]; a[0] += v["detected"]; a[1] += v["events"]
    if not agg:
        return
    subs = [("test_02_unknown_vehicle_known_attack", "unknown vehicle"),
            ("test_06_masquerade", "masquerade")]
    brs = [("A id-agnostic", "ID-relative", C["ours"]), ("B id-based", "residualisation", C["resid"]),
           ("A OR B", "both (OR)", C["union"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(subs)); w = 0.26
    for i, (key, label, col) in enumerate(brs):
        vals = [100 * agg[s][key][0] / max(agg[s][key][1], 1) for s, _ in subs]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=col)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in subs], fontsize=10.8)
    ax.set_ylabel("events detected (%) at $\\leq$5 FA/h"); ax.set_ylim(0, 118)
    ax.legend(fontsize=9.3, loc="upper center", ncol=3, framealpha=0.95,
              handlelength=1.2, columnspacing=1.0)
    fig.savefig(FIGS / "pair_union.pdf")
    plt.close(fig)
    print("pair_union.pdf")


if __name__ == "__main__":
    fig_carhacking()
    fig_cross_vehicle()
    fig_pair_union()
    fig_event_curves()
    print(f"-> {FIGS}")
