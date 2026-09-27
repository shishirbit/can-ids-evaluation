"""System-architecture diagram and the remaining paper figures (PDF, no TikZ).

    python scripts/make_figures2.py

All values are read from experiments/paper_numbers.json, which records the script that produced
each number, so figures cannot drift from the text.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
N = json.load(open(ROOT / "experiments" / "paper_numbers.json"))

plt.rcParams.update({"font.size": 12.4, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#1b6ca8", "#c9603a", "#2e7d32", "#6b6b6b", "#8e6bbf"


# --------------------------------------------------------------------------- architecture
def _box(ax, x, y, w, h, text, fc, ec=None, fs=10.2, bold=False, ls="-", lw=0.9):
    """Box with its own outline; clip_on=False so strokes are never cut by the axes."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, facecolor=fc, edgecolor=ec or "#333333",
                                linestyle=ls, zorder=2, clip_on=False, mutation_aspect=0.55))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45, clip_on=False)


def _arrow(ax, p, q, style="-|>", color="#333333", ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, linewidth=lw,
                                 color=color, linestyle=ls, zorder=4, clip_on=False,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    """Vertical stage diagram. Drawn at the width of the journal text block so that LaTeX scales
    it by less than 10%; every box is on an explicit grid row and no two elements overlap."""
    fig, ax = plt.subplots(figsize=(5.4, 6.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    ax.set_position([0.005, 0.005, 0.99, 0.99])

    FS = 8.0
    LEFT, RIGHT, W = 0.045, 0.525, 0.43          # two side-by-side columns
    CL, CR = LEFT + W / 2, RIGHT + W / 2          # their centres
    FULL_X, FULL_W = 0.045, 0.91

    # row 1 ---------------------------------------------------------------- input
    _box(ax, FULL_X, 0.925, FULL_W, 0.060,
         "CAN bus traffic: frames $(t,\ \mathrm{ID},\ \mathrm{DLC},\ \mathrm{payload})$",
         "#eef3f8", fs=FS)
    # row 2 ---------------------------------------------------------------- framing + references
    _box(ax, LEFT, 0.820, W, 0.075, "Sliding frame window\n$F=64$, stride $S=16$", "#eef3f8", fs=FS)
    _box(ax, RIGHT, 0.820, W, 0.075, "Per-ID warm-up references\nfirst $W=30$ s, label-free",
         "#fdf3ec", ORANGE, fs=FS)
    _arrow(ax, (CL, 0.925), (CL, 0.897))
    _arrow(ax, (CR, 0.925), (CR, 0.897))

    # row 3 ---------------------------------------------------------------- representations
    _box(ax, LEFT, 0.630, W, 0.135,
         "Branch A: ID-relative\n$\\phi^{\\mathrm{rel}}$, 12 features/frame\n"
         "cadence ratio, bit/byte\nchange, range excursion,\nrate ratios", "#e8f1f8", BLUE, fs=FS)
    _box(ax, RIGHT, 0.630, W, 0.135,
         "Branch B: residualisation\n$\\phi^{\\mathrm{res}}$, 14 features/frame\n"
         "window statistics z-scored\nagainst that identifier's\nown baseline", "#fdeee7", ORANGE, fs=FS)
    _arrow(ax, (CL, 0.820), (CL, 0.767))
    _arrow(ax, (CR, 0.820), (CR, 0.767))
    _arrow(ax, (RIGHT, 0.8575), (LEFT + W, 0.8575), style="-|>", color=ORANGE, ls=":")

    # row 4 ---------------------------------------------------------------- detectors
    _box(ax, LEFT, 0.490, W, 0.105,
         "ConvGRU$_A$ (no ID embedding)\n98 k parameters\nscore $s^A_w\\in[0,1]$ per\n100 ms window",
         "#e8f1f8", BLUE, fs=FS)
    _box(ax, RIGHT, 0.490, W, 0.105,
         "ConvGRU$_B$ (no ID embedding)\n98 k parameters\nscore $s^B_w\\in[0,1]$ per\n100 ms window",
         "#fdeee7", ORANGE, fs=FS)
    _arrow(ax, (CL, 0.630), (CL, 0.597))
    _arrow(ax, (CR, 0.630), (CR, 0.597))

    # row 5 ---------------------------------------------------------------- alarm rules
    _box(ax, LEFT, 0.375, W, 0.075, "$k$-of-$n$ alarm rule\nthreshold $\\tau_A$", "#f3f6f9", BLUE, fs=FS)
    _box(ax, RIGHT, 0.375, W, 0.075, "$k$-of-$n$ alarm rule\nthreshold $\\tau_B$", "#f9f2ee", ORANGE, fs=FS)
    _arrow(ax, (CL, 0.490), (CL, 0.452))
    _arrow(ax, (CR, 0.490), (CR, 0.452))

    # row 6 ---------------------------------------------------------------- disjunction
    _box(ax, 0.22, 0.268, 0.56, 0.068, "OR: alarm when either branch fires",
         "#e9f4ea", GREEN, bold=True, fs=FS)
    _arrow(ax, (CL, 0.375), (0.38, 0.338), rad=-0.10)
    _arrow(ax, (CR, 0.375), (0.62, 0.338), rad=0.10)

    # row 7 ---------------------------------------------------------------- calibration
    _box(ax, LEFT, 0.150, W, 0.085,
         "Per-vehicle calibration\ntarget vehicle's own\nattack-free traffic (no labels)",
         "#f2eef8", PURPLE, fs=FS)
    _box(ax, RIGHT, 0.150, W, 0.085,
         "Threshold search (Alg. 2)\nsmallest $\\tau$: FA/h $\\leq B$,\nduty $\\leq\\delta$; each branch $B/2$",
         "#f2eef8", PURPLE, fs=FS)
    _arrow(ax, (LEFT + W, 0.1925), (RIGHT, 0.1925), color=PURPLE)
    _arrow(ax, (RIGHT + W, 0.235), (RIGHT + W - 0.02, 0.375), color=PURPLE, ls="--", rad=0.25)
    _arrow(ax, (RIGHT, 0.235), (LEFT + 0.02, 0.375), color=PURPLE, ls="--", rad=-0.25)

    # row 8 ---------------------------------------------------------------- evaluation harness
    _box(ax, FULL_X, 0.020, FULL_W, 0.105,
         "Evaluation harness (Section 5)\n"
         "capture-level cross-validation; an attack and its masquerade variant never split\n"
         "whole held-out normal drives; event = alarm inside an episode, latency from onset\n"
         "false alarm = rising edge on attack-free traffic; detection reported at budget $B$",
         "#f7f7f7", GREY, fs=FS - 0.6)
    _arrow(ax, (0.50, 0.268), (0.50, 0.128), color=GREY, ls="--")

    # stage labels down the left margin
    for y, name in ((0.955, "input"), (0.858, "framing"), (0.698, "representation"),
                    (0.543, "detection"), (0.413, "alarm rule"), (0.302, "decision"),
                    (0.193, "calibration")):
        ax.text(0.012, y, name, rotation=90, ha="center", va="center", fontsize=FS - 1.4,
                style="italic", color="#666666")

    fig.savefig(FIGS / "architecture.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("architecture.pdf")


# --------------------------------------------------------------------------- comparison
def fig_comparison():
    rows = N["road_models"]["rows"]
    labels = [r["model"].replace(" (unsupervised)", "\n(unsup.)").replace(" (raw frames)", "\n(raw frames)")
              .replace(" + payload novelty", "\n+ payload novelty").replace(" + payload", "\n+ payload") for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = np.arange(len(rows)); w = 0.38
    for i, (key, sd, lab, col) in enumerate([("at5", "at5_sd", "$\\leq$5 FA/h", BLUE),
                                             ("at30", "at30_sd", "$\\leq$30 FA/h", GREEN)]):
        ax.bar(x + (i - 0.5) * w, [r[key] for r in rows], w, yerr=[r[sd] for r in rows],
               capsize=2, label=lab, color=col, error_kw={"lw": 0.7})
    for xi, r in zip(x, rows):
        ax.text(xi - 0.5 * w, r["at5"] + r["at5_sd"] + 0.6, f"{r['at5']:.1f}", ha="center", fontsize=9.3)
        ax.text(xi + 0.5 * w, r["at30"] + r["at30_sd"] + 0.6, f"{r['at30']:.1f}", ha="center", fontsize=9.3)
    ceiling = N["road_models"]["tests"]["ceiling"]
    ax.axhline(ceiling, color=GREY, ls="--", lw=0.8)
    # label the ceiling over the empty region above the two unsupervised detectors
    ax.text(0.4, ceiling + 0.9, f"data ceiling ({ceiling}/{N['road_models']['n_events']})",
            ha="left", va="bottom", fontsize=9.3, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(f"events detected (of {N['road_models']['n_events']})")
    ax.set_ylim(0, 33); ax.legend(fontsize=10.1, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    fig.savefig(FIGS / "model_comparison.pdf")
    plt.close(fig)
    print("model_comparison.pdf")


# --------------------------------------------------------------------------- ablations
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    fig.subplots_adjust(wspace=0.42)
    # (a) architecture/feature ablation on ROAD at 30 FA/h
    rows = N["road_models"]["rows"]
    names = ["rate\nexceed.", "payload\nnovelty", "window\nGBDT", "graph\n+ payload", "Conv\nGRU", "ConvGRU\n+ payload"]
    vals = [r["at30"] for r in rows]; errs = [r["at30_sd"] for r in rows]
    cols = [GREY, GREY, PURPLE, ORANGE, BLUE, BLUE]
    axes[0].bar(np.arange(len(rows)), vals, 0.6, yerr=errs, capsize=2, color=cols, error_kw={"lw": 0.7})
    axes[0].set_xticks(np.arange(len(rows))); axes[0].set_xticklabels(names, fontsize=8.5, rotation=0)
    axes[0].set_ylabel("events detected (of 29)\nat $\\leq$30 FA/h"); axes[0].set_ylim(0, 31)
    axes[0].set_title("(a) ROAD: detector ablation", fontsize=12.4)

    # (b) representation ablation, cross-vehicle
    cv = N["cross_vehicle"]["totals"]; ev = cv["events"]
    labs = ["ID\nembedding", "$\\phi^{res}$\nresidual.", "$\\phi^{rel}$\nrelative",
            "concat\nof both", "OR of\nbranches"]
    pair = N["pair"]
    union_pct = 100 * pair["rows"][2]["unknown_at5"] / pair["unknown_events"]
    vals2 = [100 * cv["idbased"] / ev, 100 * cv["residual"] / ev, 100 * cv["relative"] / ev,
             100 * cv["both_concat"] / ev, union_pct]
    cols2 = [GREY, ORANGE, BLUE, PURPLE, GREEN]
    axes[1].bar(np.arange(5), vals2, 0.6, color=cols2)
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9.3)
    axes[1].set_xticks(np.arange(5)); axes[1].set_xticklabels(labs, fontsize=8.5, rotation=0)
    axes[1].set_ylabel("unknown-vehicle events\ndetected (%) at $\\leq$5 FA/h"); axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) cross-vehicle: representation ablation", fontsize=12.4)
    fig.savefig(FIGS / "ablation.pdf")
    plt.close(fig)
    print("ablation.pdf")


# --------------------------------------------------------------------------- latency & tiers
def fig_latency_tiers():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    fig.subplots_adjust(wspace=0.38)
    lat = N["latency"]["rows"]
    ax = axes[0]
    xs = [r["latency_ms"] for r in lat]; ys = [r["events"] for r in lat]; es = [r["sd"] for r in lat]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=BLUE, capsize=2, lw=1.1, ms=4)
    for r in lat:
        ax.annotate(r["rule"], (r["latency_ms"], r["events"]), textcoords="offset points",
                    xytext=(6, -9), fontsize=10.1)
    ax.set_xlabel("median detection latency (ms)")
    ax.set_ylabel("events detected (of 29)\nat $\\leq$5 FA/h")
    ax.set_xlim(-20, 270); ax.set_ylim(0, 31)
    ax.set_title("(a) alarm rule sets the latency floor", fontsize=12.4)

    ax = axes[1]
    tiers = N["tiers"]["rows"]
    x = np.arange(len(tiers)); w = 0.38
    ax.bar(x - w / 2, [100 * r["value_detected"] / r["value_total"] for r in tiers], w,
           label="value attacks", color=BLUE)
    ax.bar(x + w / 2, [100 * r["context_detected"] / r["context_total"] for r in tiers], w,
           label="contextual attacks", color=ORANGE)
    for i, r in enumerate(tiers):
        ax.text(i - w / 2, 100 * r["value_detected"] / r["value_total"] + 2,
                f"{r['value_detected']}/{r['value_total']}", ha="center", fontsize=8.6, color=BLUE)
        ax.text(i + w / 2, 100 * r["context_detected"] / r["context_total"] + 2,
                f"{r['context_detected']}/{r['context_total']}", ha="center", fontsize=8.6, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels([r["detector"].replace(" ", "\n") for r in tiers], fontsize=9.3)
    ax.set_ylabel("events detected (%)"); ax.set_ylim(0, 145)
    # legend above the bars so it cannot cover them
    ax.legend(fontsize=9.3, loc="upper center", ncol=2, framealpha=0.95,
              handlelength=1.3, columnspacing=1.0)
    ax.set_title("(b) contextual attacks need cross-ID evidence", fontsize=12.4)
    fig.savefig(FIGS / "latency_tiers.pdf")
    plt.close(fig)
    print("latency_tiers.pdf")


# --------------------------------------------------------------------------- adaptation
def fig_adaptation():
    rows = N["adaptation"]["rows"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r["at5"] for r in rows], w, label="$\\leq$5 FA/h", color=BLUE)
    ax.bar(x + w / 2, [r["at30"] for r in rows], w, label="$\\leq$30 FA/h", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['minutes']:g} min" for r in rows], fontsize=10.8)
    ax.set_xlabel("unlabelled target-vehicle traffic used for adaptation")
    ax.set_ylabel(f"events detected (of {rows[0]['events']})")
    ax.set_ylim(0, 8); ax.legend(fontsize=10.1)
    ax.text(1.5, 5.6, "no trend: unlabelled adaptation\ndoes not restore detection",
            ha="center", fontsize=10.1, color=GREY)
    fig.savefig(FIGS / "adaptation.pdf")
    plt.close(fig)
    print("adaptation.pdf")


def fig_threshold_transfer():
    """Calibrated budget vs measured false-alarm rate under the two ambient-split protocols."""
    tt = N["threshold_transfer"]
    b = tt["budgets"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(b, tt["time_sliced_measured"], "o-", color=ORANGE, lw=1.2, ms=4,
            label="calibrated on time slices\nof the test drives")
    ax.plot(b, tt["whole_drive_measured"], "s-", color=BLUE, lw=1.2, ms=4,
            label="calibrated on whole\nheld-out drives")
    lim = [4, 2000]
    ax.plot([5, 100], [5, 100], ":", color=GREY, lw=1.0, label="budget honoured")
    for x, y in zip(b, tt["time_sliced_measured"]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9.3)
    for x, y in zip(b, tt["whole_drive_measured"]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=9.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*lim)
    ax.set_xlabel("false-alarm budget requested (per hour)")
    ax.set_ylabel("false alarms measured\non unseen drives (per hour)")
    ax.legend(fontsize=9.0, loc="upper left")
    fig.savefig(FIGS / "threshold_transfer.pdf")
    plt.close(fig)
    print("threshold_transfer.pdf")


if __name__ == "__main__":
    fig_architecture()
    fig_threshold_transfer()
    fig_comparison()
    fig_ablation()
    fig_latency_tiers()
    fig_adaptation()
    print(f"-> {FIGS}")
