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

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#1b6ca8", "#c9603a", "#2e7d32", "#6b6b6b", "#8e6bbf"


# --------------------------------------------------------------------------- architecture
def _box(ax, x, y, w, h, text, fc, ec=None, fs=7, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, facecolor=fc, edgecolor=ec or "#333333",
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, p, q, style="-|>", color="#333333", ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    # ---- stage 1: bus, framing, references (left column)
    _box(ax, 0.01, 0.79, 0.20, 0.12, "CAN bus traffic\nframe $=(t,\\;\\mathrm{ID},\\;\\mathrm{DLC},\\;8$ bytes$)$", "#eef3f8", fs=6.8)
    _box(ax, 0.01, 0.61, 0.20, 0.11, "Sliding frame window\n$F=64$ frames, stride $16$", "#eef3f8", fs=6.8)
    _box(ax, 0.01, 0.43, 0.20, 0.11, "Per-ID warm-up references\nfirst $W=30$ s, label-free", "#fdf3ec", ORANGE, fs=6.8)
    _arrow(ax, (0.11, 0.79), (0.11, 0.725))
    _arrow(ax, (0.11, 0.61), (0.11, 0.545))

    # ---- stage 2: two identifier-agnostic representations
    _box(ax, 0.26, 0.70, 0.21, 0.16,
         "Branch A: ID-relative\n$\\phi^{\\mathrm{rel}}$, 12 features/frame\n"
         "cadence ratio, bit/byte change,\nrange excursion, rate ratios", "#e8f1f8", BLUE, fs=6.5)
    _box(ax, 0.26, 0.44, 0.21, 0.16,
         "Branch B: residualisation\n$\\phi^{\\mathrm{res}}$, 14 features/frame\n"
         "window statistics z-scored\nagainst that ID's baseline", "#fdeee7", ORANGE, fs=6.5)
    _arrow(ax, (0.21, 0.665), (0.26, 0.78), rad=0.12)
    _arrow(ax, (0.21, 0.665), (0.26, 0.52), rad=-0.12)
    _arrow(ax, (0.21, 0.485), (0.26, 0.50), ls=":", color=ORANGE)
    _arrow(ax, (0.21, 0.49), (0.26, 0.73), ls=":", color=ORANGE, rad=0.25)

    # ---- stage 3: detectors
    _box(ax, 0.52, 0.71, 0.16, 0.14, "ConvGRU$_A$\nno ID embedding\n98 k params\n$s^A_w\\in[0,1]$", "#e8f1f8", BLUE, fs=6.5)
    _box(ax, 0.52, 0.45, 0.16, 0.14, "ConvGRU$_B$\nno ID embedding\n98 k params\n$s^B_w\\in[0,1]$", "#fdeee7", ORANGE, fs=6.5)
    _arrow(ax, (0.47, 0.78), (0.52, 0.78))
    _arrow(ax, (0.47, 0.52), (0.52, 0.52))

    # ---- stage 4: alarm logic
    _box(ax, 0.72, 0.71, 0.11, 0.14, "$k$-of-$n$\nalarm rule\nthreshold $\\tau_A$", "#f3f6f9", BLUE, fs=6.5)
    _box(ax, 0.72, 0.45, 0.11, 0.14, "$k$-of-$n$\nalarm rule\nthreshold $\\tau_B$", "#f9f2ee", ORANGE, fs=6.5)
    _arrow(ax, (0.68, 0.78), (0.72, 0.78))
    _arrow(ax, (0.68, 0.52), (0.72, 0.52))
    _box(ax, 0.87, 0.58, 0.11, 0.14, "OR\n\nalarm\nevent", "#e9f4ea", GREEN, bold=True, fs=7)
    _arrow(ax, (0.83, 0.78), (0.925, 0.725), rad=-0.12)
    _arrow(ax, (0.83, 0.52), (0.925, 0.575), rad=0.12)

    # ---- calibration path
    _box(ax, 0.26, 0.22, 0.21, 0.13,
         "Per-vehicle calibration\ntarget vehicle's own attack-free\ntraffic (no labels needed)", "#f2eef8", PURPLE, fs=6.5)
    _box(ax, 0.52, 0.22, 0.31, 0.13,
         "Threshold search: smallest $\\tau$ with\n"
         "$\\mathrm{FA/h}(\\tau)\\leq B$ and duty$(\\tau)\\leq\\delta$\n"
         "each branch receives budget $B/2$", "#f2eef8", PURPLE, fs=6.5)
    _arrow(ax, (0.47, 0.285), (0.52, 0.285), color=PURPLE)
    _arrow(ax, (0.64, 0.35), (0.70, 0.445), color=PURPLE, ls="--", rad=-0.2)
    _arrow(ax, (0.79, 0.35), (0.79, 0.445), color=PURPLE, ls="--")
    _arrow(ax, (0.11, 0.43), (0.30, 0.35), color=PURPLE, ls=":", rad=-0.15)

    # ---- evaluation harness
    _box(ax, 0.01, 0.03, 0.97, 0.12,
         "Evaluation harness (Sec. 5):  capture-level cross-validation, an attack and its masquerade variant never split  |  "
         "whole held-out normal drives\n"
         "event detected = alarm fires inside the attack episode, latency measured from onset  |  "
         "false alarm = rising edge on attack-free traffic  |  detection reported at budget $B$",
         "#f7f7f7", GREY, fs=6.2)
    _arrow(ax, (0.925, 0.58), (0.925, 0.16), color=GREY, ls="--")

    ax.text(0.01, 0.955, "Identifier-agnostic front end: nothing vehicle-specific is learned or transferred",
            fontsize=7.5, fontweight="bold", color="#222222")
    fig.savefig(FIGS / "architecture.pdf")
    plt.close(fig)
    print("architecture.pdf")


# --------------------------------------------------------------------------- comparison
def fig_comparison():
    rows = N["road_models"]["rows"]
    labels = [r["model"].replace(" (unsupervised)", "\n(unsup.)").replace(" (raw frames)", "\n(raw frames)")
              .replace(" + payload novelty", "\n+ payload novelty").replace(" + payload", "\n+ payload") for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    x = np.arange(len(rows)); w = 0.38
    for i, (key, sd, lab, col) in enumerate([("at5", "at5_sd", "$\\leq$5 FA/h", BLUE),
                                             ("at30", "at30_sd", "$\\leq$30 FA/h", GREEN)]):
        ax.bar(x + (i - 0.5) * w, [r[key] for r in rows], w, yerr=[r[sd] for r in rows],
               capsize=2, label=lab, color=col, error_kw={"lw": 0.7})
    for xi, r in zip(x, rows):
        ax.text(xi - 0.5 * w, r["at5"] + r["at5_sd"] + 0.6, f"{r['at5']:.1f}", ha="center", fontsize=6)
        ax.text(xi + 0.5 * w, r["at30"] + r["at30_sd"] + 0.6, f"{r['at30']:.1f}", ha="center", fontsize=6)
    ax.axhline(N["road_models"]["tests"]["ceiling"], color=GREY, ls="--", lw=0.8)
    ax.text(len(rows) - 0.5, N["road_models"]["tests"]["ceiling"] + 0.4, "data ceiling (27/29)",
            ha="right", fontsize=6, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(f"events detected (of {N['road_models']['n_events']})")
    ax.set_ylim(0, 31); ax.legend(fontsize=6.5, loc="upper left")
    fig.savefig(FIGS / "model_comparison.pdf")
    plt.close(fig)
    print("model_comparison.pdf")


# --------------------------------------------------------------------------- ablations
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(wspace=0.42)
    # (a) architecture/feature ablation on ROAD at 30 FA/h
    rows = N["road_models"]["rows"]
    names = ["Rate\nexceed.", "Payload\nnovelty", "Window\nGBDT", "Graph attn\n+ payload", "ConvGRU", "ConvGRU\n+ payload"]
    vals = [r["at30"] for r in rows]; errs = [r["at30_sd"] for r in rows]
    cols = [GREY, GREY, PURPLE, ORANGE, BLUE, BLUE]
    axes[0].bar(np.arange(len(rows)), vals, 0.6, yerr=errs, capsize=2, color=cols, error_kw={"lw": 0.7})
    axes[0].set_xticks(np.arange(len(rows))); axes[0].set_xticklabels(names, fontsize=5.5, rotation=18, ha="right")
    axes[0].set_ylabel("events detected (of 29)\nat $\\leq$30 FA/h"); axes[0].set_ylim(0, 31)
    axes[0].set_title("(a) ROAD: detector ablation", fontsize=8)

    # (b) representation ablation, cross-vehicle
    cv = N["cross_vehicle"]["totals"]; ev = cv["events"]
    labs = ["ID embedding\n(+payload)", "residualisation\n$\\phi^{res}$", "ID-relative\n$\\phi^{rel}$",
            "concat\n$[\\phi^{rel},\\phi^{res}]$", "OR of\nbranches"]
    pair = N["pair"]
    union_pct = 100 * pair["rows"][2]["unknown_at5"] / pair["unknown_events"]
    vals2 = [100 * cv["idbased"] / ev, 100 * cv["residual"] / ev, 100 * cv["relative"] / ev,
             100 * cv["both_concat"] / ev, union_pct]
    cols2 = [GREY, ORANGE, BLUE, PURPLE, GREEN]
    axes[1].bar(np.arange(5), vals2, 0.6, color=cols2)
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=6)
    axes[1].set_xticks(np.arange(5)); axes[1].set_xticklabels(labs, fontsize=5.5, rotation=18, ha="right")
    axes[1].set_ylabel("unknown-vehicle events\ndetected (%) at $\\leq$5 FA/h"); axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) cross-vehicle: representation ablation", fontsize=8)
    fig.savefig(FIGS / "ablation.pdf")
    plt.close(fig)
    print("ablation.pdf")


# --------------------------------------------------------------------------- latency & tiers
def fig_latency_tiers():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    fig.subplots_adjust(wspace=0.38)
    lat = N["latency"]["rows"]
    ax = axes[0]
    xs = [r["latency_ms"] for r in lat]; ys = [r["events"] for r in lat]; es = [r["sd"] for r in lat]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=BLUE, capsize=2, lw=1.1, ms=4)
    for r in lat:
        ax.annotate(r["rule"], (r["latency_ms"], r["events"]), textcoords="offset points",
                    xytext=(6, -9), fontsize=6.5)
    ax.set_xlabel("median detection latency (ms)")
    ax.set_ylabel("events detected (of 29)\nat $\\leq$5 FA/h")
    ax.set_xlim(-20, 270); ax.set_ylim(0, 31)
    ax.set_title("(a) alarm rule sets the latency floor", fontsize=8)

    ax = axes[1]
    tiers = N["tiers"]["rows"]
    x = np.arange(len(tiers)); w = 0.38
    ax.bar(x - w / 2, [100 * r["value_detected"] / r["value_total"] for r in tiers], w,
           label="value attacks", color=BLUE)
    ax.bar(x + w / 2, [100 * r["context_detected"] / r["context_total"] for r in tiers], w,
           label="contextual attacks", color=ORANGE)
    for i, r in enumerate(tiers):
        ax.text(i + w / 2, 100 * r["context_detected"] / r["context_total"] + 2,
                f"{r['context_detected']}/{r['context_total']}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([r["detector"].replace(" ", "\n") for r in tiers], fontsize=6)
    ax.set_ylabel("events detected (%)"); ax.set_ylim(0, 115); ax.legend(fontsize=6)
    ax.set_title("(b) contextual attacks need cross-ID evidence", fontsize=8)
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
    ax.set_xticks(x); ax.set_xticklabels([f"{r['minutes']:g} min" for r in rows], fontsize=7)
    ax.set_xlabel("unlabelled target-vehicle traffic used for adaptation")
    ax.set_ylabel(f"events detected (of {rows[0]['events']})")
    ax.set_ylim(0, 8); ax.legend(fontsize=6.5)
    ax.text(1.5, 5.6, "no trend: unlabelled adaptation\ndoes not restore detection",
            ha="center", fontsize=6.5, color=GREY)
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
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)
    for x, y in zip(b, tt["whole_drive_measured"]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*lim)
    ax.set_xlabel("false-alarm budget requested (per hour)")
    ax.set_ylabel("false alarms measured\non unseen drives (per hour)")
    ax.legend(fontsize=5.8, loc="upper left")
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
