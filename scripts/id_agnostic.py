"""ID-agnostic cross-vehicle detection on can-train-and-test-v1.5 set_01.

Same task as scripts/cross_vehicle.py (train on Chevrolet Impala, test on Chevrolet Silverado) but
frames are described only by ID-relative features (galaxy_its.data.relative) and the model has no
ID embedding, so nothing vehicle-specific is learned or transferred. Per-ID reference statistics
come from a warm-up prefix of each capture: unsupervised and label-free.

    python scripts/id_agnostic.py [--warmup 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from baseline_frames import batches
from galaxy_its.data.can_io import read_cantt_csv
from galaxy_its.data.graphs import fill_short_gaps
from galaxy_its.data.relative import FEATURES as REL_FEATURES, relative_features
from galaxy_its.data.residual import FEATURES as RES_FEATURES, residual_features
from galaxy_its.eval.operating import event_detection, false_alarms_per_hour
from galaxy_its.models.frame_models import ConvGRU

DT, RULE, FRAMES, STRIDE = 0.1, (3, 5), 64, 16
SUBSETS = ["train_01_attack_free", "train_02_with_attacks", "test_01_known_vehicle_known_attack",
           "test_02_unknown_vehicle_known_attack", "test_06_masquerade"]


def build(root: Path, warmup: float, featfn):
    seqdict = {}
    for sub in SUBSETS:
        for f in sorted((root / sub).glob("*.csv")):
            log = read_cantt_csv(f)
            data = featfn(log, warmup_s=warmup)
            starts = np.arange(0, max(len(log) - FRAMES, 0), STRIDE)
            cs = np.concatenate([[0], np.cumsum(log.label.astype(np.int64))])
            rel = log.ts - log.ts[0]
            T = int(rel[-1] / DT) + 1
            win = np.zeros(T, np.int8)
            np.maximum.at(win, (rel / DT).astype(np.int64), log.label)
            # frame windows overlapping the warm-up prefix are excluded everywhere: their
            # reference statistics are estimated from the same data
            keep = rel[starts] >= warmup
            starts = starts[keep]
            seqdict[f"{sub}/{f.stem}"] = {
                "node": np.zeros(len(log), np.int64), "data": data, "starts": starts,
                "lab": (cs[starts + FRAMES] - cs[starts] > 0).astype(np.int8),
                "w_end": (rel[starts + FRAMES - 1] / DT).astype(np.int64),
                "win_inj": fill_short_gaps(win, int(1.0 / DT)), "T": T, "sub": sub}
    return seqdict


@torch.no_grad()
def score(model, seqdict, names, batch):
    model.eval(); out = {}
    for name in names:
        d = seqdict[name]
        if not len(d["starts"]):
            continue
        items = np.array([(name, i) for i in range(len(d["starts"]))], dtype=object)
        win = np.full(d["T"], np.nan)
        for ids, dat, _, sel in batches(seqdict, items, FRAMES, batch, "cuda"):
            s = torch.sigmoid(model(ids, dat)).cpu().numpy()
            for (_, k), v in zip(sel, s):
                w = d["w_end"][k]
                win[w] = v if np.isnan(win[w]) else max(win[w], v)
        ok = ~np.isnan(win)
        out[name] = (win[ok], d["win_inj"][ok])
    return out


def thr_for(seqs, budget):
    cand = np.unique(np.concatenate([s for s, _ in seqs]))
    cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), 200)).astype(int)]
    for t in cand:
        fa, _, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
        if fa <= budget and duty <= 0.02:
            return float(t)
    return float(cand[-1]) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/cantt/set_01")
    ap.add_argument("--warmup", type=float, default=30.0)
    ap.add_argument("--features", choices=["relative", "residual", "both"], default="relative",
                    help="relative = ours (per-frame ratios); residual = per-ID behavioural residualisation baseline")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/id_agnostic")
    args = ap.parse_args()
    root = ROOT / args.root
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    setname = Path(args.root).name
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.features == "relative":
        featfn, FEATURES = relative_features, REL_FEATURES
    elif args.features == "residual":
        featfn, FEATURES = residual_features, RES_FEATURES
    else:                                   # both: concatenate the two ID-agnostic feature sets
        import numpy as _np

        def featfn(log, warmup_s):
            return _np.concatenate([relative_features(log, warmup_s=warmup_s),
                                    residual_features(log, warmup_s=warmup_s)], -1)
        FEATURES = REL_FEATURES + RES_FEATURES
    seqdict = build(root, args.warmup, featfn)
    train_items = np.array([(n, i) for n, d in seqdict.items() if d["sub"] == "train_02_with_attacks"
                            for i in range(len(d["starts"]))], dtype=object)
    print(f"{len(seqdict)} files, {len(train_items)} training windows, {len(FEATURES)} features/frame", flush=True)

    model = ConvGRU(0, n_channels=len(FEATURES)).cuda()      # no ID embedding
    print(f"params={sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device="cuda"))
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for step, (ids, dat, lab, _) in enumerate(batches(seqdict, train_items, FRAMES, args.batch, "cuda", True, rng)):
            if step >= args.max_steps:
                break
            loss = lossf(model(ids, dat), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"ep {ep} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s", flush=True)

    by_sub = {s: score(model, seqdict, [n for n, d in seqdict.items() if d["sub"] == s], args.batch)
              for s in SUBSETS}
    known_clean = list(by_sub["train_01_attack_free"].values())
    results = {}
    print(f"\n{'test set':38s} {'calibration':14s} {'events':>8} {'FA/h':>7} {'duty':>6} {'lat ms':>7}")
    for sub in ("test_01_known_vehicle_known_attack", "test_02_unknown_vehicle_known_attack", "test_06_masquerade"):
        seqs, names = list(by_sub[sub].values()), list(by_sub[sub])
        for cal, cal_seqs in (("known-clean", known_clean), ("target-own", seqs)):
            for budget in (5, 30):
                t = thr_for(cal_seqs, budget)
                rows = event_detection(seqs, names, t, *RULE, DT)
                fa, hrs, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
                lat = [r["latency_ms"] for r in rows if r["detected"]]
                valid = fa <= 3 * budget and duty <= 0.02
                key = f"{sub}|{cal}|{budget}"
                results[key] = {"detected": sum(r["detected"] for r in rows), "events": len(rows),
                                "fa_per_h": fa, "duty": duty, "valid": valid, "threshold": t,
                                "median_latency_ms": float(np.median(lat)) if lat else None}
                r = results[key]
                print(f"{sub:38s} {cal + '@' + str(budget):14s} {r['detected']:3d}/{r['events']:<4d} "
                      f"{fa:7.1f} {duty*100:5.1f}% {r['median_latency_ms'] if lat else float('nan'):7.0f}"
                      f" {'' if valid else '[BUDGET VIOLATED]'}")
    json.dump({"args": vars(args), "results": results}, open(out / f"{setname}_{args.features}_s{args.seed}.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {out / f's{args.seed}.json'}")


if __name__ == "__main__":
    main()
