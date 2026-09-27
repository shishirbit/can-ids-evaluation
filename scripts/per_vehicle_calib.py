"""How much attack-free traffic from a new vehicle is needed to restore a usable operating point?

Setup (can-train-and-test-v1.5 set_01): train on the Chevrolet Impala's attack traffic, then adapt
to the Chevrolet Silverado using only N minutes of that vehicle's *attack-free* traffic (no attack
labels from the target vehicle - what a workshop or fleet operator could realistically record).

For each N: the ID vocabulary is extended with IDs seen in the adaptation traffic, the target clean
frames are added to training as negatives, the alarm threshold is calibrated on a held-out 20% of
the adaptation traffic, and detection is evaluated on target attack files *not* used for adaptation.
Files are split into two groups (instances -3 and -4) and the experiment is run both ways, so all
8 target attack events are evaluated without adaptation/evaluation ever sharing a file.

    python scripts/per_vehicle_calib.py [--payload] [--minutes 0 1 5 15]
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
from galaxy_its.data.can_io import CanLog, read_cantt_csv
from galaxy_its.data.graphs import IdVocab, fill_short_gaps
from galaxy_its.data.payload import ByteProfile
from galaxy_its.eval.operating import event_detection, false_alarms_per_hour
from galaxy_its.models.frame_models import ConvGRU

DT, RULE, FRAMES, STRIDE = 0.1, (3, 5), 64, 16


def clean_prefix(log: CanLog) -> CanLog:
    """The attack-free part of a capture, i.e. everything before the first injected frame."""
    n = int(np.argmax(log.label)) if log.label.any() else len(log)
    return log.slice(0, n)


def windows(log, vocab, profile, with_labels=True):
    node = vocab.map(log.can_id).astype(np.int64)
    data = log.data.astype(np.float32) / 255.0
    if profile is not None:
        data = np.concatenate([data, profile.surprisal(node, log.data)], -1)
    starts = np.arange(0, max(len(log) - FRAMES, 0), STRIDE)
    cs = np.concatenate([[0], np.cumsum(log.label.astype(np.int64))])
    lab = (cs[starts + FRAMES] - cs[starts] > 0).astype(np.int8) if with_labels and len(starts) else np.zeros(len(starts), np.int8)
    rel = log.ts - log.ts[0]
    T = int(rel[-1] / DT) + 1 if len(log) else 1
    win = np.zeros(T, np.int8)
    if len(log):
        np.maximum.at(win, (rel / DT).astype(np.int64), log.label)
    return {"node": node, "data": data, "starts": starts, "lab": lab,
            "w_end": (rel[starts + FRAMES - 1] / DT).astype(np.int64) if len(starts) else np.zeros(0, np.int64),
            "win_inj": fill_short_gaps(win, int(1.0 / DT)), "T": T}


def train_model(seqdict, items, n_ids, n_extra, args, rng):
    model = ConvGRU(n_ids, n_extra=n_extra).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device="cuda"))
    for ep in range(args.epochs):
        model.train(); losses = []
        for step, (ids, dat, lab, _) in enumerate(batches(seqdict, items, FRAMES, args.batch, "cuda", True, rng)):
            if step >= args.max_steps:
                break
            loss = lossf(model(ids, dat), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
    return model


@torch.no_grad()
def score_files(model, seqdict, names, args):
    model.eval(); out = {}
    for name in names:
        d = seqdict[name]
        items = np.array([(name, i) for i in range(len(d["starts"]))], dtype=object)
        win = np.full(d["T"], np.nan)
        for ids, dat, _, sel in batches(seqdict, items, FRAMES, args.batch, "cuda"):
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
    ap.add_argument("--minutes", type=float, nargs="+", default=[0, 1, 5, 15])
    ap.add_argument("--payload", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/per_vehicle_calib")
    args = ap.parse_args()
    root = ROOT / args.root
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    src_logs = [read_cantt_csv(f) for f in sorted((root / "train_02_with_attacks").glob("*.csv"))]
    tgt_files = sorted((root / "test_02_unknown_vehicle_known_attack").glob("*.csv"))
    tgt_logs = {f.stem: read_cantt_csv(f) for f in tgt_files}
    groups = {"A": [n for n in tgt_logs if n.endswith("-3")], "B": [n for n in tgt_logs if n.endswith("-4")]}
    print(f"source files={len(src_logs)}, target files={list(tgt_logs)}", flush=True)

    results = {}
    for minutes in args.minutes:
        for adapt_g, eval_g in (("A", "B"), ("B", "A")):
            # adaptation pool: clean prefixes of the adaptation group's files, first `minutes`
            adapt = []
            budget_s = minutes * 60
            for n in groups[adapt_g]:
                if budget_s <= 0:
                    break
                c = clean_prefix(tgt_logs[n])
                take = min(budget_s, float(c.ts[-1] - c.ts[0]) if len(c) else 0)
                if take <= 1:
                    continue
                k = int(np.searchsorted(c.ts - c.ts[0], take))
                adapt.append(c.slice(0, k)); budget_s -= take
            vocab_logs = src_logs + adapt
            vocab = IdVocab.fit(vocab_logs)
            profile = ByteProfile.fit(vocab_logs, vocab) if args.payload else None

            seqdict, items = {}, []
            for i, l in enumerate(src_logs):
                seqdict[f"src{i}"] = windows(l, vocab, profile)
                items += [(f"src{i}", k) for k in range(len(seqdict[f"src{i}"]["starts"]))]
            cal_seqs = []
            for i, l in enumerate(adapt):                    # 80% train as negatives, 20% calibrate
                w = windows(l, vocab, profile)
                cut = int(0.8 * len(w["starts"]))
                seqdict[f"tgt{i}"] = w
                items += [(f"tgt{i}", k) for k in range(cut)] * 3      # oversample target normal
                if cut < len(w["starts"]):
                    sc = None  # calibration scored after training
                    cal_seqs.append((f"tgt{i}", cut))
            items = np.array(items, dtype=object)
            model = train_model(seqdict, items, vocab.n_nodes, 8 if args.payload else 0, args, rng)

            # calibration scores from the held-out tail of the adaptation traffic
            cal = []
            if cal_seqs:
                for name, cut in cal_seqs:
                    d = seqdict[name]
                    sub = np.array([(name, k) for k in range(cut, len(d["starts"]))], dtype=object)
                    win = np.full(d["T"], np.nan)
                    with torch.no_grad():
                        model.eval()
                        for ids, dat, _, sel in batches(seqdict, sub, FRAMES, args.batch, "cuda"):
                            s = torch.sigmoid(model(ids, dat)).cpu().numpy()
                            for (_, k), v in zip(sel, s):
                                w2 = d["w_end"][k]
                                win[w2] = v if np.isnan(win[w2]) else max(win[w2], v)
                    ok = ~np.isnan(win)
                    cal.append((win[ok], np.zeros(ok.sum(), np.int8)))

            for n in groups[eval_g]:
                seqdict[n] = windows(tgt_logs[n], vocab, profile)
            ev = score_files(model, seqdict, groups[eval_g], args)
            seqs, names = list(ev.values()), list(ev)
            cal_source = cal if cal else seqs                 # minutes=0 -> nothing from target
            for budget in (5, 30):
                t = thr_for(cal_source, budget)
                rows = event_detection(seqs, names, t, *RULE, DT)
                fa, hrs, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
                lat = [r["latency_ms"] for r in rows if r["detected"]]
                key = f"{minutes}min|{adapt_g}->{eval_g}|{budget}"
                # an operating point only counts if the budget actually holds on test data
                valid = fa <= 3 * budget and duty <= 0.02
                results[key] = {"detected": sum(r["detected"] for r in rows), "events": len(rows),
                                "fa_per_h": fa, "duty": duty, "threshold": t, "valid": valid,
                                "median_latency_ms": float(np.median(lat)) if lat else None,
                                "adapt_minutes_used": minutes, "n_ids": vocab.n_nodes}
                print(f"{key:24s} det={results[key]['detected']}/{results[key]['events']} "
                      f"FA/h={fa:6.1f} duty={duty*100:5.1f}% lat={results[key]['median_latency_ms']} "
                      f"{'' if valid else '[BUDGET VIOLATED]'}", flush=True)

    # pool the two directions
    print(f"\n{'adapt minutes':14s} {'@5/h events':>12} {'duty':>7} {'@30/h events':>13} {'duty':>7}")
    summary = {}
    for minutes in args.minutes:
        row = {}
        for budget in (5, 30):
            ks = [k for k in results if k.startswith(f"{minutes}min") and k.endswith(f"|{budget}")]
            det = sum(results[k]["detected"] for k in ks); ev = sum(results[k]["events"] for k in ks)
            duty = float(np.mean([results[k]["duty"] for k in ks]))
            det_valid = sum(results[k]["detected"] for k in ks if results[k]["valid"])
            ev_valid = sum(results[k]["events"] for k in ks if results[k]["valid"])
            row[budget] = {"detected": det, "events": ev, "duty": duty,
                           "detected_valid": det_valid, "events_valid": ev_valid}
        summary[minutes] = row
        print(f"{minutes:14g} {row[5]['detected']:5d}/{row[5]['events']:<6d} {row[5]['duty']*100:6.1f}% "
              f"{row[30]['detected']:6d}/{row[30]['events']:<6d} {row[30]['duty']*100:6.1f}%"
              f"   valid-only: {row[5]['detected_valid']}/{row[5]['events_valid']} @5/h, "
              f"{row[30]['detected_valid']}/{row[30]['events_valid']} @30/h")
    tag = "payload" if args.payload else "plain"
    json.dump({"args": vars(args), "results": results, "summary": summary},
              open(out / f"{tag}_s{args.seed}.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {out / f'{tag}_s{args.seed}.json'}")


if __name__ == "__main__":
    main()
