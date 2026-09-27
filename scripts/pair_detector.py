"""Paired detector: ID-agnostic branch (transfers across vehicles) OR ID-based branch (per-vehicle).

Motivation (README §ID-agnostic): ID-relative features transfer across vehicles but miss masquerade
attacks; ID-based features catch masquerade on the vehicle they were trained on but transfer to
nothing. A deployable design runs both and raises an alarm if either fires.

Both branches are trained on the same source vehicle's attack traffic and evaluated on the *known*
vehicle's test sets (test_01 known attacks + test_06 masquerade), where both are valid. Each branch
gets half the total false-alarm budget, so the union is compared fairly against each branch alone
using its full budget.

    python scripts/pair_detector.py --root data/raw/cantt/set_01
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
from galaxy_its.data.graphs import IdVocab, fill_short_gaps
from galaxy_its.data.payload import ByteProfile
from galaxy_its.data.relative import FEATURES as REL_FEATURES, relative_features
from galaxy_its.data.residual import FEATURES as RES_FEATURES, residual_features
from galaxy_its.eval.operating import event_detection, false_alarms_per_hour, k_of_n
from galaxy_its.models.frame_models import ConvGRU

DT, RULE, FRAMES, STRIDE, WARMUP = 0.1, (3, 5), 64, 16, 30.0
SUBSETS = ["train_01_attack_free", "train_02_with_attacks", "test_01_known_vehicle_known_attack",
           "test_02_unknown_vehicle_known_attack", "test_06_masquerade"]


def build(root: Path, mode: str, vocab=None, profile=None):
    """mode 'id': ID embedding + payload bytes (+novelty); 'rel': ID-relative per-frame ratios;
    'res': per-ID behavioural residualisation."""
    seqdict = {}
    for sub in SUBSETS:
        for f in sorted((root / sub).glob("*.csv")):
            log = read_cantt_csv(f)
            rel = log.ts - log.ts[0]
            starts = np.arange(0, max(len(log) - FRAMES, 0), STRIDE)
            if mode in ("rel", "res"):
                data = (relative_features(log, warmup_s=WARMUP) if mode == "rel"
                        else residual_features(log, warmup_s=WARMUP))
                node = np.zeros(len(log), np.int64)
                starts = starts[rel[starts] >= WARMUP]          # warm-up windows excluded
            else:
                node = vocab.map(log.can_id).astype(np.int64)
                data = log.data.astype(np.float32) / 255.0
                if profile is not None:
                    data = np.concatenate([data, profile.surprisal(node, log.data)], -1)
            cs = np.concatenate([[0], np.cumsum(log.label.astype(np.int64))])
            T = int(rel[-1] / DT) + 1
            win = np.zeros(T, np.int8); np.maximum.at(win, (rel / DT).astype(np.int64), log.label)
            seqdict[f"{sub}/{f.stem}"] = {
                "node": node, "data": data, "starts": starts,
                "lab": (cs[starts + FRAMES] - cs[starts] > 0).astype(np.int8),
                "w_end": (rel[starts + FRAMES - 1] / DT).astype(np.int64),
                "win_inj": fill_short_gaps(win, int(1.0 / DT)), "T": T, "sub": sub}
    return seqdict


def train(seqdict, n_ids, n_channels, n_extra, args, rng):
    items = np.array([(n, i) for n, d in seqdict.items() if d["sub"] == "train_02_with_attacks"
                      for i in range(len(d["starts"]))], dtype=object)
    model = ConvGRU(n_ids, n_channels=n_channels, n_extra=n_extra).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device="cuda"))
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for step, (ids, dat, lab, _) in enumerate(batches(seqdict, items, FRAMES, args.batch, "cuda", True, rng)):
            if step >= args.max_steps:
                break
            loss = lossf(model(ids, dat), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  ep {ep} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s", flush=True)
    return model


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
        out[name] = win                                  # NaN where no frame window ended
    return out


def align(a: np.ndarray, b: np.ndarray, inj: np.ndarray):
    """Keep windows scored by both branches (the ID-agnostic branch skips its warm-up)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    return a[ok], b[ok], inj[ok]


def thr_for(seqs, budget):
    cand = np.unique(np.concatenate([s for s, _ in seqs]))
    cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), 200)).astype(int)]
    for t in cand:
        fa, _, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
        if fa <= budget and duty <= 0.02:
            return float(t)
    return float(cand[-1]) + 1e-6


def union_eval(pairs, names, ta, tb):
    """pairs: [(score_a, score_b, inj)]. Alarm = k-of-n on A OR k-of-n on B."""
    seqs = []
    for a, b, inj in pairs:
        al = k_of_n(a >= ta, *RULE) | k_of_n(b >= tb, *RULE)
        seqs.append((al.astype(float), inj))             # 0/1 "scores"; threshold 0.5 below
    fa, hrs, duty = false_alarms_per_hour(seqs, 0.5, 1, 1, DT)
    rows = event_detection(seqs, names, 0.5, 1, 1, DT)
    return rows, fa, duty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/cantt/set_01")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--branches", choices=["id+rel", "rel+res"], default="id+rel",
                    help="which two detectors to pair")
    ap.add_argument("--tests", nargs="+",
                    default=["test_01_known_vehicle_known_attack", "test_06_masquerade"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/pair_detector")
    args = ap.parse_args()
    root = ROOT / args.root; setname = Path(args.root).name
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.branches == "id+rel":
        src = [read_cantt_csv(f) for f in sorted((root / "train_02_with_attacks").glob("*.csv"))]
        src += [read_cantt_csv(f) for f in sorted((root / "train_01_attack_free").glob("*.csv"))]
        vocab = IdVocab.fit(src); profile = ByteProfile.fit(src, vocab); del src
        print("branch B (ID-based + payload):", flush=True)
        sd_id = build(root, "id", vocab, profile)
        mb = train(sd_id, vocab.n_nodes, 8, 8, args, rng)
    else:
        print("branch B (per-ID residualisation):", flush=True)
        sd_id = build(root, "res")
        mb = train(sd_id, 0, len(RES_FEATURES), 0, args, rng)
    print("branch A (ID-relative):", flush=True)
    sd_rel = build(root, "rel")
    ma = train(sd_rel, 0, len(REL_FEATURES), 0, args, rng)

    tests = args.tests
    names = [n for n in sd_rel if sd_rel[n]["sub"] in tests]
    cal_names = [n for n in sd_rel if sd_rel[n]["sub"] == "train_01_attack_free"]
    sa, sb = score(ma, sd_rel, names + cal_names, args.batch), score(mb, sd_id, names + cal_names, args.batch)

    results = {}
    print(f"\n{'test':40s} {'branch':16s} {'budget':>7} {'events':>9} {'FA/h':>7} {'duty':>6}")
    for sub in tests:
        subn = [n for n in names if sd_rel[n]["sub"] == sub]
        pairs = [align(sa[n], sb[n], sd_rel[n]["win_inj"]) for n in subn]
        cal = [align(sa[n], sb[n], sd_rel[n]["win_inj"]) for n in subn]     # target-own clean windows
        for budget in (5, 30):
            # each branch alone gets the full budget; the union splits it
            ta_full = thr_for([(p[0], p[2]) for p in cal], budget)
            tb_full = thr_for([(p[1], p[2]) for p in cal], budget)
            ta_half = thr_for([(p[0], p[2]) for p in cal], budget / 2)
            tb_half = thr_for([(p[1], p[2]) for p in cal], budget / 2)
            for label, rows_fa in (
                ("A id-agnostic", event_detection([(p[0], p[2]) for p in pairs], subn, ta_full, *RULE, DT)),
                ("B second branch", event_detection([(p[1], p[2]) for p in pairs], subn, tb_full, *RULE, DT)),
            ):
                seqs = [(p[0] if label.startswith("A") else p[1], p[2]) for p in pairs]
                fa, _, duty = false_alarms_per_hour(seqs, ta_full if label.startswith("A") else tb_full, *RULE, DT)
                det = sum(r["detected"] for r in rows_fa)
                results[f"{sub}|{label}|{budget}"] = {"detected": det, "events": len(rows_fa),
                                                      "fa_per_h": fa, "duty": duty}
                print(f"{sub:40s} {label:16s} {budget:7.0f} {det:4d}/{len(rows_fa):<4d} {fa:7.1f} {duty*100:5.1f}%")
            rows, fa, duty = union_eval(pairs, subn, ta_half, tb_half)
            det = sum(r["detected"] for r in rows)
            results[f"{sub}|A OR B|{budget}"] = {"detected": det, "events": len(rows),
                                                 "fa_per_h": fa, "duty": duty}
            print(f"{sub:40s} {'A OR B':16s} {budget:7.0f} {det:4d}/{len(rows):<4d} {fa:7.1f} {duty*100:5.1f}%")
    json.dump({"args": vars(args), "results": results},
              open(out / f"{setname}_{args.branches.replace(chr(43), chr(45))}_s{args.seed}.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {out / f'{setname}_s{args.seed}.json'}")


if __name__ == "__main__":
    main()
