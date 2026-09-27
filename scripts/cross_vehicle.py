"""Cross-vehicle generalisation on can-train-and-test-v1.5 set_01.

Known vehicle = Chevrolet Impala, unknown vehicle = Chevrolet Silverado (the repository's own
split definition). Trains the frame models on the known vehicle only, then evaluates:

    test_01  known vehicle, known attacks      (upper bound)
    test_02  UNKNOWN vehicle, known attacks    (cross-vehicle transfer)
    test_06  masquerade attacks

Thresholds are calibrated two ways per test set: on the *known* vehicle's attack-free traffic
(pure transfer - nothing from the target vehicle) and on the target files' own pre-onset clean
windows (adaptation). The gap between them separates "the model does not transfer" from
"the threshold does not transfer".

    python scripts/cross_vehicle.py --model convgru [--payload]
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
from galaxy_its.eval.operating import event_detection, false_alarms_per_hour
from galaxy_its.models.frame_models import ConvGRU

DT, RULE = 0.1, (3, 5)
SUBSETS = ["train_01_attack_free", "train_02_with_attacks", "test_01_known_vehicle_known_attack",
           "test_02_unknown_vehicle_known_attack", "test_06_masquerade"]


def build(root: Path, vocab, profile, frames: int, stride: int):
    seqdict = {}
    for sub in SUBSETS:
        for f in sorted((root / sub).glob("*.csv")):
            log = read_cantt_csv(f)
            node = vocab.map(log.can_id).astype(np.int64)
            data = log.data.astype(np.float32) / 255.0
            if profile is not None:
                data = np.concatenate([data, profile.surprisal(node, log.data)], -1)
            starts = np.arange(0, len(log) - frames, stride)
            cs = np.concatenate([[0], np.cumsum(log.label.astype(np.int64))])
            lab = (cs[starts + frames] - cs[starts] > 0).astype(np.int8)
            rel = log.ts - log.ts[0]
            T = int(rel[-1] / DT) + 1
            win = np.zeros(T, np.int8); np.maximum.at(win, (rel / DT).astype(np.int64), log.label)
            seqdict[f"{sub}/{f.stem}"] = {
                "node": node, "data": data, "starts": starts, "lab": lab,
                "w_end": (rel[starts + frames - 1] / DT).astype(np.int64),
                "win_inj": fill_short_gaps(win, int(1.0 / DT)), "T": T, "sub": sub}
    return seqdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/cantt/set_01")
    ap.add_argument("--payload", action="store_true")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/cross_vehicle")
    args = ap.parse_args()
    root = ROOT / args.root
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tag = f"{Path(args.root).name}_{'convgru_payload' if args.payload else 'convgru'}_s{args.seed}"
    out = Path(args.out) / tag; out.mkdir(parents=True, exist_ok=True)

    # vocabulary and byte profile from the KNOWN vehicle's training traffic only
    train_logs = [read_cantt_csv(f) for f in sorted((root / "train_02_with_attacks").glob("*.csv"))]
    train_logs += [read_cantt_csv(f) for f in sorted((root / "train_01_attack_free").glob("*.csv"))]
    vocab = IdVocab.fit(train_logs)
    profile = ByteProfile.fit(train_logs, vocab) if args.payload else None
    del train_logs
    print(f"vocab={vocab.n_nodes} (known vehicle only)", flush=True)

    seqdict = build(root, vocab, profile, args.frames, args.stride)
    train_items = np.array([(n, i) for n, d in seqdict.items() if d["sub"] == "train_02_with_attacks"
                            for i in range(len(d["starts"]))], dtype=object)
    print(f"{len(seqdict)} files, {len(train_items)} training frame-windows", flush=True)

    model = ConvGRU(vocab.n_nodes, n_extra=8 if args.payload else 0).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device="cuda"))
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for step, (ids, dat, lab, _) in enumerate(batches(seqdict, train_items, args.frames, args.batch, "cuda", True, rng)):
            if step >= args.max_steps:
                break
            loss = lossf(model(ids, dat), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"ep {ep} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def score(names):
        model.eval(); res = {}
        for name in names:
            d = seqdict[name]
            items = np.array([(name, i) for i in range(len(d["starts"]))], dtype=object)
            win = np.full(d["T"], np.nan)
            for ids, dat, _, sel in batches(seqdict, items, args.frames, args.batch, "cuda"):
                s = torch.sigmoid(model(ids, dat)).cpu().numpy()
                for (_, k), v in zip(sel, s):
                    w = d["w_end"][k]
                    win[w] = v if np.isnan(win[w]) else max(win[w], v)
            ok = ~np.isnan(win)
            res[name] = (win[ok], d["win_inj"][ok])
        return res

    by_sub = {}
    for sub in SUBSETS:
        names = [n for n, d in seqdict.items() if d["sub"] == sub]
        by_sub[sub] = score(names)
        print(f"scored {sub}: {len(names)} files", flush=True)

    def thr_for(seqs, budget):
        cand = np.unique(np.concatenate([s for s, _ in seqs]))
        cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), 200)).astype(int)]
        for t in cand:
            fa, _, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
            if fa <= budget and duty <= 0.02:
                return float(t)
        return float(cand[-1]) + 1e-6

    results = {}
    known_clean = list(by_sub["train_01_attack_free"].values())
    print(f"\n{'test set':38s} {'calibration':12s} {'events':>8} {'FA/h':>7} {'median ms':>9}")
    for sub in ("test_01_known_vehicle_known_attack", "test_02_unknown_vehicle_known_attack", "test_06_masquerade"):
        seqs = list(by_sub[sub].values()); names = list(by_sub[sub])
        for cal, cal_seqs in (("known-clean", known_clean), ("target-own", seqs)):
            for budget in (5, 30):
                t = thr_for(cal_seqs, budget)
                rows = event_detection(seqs, names, t, *RULE, DT)
                fa, hrs, duty = false_alarms_per_hour(seqs, t, *RULE, DT)
                lat = [r["latency_ms"] for r in rows if r["detected"]]
                key = f"{sub}|{cal}|{budget}"
                results[key] = {"threshold": t, "detected": sum(r["detected"] for r in rows),
                                "events": len(rows), "fa_per_h": fa, "duty": duty,
                                "median_latency_ms": float(np.median(lat)) if lat else None,
                                "rows": rows}
                r = results[key]
                print(f"{sub:38s} {cal + '@' + str(budget):12s} {r['detected']:3d}/{r['events']:<4d} "
                      f"{fa:7.1f} {r['median_latency_ms'] if lat else float('nan'):9.0f}")
    json.dump({"args": vars(args), "results": results}, open(out / "results.json", "w"), indent=2, default=float)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
