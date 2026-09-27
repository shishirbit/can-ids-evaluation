"""Train and evaluate STTF, plus persistence and timing baselines.

Examples
    # pipeline check on synthetic data (no downloads needed)
    python scripts/train_sttf.py --synthetic dos --epochs 5
    python scripts/train_sttf.py --synthetic dos --precursor --epochs 10

    # real data (after scripts/preprocess_car_hacking.py)
    python scripts/train_sttf.py --data data/processed/car_hacking --epochs 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_its.data import synthetic
from galaxy_its.data.can_io import CanLog, split_on_gaps
from galaxy_its.data.dataset import GraphSeqStore, assigned_split, temporal_split
from galaxy_its.data.graphs import NODE_FEATURES, IdVocab, build_graph_sequence
from galaxy_its.data.payload import PAYLOAD_FEATURES, ByteProfile, payload_features
from galaxy_its.eval.baselines import persistence, timing_fit_predict
from galaxy_its.eval.metrics import best_f1_threshold, evaluate, format_table, onset_latency
from galaxy_its.models.sttf import STTF, STTFConfig, sttf_loss


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, help="dir of CanLog .npz files")
    ap.add_argument("--road", type=str, help="dir of preprocessed ROAD captures (with index.json)")
    ap.add_argument("--fold", type=int, default=0, help="ROAD CV fold 0-2: test instance fold+1")
    ap.add_argument("--ambient-holdout", action="store_true",
                    help="hold out whole ambient captures per split instead of time slices")
    ap.add_argument("--synthetic", choices=["dos", "fuzzy", "spoof"])
    ap.add_argument("--precursor", action="store_true")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--horizons", type=float, nargs="+", default=[0, 1, 2, 5, 9], help="seconds")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--gat-heads", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--payload", action="store_true", help="add byte-level payload features (v2)")
    ap.add_argument("--per-node-norm", action="store_true", help="standardise features per (node, feature)")
    ap.add_argument("--no-graph", action="store_true", help="ablation (a)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--mc-passes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/sttf")
    ap.add_argument("--ckpt", type=str, help="evaluate this checkpoint instead of training")
    return ap.parse_args()


def load_logs(args) -> tuple[list[CanLog], list[str]]:
    """Returns logs and a per-log split assignment ('temporal' or 'train'/'val'/'test')."""
    if args.synthetic:
        return [synthetic.generate(duration_s=900, attack=args.synthetic, precursor=args.precursor, seed=args.seed)], ["temporal"]
    if args.road:
        return load_road(args)
    return load_car_hacking(args), None


def load_road(args):
    """Capture-level CV. Each attack family has instances 1-3; fold f tests instance f+1,
    validates on the next instance, trains on the rest. An attack and its _masquerade copy
    (the same injection event) always share a split. Ambient captures are split in time."""
    index = json.load(open(Path(args.road) / "index.json"))
    test_inst, val_inst = args.fold + 1, (args.fold + 1) % 3 + 1
    # whole-capture ambient holdout: ambient captures sorted by duration are dealt round-robin
    # across folds, so calibration and test see different drives (thresholds must transfer
    # between recordings, as in deployment) instead of slices of the same drive
    amb = sorted([n for n, m in index.items() if m["kind"] == "ambient"],
                 key=lambda n: -index[n]["duration_s"])
    amb_split = {n: ("test", "val", "train")[(i + args.fold) % 3] if args.ambient_holdout else "temporal"
                 for i, n in enumerate(amb)}
    logs, assign = [], []
    for name, m in index.items():
        logs.append(CanLog.load_npz(Path(args.road) / f"{name}.npz"))
        if m["kind"] == "ambient":
            assign.append(amb_split[name])
        elif m["instance"] == test_inst:
            assign.append("test")
        elif m["instance"] == val_inst:
            assign.append("val")
        else:
            assign.append("train")
    for l, a in zip(logs, assign):
        print(f"  {a:8s} {l.name}")
    return logs, assign


def load_car_hacking(args) -> list[CanLog]:
    # HCRL Car-Hacking appends the same 481.6 s attack-free recording to every attack file;
    # keep one copy so it is not counted four times.
    segs, seen = [], set()
    for p in sorted(Path(args.data).glob("*.npz")):
        for seg in split_on_gaps(CanLog.load_npz(p)):
            key = (len(seg), float(seg.ts[0]), float(seg.ts[-1]))
            if key in seen:
                print(f"skip duplicate segment {seg.name}")
                continue
            seen.add(key); segs.append(seg)
    return segs


@torch.no_grad()
def predict(model, store, gidx, batch=256, mc_passes=0):
    model.eval()
    ps, vs = [], []
    for i in range(0, len(gidx), batch):
        b = store.batch(gidx[i:i + batch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mc_passes:
                m, v = model.mc_predict(b["x"], b["adj"], mc_passes)
            else:
                m = torch.sigmoid(model(b["x"], b["adj"])["logit"]); v = torch.zeros_like(m)
        ps.append(m.float().cpu()); vs.append(v.float().cpu())
    return torch.cat(ps).numpy(), torch.cat(vs).numpy()


def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    horizons = [int(round(h / args.dt)) for h in args.horizons]
    L = args.seq_len

    logs, assign = load_logs(args)
    assign = assign or ["temporal"] * len(logs)
    if not args.road:
        for l in logs:
            print(l.summary())
    # vocab from training data only: first 70% of temporally split logs + train-assigned logs
    vocab = IdVocab.fit([l.slice(0, int(len(l) * 0.7)) if a == "temporal" else l
                         for l, a in zip(logs, assign) if a in ("temporal", "train")])
    seqs = [build_graph_sequence(l, vocab, args.dt) for l in logs]
    n_feat = len(NODE_FEATURES)
    if args.payload:
        profile = ByteProfile.fit([l.slice(0, int(len(l) * 0.7)) if a == "temporal" else l
                                   for l, a in zip(logs, assign) if a in ("temporal", "train")], vocab)
        for l, s in zip(logs, seqs):
            s.x = np.concatenate([s.x, payload_features(l, vocab, profile, args.dt, len(s))], -1)
        n_feat += len(PAYLOAD_FEATURES)
    print(f"nodes={vocab.n_nodes} windows={sum(len(s) for s in seqs)} in {len(seqs)} sequences")

    splits = (temporal_split(seqs, L, horizons) if all(a == "temporal" for a in assign)
              else assigned_split(seqs, L, horizons, assign))
    # feature normalisation from training windows only
    tr = splits["train"]
    tr_x = np.concatenate([seqs[s].x[tr.t[tr.seq_id == s]] for s in np.unique(tr.seq_id)])
    axis = 0 if args.per_node_norm else (0, 1)          # per (node, feature) or per feature
    store = GraphSeqStore(seqs, horizons, L, "cuda", tr_x.mean(axis), tr_x.std(axis) + 1e-6)
    g = {k: store.global_index(v) for k, v in splits.items()}
    print({k: len(v) for k, v in g.items()})
    for k, v in g.items():
        pos_frac = (store.y[v] == 1).float().mean(0).cpu().numpy().round(3)
        print(f"  {k} positive rate per horizon: {pos_frac}")
        if (pos_frac == 0).all():
            raise SystemExit(f"{k} split has no attack windows - check recording gaps / split fractions")

    ytr = store.y[g["train"]].float()
    pos = (ytr == 1).float().sum(0); neg = (ytr == 0).float().sum(0)
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=20)
    print("train positive rate per horizon:", (pos / (pos + neg)).cpu().numpy().round(3))

    cfg = STTFConfig(n_nodes=vocab.n_nodes, n_feat=n_feat, n_horizons=len(horizons),
                     seq_len=L, d_model=args.d_model, gat_heads=args.gat_heads,
                     use_graph=not args.no_graph)
    model = STTF(cfg).cuda()
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)

    best, bad, hist = -1.0, 0, []
    if args.ckpt:
        shutil.copy(args.ckpt, out / "best.pt")
    for ep in range(0 if args.ckpt else args.epochs):
        model.train(); t0 = time.time(); losses = []
        for _ in range(args.steps_per_epoch):
            idx = g["train"][torch.randint(len(g["train"]), (args.batch,), device="cuda")]
            b = store.batch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b["x"], b["adj"])
            loss, parts = sttf_loss({k: v.float() for k, v in o.items()}, b, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        sched.step()
        pv, _ = predict(model, store, g["val"], args.eval_batch)
        yv = store.y[g["val"]].cpu().numpy()
        f1s = [evaluate(yv, pv, np.ones(len(yv), bool), args.horizons)[f"{h:g}s"]["all"]["f1"] for h in args.horizons]
        vf1 = float(np.nanmean(f1s))
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_f1": vf1})
        print(f"ep {ep:3d} loss {np.mean(losses):.4f} val_f1(mean over horizons) {vf1:.4f} "
              f"[{', '.join(f'{f:.3f}' for f in f1s)}] {time.time() - t0:.1f}s")
        if vf1 > best:
            best, bad = vf1, 0; torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best.pt"))
    # thresholds tuned on validation, applied to test
    pv, _ = predict(model, store, g["val"], args.eval_batch)
    yv = store.y[g["val"]].cpu().numpy()
    thr = [best_f1_threshold(yv[yv[:, k] >= 0, k], pv[yv[:, k] >= 0, k]) for k in range(len(horizons))]

    pt, var = predict(model, store, g["test"], args.eval_batch, mc_passes=args.mc_passes)
    yt = store.y[g["test"]].cpu().numpy()
    clean = store.inj[g["test"]].cpu().numpy() == 0
    results = {"sttf": evaluate(yt, pt, clean, args.horizons, thr)}

    inj_now = store.inj[g["test"]].cpu().numpy()
    results["persistence"] = evaluate(yt, persistence(inj_now, len(horizons)), clean, args.horizons)
    gtr, gva, gte = (g[k].cpu().numpy() for k in ("train", "val", "test"))
    ytr_np = store.y[g["train"]].cpu().numpy()
    tr_args = (store.tsla[gtr], store.elapsed[gtr], ytr_np)
    p_tim_val = timing_fit_predict(*tr_args, store.tsla[gva], store.elapsed[gva])
    p_tim = timing_fit_predict(*tr_args, store.tsla[gte], store.elapsed[gte])
    thr_tim = [best_f1_threshold(yv[yv[:, k] >= 0, k], p_tim_val[yv[:, k] >= 0, k]) for k in range(len(horizons))]
    results["timing"] = evaluate(yt, p_tim, clean, args.horizons, thr_tim)
    for name, r in results.items():
        print(f"\n=== {name} (test) ===\n{format_table(r)}")
    print(f"\nMC-dropout mean predictive variance: {float(var.mean()):.5f}")

    latency = None
    if 0 in horizons:
        k0 = horizons.index(0)
        latency = onset_latency(store.inj.cpu().numpy(), store.seq_of, gte, pt[:, k0], thr[k0], args.dt)
        print(f"detection latency (window-quantised, dt={args.dt}s): {latency}")

    np.savez(out / "val_preds.npz", gidx=gva, seq=store.seq_of[gva], y=yv, p=pv,
             clean=(store.inj[g["val"]].cpu().numpy() == 0),
             seq_names=np.asarray([s.name for s in seqs]))
    np.savez(out / "test_preds.npz", gidx=gte, seq=store.seq_of[gte], y=yt, p=pt, var=var, clean=clean,
             p_timing=p_tim, thr=np.asarray(thr), thr_timing=np.asarray(thr_tim),
             seq_names=np.asarray([s.name for s in seqs]))
    json.dump({"args": vars(args), "history": hist, "thresholds": thr, "results": results,
               "latency": latency, "n_nodes": vocab.n_nodes},
              open(out / "results.json", "w"), indent=2, default=float)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
