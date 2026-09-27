"""Train/evaluate the frame-sequence baselines (ConvGRU, FrameAE) under the ROAD v3 protocol.

    python scripts/baseline_frames.py --model convgru --fold 0

Sliding windows of F consecutive frames (stride S) are labelled attacked if they contain any
injected frame. Scores are mapped to the 100 ms evaluation windows (max over the frame-windows
ending in each) so the event-level curves are directly comparable with STTF.
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
from galaxy_its.data.graphs import IdVocab, fill_short_gaps
from galaxy_its.data.payload import ByteProfile
from galaxy_its.models.frame_models import ConvGRU, FrameAE
from train_sttf import load_road

DT, L = 0.1, 20            # must match the STTF runs: 100 ms windows, 20-window history offset


def prepare(args):
    logs, assign = load_road(argparse.Namespace(road=args.road, fold=args.fold, ambient_holdout=True))
    vocab = IdVocab.fit([l.slice(0, int(len(l) * 0.7)) if a == "temporal" else l
                         for l, a in zip(logs, assign) if a in ("temporal", "train")])
    F, S = args.frames, args.stride
    profile = ByteProfile.fit([l.slice(0, int(len(l) * 0.7)) if a == "temporal" else l
                               for l, a in zip(logs, assign) if a in ("temporal", "train")], vocab)         if args.payload else None
    out = {}
    for log, a in zip(logs, assign):
        node = vocab.map(log.can_id).astype(np.int64)
        data = log.data.astype(np.float32) / 255.0
        if profile is not None:      # per-frame payload novelty: -log p(byte bin | ID) / log B
            data = np.concatenate([data, profile.surprisal(node, log.data)], -1)
        starts = np.arange(0, len(log) - F, S)
        lab = np.zeros(len(starts), np.int8)
        cs = np.concatenate([[0], np.cumsum(log.label.astype(np.int64))])
        lab = (cs[starts + F] - cs[starts] > 0).astype(np.int8)
        # 100 ms window index of each frame-window's last frame, and window-level attack labels
        w_end = ((log.ts[starts + F - 1] - log.ts[0]) / DT).astype(np.int64)
        T = int((log.ts[-1] - log.ts[0]) / DT) + 1
        win_inj = np.zeros(T, np.int8)
        wi = ((log.ts - log.ts[0]) / DT).astype(np.int64)
        np.maximum.at(win_inj, wi, log.label)
        win_inj = fill_short_gaps(win_inj, int(1.0 / DT))
        split = a if a != "temporal" else None
        out[log.name] = {"node": node, "data": data, "starts": starts, "lab": lab, "w_end": w_end,
                         "win_inj": win_inj, "T": T, "split": split, "assign": a}
    return out, vocab


def frame_split(seqdict, frac=(0.70, 0.15, 0.15)):
    """Sequences assigned to a split contribute all their frame-windows; 'temporal' ones are cut."""
    idx = {k: [] for k in ("train", "val", "test")}
    for name, d in seqdict.items():
        n = len(d["starts"])
        if d["assign"] == "temporal":
            a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
            for k, (lo, hi) in {"train": (0, a), "val": (a, b), "test": (b, n)}.items():
                idx[k] += [(name, i) for i in range(lo, hi)]
        else:
            idx[d["assign"]] += [(name, i) for i in range(n)]
    return {k: np.array(v, dtype=object) for k, v in idx.items()}


def batches(seqdict, items, F, batch, device, shuffle=False, rng=None):
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for i in range(0, len(order), batch):
        sel = items[order[i:i + batch]]
        n_ch = seqdict[sel[0][0]]["data"].shape[1]
        ids = np.empty((len(sel), F), np.int64); dat = np.empty((len(sel), F, n_ch), np.float32)
        lab = np.empty(len(sel), np.float32)
        for j, (name, k) in enumerate(sel):
            d = seqdict[name]; s = d["starts"][k]
            ids[j] = d["node"][s:s + F]; dat[j] = d["data"][s:s + F]; lab[j] = d["lab"][k]
        yield (torch.from_numpy(ids).to(device), torch.from_numpy(dat).to(device),
               torch.from_numpy(lab).to(device), sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["convgru", "frameae"], default="convgru")
    ap.add_argument("--road", default="data/processed/road")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--payload", action="store_true", help="add per-frame payload-novelty channels")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=400, help="steps per epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/baseline_frames")
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = "cuda"
    suffix = "" if args.seed == 0 else f"_s{args.seed}"
    tag = args.model + ("_payload" if args.payload else "")
    out = Path(args.out) / f"{tag}{suffix}_fold{args.fold}"; out.mkdir(parents=True, exist_ok=True)

    seqdict, vocab = prepare(args)
    sp = frame_split(seqdict)
    print({k: len(v) for k, v in sp.items()}, f"n_ids={vocab.n_nodes}", flush=True)
    n_ch = 16 if args.payload else 8
    model = (ConvGRU(vocab.n_nodes, n_extra=n_ch - 8) if args.model == "convgru"
             else FrameAE(vocab.n_nodes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=dev))

    train_items = sp["train"]
    if args.model == "frameae":                      # unsupervised: attack-free frame windows only
        keep = [i for i, (n, k) in enumerate(train_items) if seqdict[n]["lab"][k] == 0]
        train_items = train_items[keep]
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for step, (ids, dat, lab, _) in enumerate(batches(seqdict, train_items, args.frames, args.batch, dev, True, rng)):
            if step >= args.max_steps:
                break
            if args.model == "convgru":
                loss = lossf(model(ids, dat), lab)
            else:
                loss = model(ids, dat)[0].mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"ep {ep} loss {np.mean(losses):.4f} {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def score(items):
        model.eval()
        per_seq = {}
        for ids, dat, _, sel in batches(seqdict, items, args.frames, args.batch, dev):
            s = (torch.sigmoid(model(ids, dat)) if args.model == "convgru" else model(ids, dat)[0]).cpu().numpy()
            for (name, k), v in zip(sel, s):
                per_seq.setdefault(name, {})[int(k)] = float(v)
        # map frame-window scores onto 100 ms evaluation windows (max), aligned with STTF windows
        res = {}
        for name, d in per_seq.items():
            sd = seqdict[name]
            win = np.full(sd["T"], np.nan)
            for k, v in d.items():
                w = sd["w_end"][k]
                win[w] = v if np.isnan(win[w]) else max(win[w], v)
            valid = ~np.isnan(win)
            valid[:L - 1] = False                    # STTF needs L-1 windows of history
            res[name] = (win[valid], sd["win_inj"][valid])
        return res

    for split in ("val", "test"):
        res = score(sp[split])
        np.savez(out / f"{split}_seqs.npz", names=np.array(list(res)),
                 p=np.array([v[0] for v in res.values()], dtype=object),
                 inj=np.array([v[1] for v in res.values()], dtype=object), allow_pickle=True)
        n_att = sum(int(v[1].any()) for v in res.values())
        print(f"{split}: {len(res)} sequences, {n_att} with attacks", flush=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
