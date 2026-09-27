"""Direction A: non-deep detection baselines on ROAD with the same folds, windows and metrics as STTF.

    python scripts/detect_road.py --road data/processed/road

For each CV fold: fit on the training split (unsupervised detectors on attack-free training
windows only), tune the alarm threshold on validation (max F1), evaluate on test. Results are
pooled over folds: window-level AUC/F1, and per-attack-family detection rate and latency.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_road import family, per_capture
from galaxy_its.data.can_io import CanLog
from galaxy_its.data.dataset import assigned_split
from galaxy_its.data.graphs import IdVocab, build_graph_sequence, forecast_labels
from galaxy_its.data.payload import ByteProfile, payload_features
from galaxy_its.eval.detectors import RateDetector, SurpriseDetector, WindowGBDT
from galaxy_its.eval.metrics import best_f1_threshold, binary_metrics
from galaxy_its.eval.operating import operating_report
from train_sttf import load_road

DT, L, HORIZONS = 0.1, 20, [0, 10, 20, 50]     # identical windows to the STTF ROAD runs


def build_fold(road: str, fold: int):
    logs, assign = load_road(argparse.Namespace(road=road, fold=fold, ambient_holdout=True))
    train_logs = [l.slice(0, int(len(l) * 0.7)) if a == "temporal" else l
                  for l, a in zip(logs, assign) if a in ("temporal", "train")]
    vocab = IdVocab.fit(train_logs)
    profile = ByteProfile.fit(train_logs, vocab)
    seqs, X = [], []
    for l in logs:
        s = build_graph_sequence(l, vocab, DT)
        seqs.append(s)
        X.append(np.concatenate([s.x, payload_features(l, vocab, profile, DT, len(s))], -1))
    return seqs, X, assigned_split(seqs, L, HORIZONS, assign)


def gather(seqs, X, split):
    x = np.concatenate([X[s][split.t[split.seq_id == s]] for s in np.unique(split.seq_id)])
    y = np.concatenate([seqs[s].inj[split.t[split.seq_id == s]] for s in np.unique(split.seq_id)])
    seq = np.concatenate([np.full((split.seq_id == s).sum(), s) for s in np.unique(split.seq_id)])
    t = np.concatenate([split.t[split.seq_id == s] for s in np.unique(split.seq_id)])
    return x, y.astype(int), seq, t


def by_sequence(p, y, seq, t):
    """[(seq_index, (scores, inj)), ...] with windows in time order."""
    out = []
    for s in np.unique(seq):
        m = seq == s
        o = np.argsort(t[m])
        out.append((int(s), (p[m][o], y[m][o])))
    return out


def tier(name: str) -> str:
    return "contextual" if name.startswith("reverse_light") else "value"


def print_operating(operating: dict):
    """Pool the per-fold operating reports: same (rule, budget) across folds."""
    print("\n=== operating points (threshold set on validation for a false-alarm budget) ===")
    print(f"{'detector':12s} {'rule':7s} {'budget/h':>8} {'test FA/h':>9} {'duty%':>6} "
          f"{'value':>7} {'value-masq':>10} {'context':>7} {'ctx-masq':>8} {'median ms':>9}")
    for name, folds in operating.items():
        for i, cfg in enumerate(folds[0]):
            ev = [e for f in folds for e in f[i]["events"]]
            fa_events = sum(f[i]["test_false_alarms_per_h"] * f[i]["test_clean_hours"] for f in folds)
            hrs = sum(f[i]["test_clean_hours"] for f in folds)

            def cnt(t, masq):
                sel = [e for e in ev if tier(e["name"]) == t and e["name"].endswith("_masquerade") == masq]
                return f"{sum(e['detected'] for e in sel)}/{len(sel)}"
            lat = [e["latency_ms"] for e in ev if e["detected"]]
            duty = np.mean([f[i].get("test_alarm_duty", float("nan")) for f in folds]) * 100
            print(f"{name:12s} {cfg['rule']:7s} {cfg['budget_per_h']:8.0f} {fa_events / max(hrs, 1e-9):9.1f} {duty:6.2f} "
                  f"{cnt('value', False):>7} {cnt('value', True):>10} {cnt('contextual', False):>7} "
                  f"{cnt('contextual', True):>8} {np.median(lat) if lat else float('nan'):9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--road", default="data/processed/road")
    ap.add_argument("--out", default="experiments/road_detectors")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pooled = defaultdict(lambda: {"y": [], "p": [], "bin": []})
    captures = defaultdict(list)
    operating = defaultdict(list)
    for fold in range(3):
        print(f"\n##### fold {fold}", flush=True)
        seqs, X, sp = build_fold(args.road, fold)
        names = np.asarray([s.name for s in seqs])
        (xtr, ytr, _, _), (xva, yva, sva, tva), (xte, yte, ste, tte) = (gather(seqs, X, sp[k]) for k in ("train", "val", "test"))
        print(f"train {len(xtr)} val {len(xva)} test {len(xte)} windows; features {xtr.shape[1:]}", flush=True)
        dets = {"rate": RateDetector().fit(xtr[ytr == 0]),
                "surprise": SurpriseDetector().fit(xtr[ytr == 0]),
                "window_gbdt": WindowGBDT().fit(xtr, ytr)}
        for name, d in dets.items():
            thr = best_f1_threshold(yva, d.score(xva))
            p = d.score(xte)
            m = binary_metrics(yte, p, thr)
            print(f"  {name:12s} test AUC {m['auc_roc']:.4f}  F1 {m['f1']:.4f}  FPR {m['fpr']:.4f}", flush=True)
            pooled[name]["y"].append(yte); pooled[name]["p"].append(p); pooled[name]["bin"].append(p >= thr)
            z = {"seq": ste, "gidx": tte, "y": yte[:, None], "seq_names": names}
            captures[name] += per_capture(z, p, thr)
            vs = by_sequence(d.score(xva), yva, sva, tva)
            ts = by_sequence(p, yte, ste, tte)
            np.savez(out / f"scores_{name}_fold{fold}.npz", p=p, y=yte, seq=ste, gidx=tte, seq_names=names)
            operating[name].append(operating_report([v for _, v in vs], [v for _, v in ts],
                                                    [names[i] for i, _ in ts], DT))

    summary = {}
    for name, d in pooled.items():
        y = np.concatenate(d["y"]); p = np.concatenate(d["p"]); b = np.concatenate(d["bin"]).astype(float)
        m = binary_metrics(y, p); mb = binary_metrics(y, b, 0.5)
        rows = captures[name]
        att = [r for r in rows if r["attack"]]; amb = [r for r in rows if not r["attack"]]
        hours = sum(r["hours"] for r in amb); fa = sum(r["false_alarm_windows"] for r in amb)
        fam = defaultdict(list)
        for r in att:
            fam[family(r["name"])].append(r)
        print(f"\n=== {name} (pooled 3 folds) AUC {m['auc_roc']:.4f} AUC-PR {m['auc_pr']:.4f} "
              f"F1 {mb['f1']:.4f} FPR {mb['fpr']:.4f} | false alarms {fa / max(hours, 1e-9):.1f}/h")
        for f, rs in sorted(fam.items()):
            lat = [r["latency_ms"] for r in rs if r["detected"]]
            print(f"   {f:32s} detected {sum(r['detected'] for r in rs)}/{len(rs)}  "
                  f"median latency {np.median(lat) if lat else float('nan'):.0f} ms")
        summary[name] = {"auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"], "f1": mb["f1"], "fpr": mb["fpr"],
                         "false_alarms_per_hour": fa / max(hours, 1e-9), "captures": rows}
    print_operating(operating)
    for name in summary:
        summary[name]["operating"] = operating[name]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsaved -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
