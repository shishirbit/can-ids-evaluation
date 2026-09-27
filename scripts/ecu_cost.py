"""Compute/latency budget for in-vehicle deployment.

Measures, for the models we actually compare:
  * parameter count and fp32 / int8 model size
  * batch-1 inference latency on one CPU thread (median, p99) and on the GPU
  * the required inference *rate* implied by each model's evaluation protocol, and the resulting
    CPU duty cycle (fraction of a single core consumed)
  * feature-extraction cost per second of CAN traffic (the part most papers omit)

A laptop CPU core is several times faster than a typical automotive Cortex-A/R core, so the CPU
numbers are optimistic lower bounds; the duty-cycle column is what scales.

    python scripts/ecu_cost.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from galaxy_its.data.can_io import CanLog
from galaxy_its.data.graphs import IdVocab, build_graph_sequence
from galaxy_its.data.payload import ByteProfile, payload_features
from galaxy_its.models.frame_models import ConvGRU
from galaxy_its.models.sttf import STTF, STTFConfig

N_IDS, FRAMES, DT = 107, 64, 0.1
FRAME_RATE = 2400.0            # ROAD: ~2400 frames/s
STRIDE = 16                    # ConvGRU sliding-window stride -> 150 inferences/s


def bench(fn, n_warm=5, n_rep=50):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3), float(np.percentile(ts, 99) * 1e3)


def model_size(m):
    n = sum(p.numel() for p in m.parameters())
    return n, n * 4 / 1e6, n / 1e6          # params, MB fp32, MB int8


def main():
    torch.set_num_threads(1)
    rows = []

    # --- ConvGRU + payload novelty (the configuration that wins) ---
    cg = ConvGRU(N_IDS, n_extra=8).eval()
    ids = torch.zeros(1, FRAMES, dtype=torch.long); dat = torch.zeros(1, FRAMES, 16)
    with torch.no_grad():
        med, p99 = bench(lambda: cg(ids, dat))
    n, mb32, mb8 = model_size(cg)
    rate = FRAME_RATE / STRIDE
    rows.append(("ConvGRU+payload (CPU, 1 thread)", n, mb32, mb8, med, p99, rate, med * rate / 1000))

    # --- STTF (graph) ---
    cfg = STTFConfig(n_nodes=N_IDS, n_feat=32, n_horizons=4, seq_len=20, d_model=128, gat_heads=4)
    st = STTF(cfg).eval()
    x = torch.zeros(1, 20, N_IDS, 32); adj = torch.zeros(1, 20, N_IDS, N_IDS)
    with torch.no_grad():
        med_s, p99_s = bench(lambda: st(x, adj), n_rep=20)
    n_s, mb32_s, mb8_s = model_size(st)
    rows.append(("STTF graph (CPU, 1 thread)", n_s, mb32_s, mb8_s, med_s, p99_s, 1 / DT, med_s * (1 / DT) / 1000))
    # STTF with 50 MC-dropout passes, as specified in the proposal
    rows.append(("STTF graph + 50x MC dropout (CPU)", n_s, mb32_s, mb8_s, med_s * 50, p99_s * 50,
                 1 / DT, med_s * 50 * (1 / DT) / 1000))

    if torch.cuda.is_available():
        cg_g, st_g = cg.cuda(), st.cuda()
        ids_g, dat_g = ids.cuda(), dat.cuda()
        xg, ag = x.cuda(), adj.cuda()
        with torch.no_grad():
            def run_cg():
                cg_g(ids_g, dat_g); torch.cuda.synchronize()

            def run_st():
                st_g(xg, ag); torch.cuda.synchronize()
            med_g, p99_g = bench(run_cg)
            med_sg, p99_sg = bench(run_st, n_rep=20)
        rows.append(("ConvGRU+payload (RTX 5060)", n, mb32, mb8, med_g, p99_g, rate, med_g * rate / 1000))
        rows.append(("STTF graph (RTX 5060)", n_s, mb32_s, mb8_s, med_sg, p99_sg, 1 / DT, med_sg * (1 / DT) / 1000))

    print(f"{'model':36s} {'params':>8} {'fp32 MB':>7} {'int8 MB':>7} {'med ms':>7} {'p99 ms':>7} "
          f"{'infer/s':>8} {'core duty':>9}")
    for r in rows:
        print(f"{r[0]:36s} {r[1]:8d} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7.2f} {r[6]:8.0f} "
              f"{r[7] * 100:8.1f}%")

    # --- feature extraction cost on real traffic ---
    road = ROOT / "data/processed/road"
    log = CanLog.load_npz(road / "ambient_dyno_drive_basic_short.npz")
    vocab = IdVocab.fit([log])
    profile = ByteProfile.fit([log], vocab)
    secs = float(log.ts[-1] - log.ts[0])
    t0 = time.perf_counter(); seq = build_graph_sequence(log, vocab, DT); t_graph = time.perf_counter() - t0
    t0 = time.perf_counter(); payload_features(log, vocab, profile, DT, len(seq)); t_pay = time.perf_counter() - t0
    print(f"\nfeature extraction over {secs:.0f} s of traffic ({len(log)/1e6:.2f} M frames, 1 CPU thread):")
    print(f"  window/graph features : {t_graph:6.2f} s  -> {100 * t_graph / secs:5.2f}% of one core")
    print(f"  payload novelty       : {t_pay:6.2f} s  -> {100 * t_pay / secs:5.2f}% of one core")
    print("  (batch numpy implementation; a streaming implementation would differ)")

    json.dump({"rows": [dict(zip(("model", "params", "mb_fp32", "mb_int8", "median_ms", "p99_ms",
                                  "infer_per_s", "core_duty"), r)) for r in rows],
               "feature_extraction": {"seconds_of_traffic": secs, "graph_s": t_graph, "payload_s": t_pay}},
              open(ROOT / "experiments" / "ecu_cost.json", "w"), indent=2)
    print("\nsaved -> experiments/ecu_cost.json")


if __name__ == "__main__":
    main()
