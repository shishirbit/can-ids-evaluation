"""Convert raw HCRL Car-Hacking files into CanLog .npz files and print a data report.

    python scripts/preprocess_car_hacking.py --raw data/raw/car_hacking --out data/processed/car_hacking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_its.data.can_io import read_car_hacking_csv, read_car_hacking_normal_txt


def attack_episodes(log, gap_s=1.0):
    """Contiguous attack bursts: returns (start, end) times, merging injections < gap_s apart."""
    t = log.ts[log.label == 1]
    if len(t) == 0:
        return np.zeros((0, 2))
    brk = np.where(np.diff(t) > gap_s)[0]
    starts = np.concatenate([[t[0]], t[brk + 1]]); ends = np.concatenate([t[brk], [t[-1]]])
    return np.stack([starts, ends], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/car_hacking")
    ap.add_argument("--out", default="data/processed/car_hacking")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    archive = raw / "normal_run_data.7z"
    if archive.exists() and not list(raw.glob("**/normal_run_data*.txt")):
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(raw)

    report = {}
    files = sorted(raw.glob("*_dataset.csv")) + sorted(raw.glob("**/normal_run_data*.txt"))
    for f in files:
        print(f"reading {f.name} ...", flush=True)
        log = read_car_hacking_csv(f) if f.suffix == ".csv" else read_car_hacking_normal_txt(f)
        log.save_npz(out / f"{log.name}.npz")
        s = log.summary()
        ep = attack_episodes(log)
        if len(ep):
            dur = ep[:, 1] - ep[:, 0]
            gaps = ep[1:, 0] - ep[:-1, 1]
            s.update({
                "episodes": len(ep),
                "episode_len_s": [round(float(x), 2) for x in (dur.min(), np.median(dur), dur.max())],
                "gap_s(min/med/max)": [round(float(x), 2) for x in (gaps.min(), np.median(gaps), gaps.max())] if len(gaps) else None,
                "gap_cv": round(float(gaps.std() / gaps.mean()), 3) if len(gaps) > 1 else None,
            })
        report[log.name] = s
        print(json.dumps(s, indent=1), flush=True)
    json.dump(report, open(out / "report.json", "w"), indent=2)
    print(f"saved {len(report)} logs -> {out}")


if __name__ == "__main__":
    main()
