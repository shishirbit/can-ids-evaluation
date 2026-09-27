"""Convert ROAD raw CAN captures into labelled CanLog .npz files plus a capture index.

    python scripts/preprocess_road.py --raw data/raw/road/road --out data/processed/road

Included: dyno ambient captures (the attacks were recorded on the dyno) except
`ambient_dyno_exercise_all_bits` (a synthetic bit-exercising capture, 36 min, not normal driving);
all attack captures with a labelled injection (original + `_masquerade`).
Excluded: accelerator attacks (no injected frames / no labels), on-road ambient captures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_its.data.can_io import label_road, read_candump_log

SKIP_AMBIENT = {"ambient_dyno_exercise_all_bits"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/road/road")
    ap.add_argument("--out", default="data/processed/road")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}

    amb = json.load(open(raw / "ambient" / "capture_metadata.json"))
    for name, m in amb.items():
        if not m["on_dyno"] or name in SKIP_AMBIENT:
            continue
        log = read_candump_log(raw / "ambient" / f"{name}.log")
        log.save_npz(out / f"{name}.npz")
        index[name] = {"kind": "ambient", **log.summary()}
        print(index[name], flush=True)

    att = json.load(open(raw / "attacks" / "capture_metadata.json"))
    for name, m in att.items():
        if not m.get("injection_interval"):
            continue
        log = label_road(read_candump_log(raw / "attacks" / f"{name}.log"), m)
        log.save_npz(out / f"{name}.npz")
        base = name.removesuffix("_masquerade")
        mm = re.match(r"(.*?)(?:_(\d+))?$", base)
        index[name] = {"kind": "attack", "family": mm.group(1), "instance": int(mm.group(2) or 1),
                       "masquerade": name.endswith("_masquerade"), "base": base,
                       "injection_interval": m["injection_interval"], **log.summary()}
        print({k: index[name][k] for k in ("family", "instance", "masquerade", "frames", "attack_frames")}, flush=True)

    json.dump(index, open(out / "index.json", "w"), indent=2)
    print(f"saved {len(index)} captures -> {out}")


if __name__ == "__main__":
    main()
