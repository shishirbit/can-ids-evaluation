#!/usr/bin/env bash
# Resumable download of can-train-and-test-v1.5 set_01 (labelled CSVs, pre-defined cross-vehicle
# splits). set_01: known vehicle = Chevrolet Impala, unknown vehicle = Chevrolet Silverado.
# ~1.32 GB in 42 files. Re-run safely; completed files are skipped.
set -u
REPO="https://bitbucket.org/brooke-lampe/can-train-and-test-v1.5/raw/master"
API="https://api.bitbucket.org/2.0/repositories/brooke-lampe/can-train-and-test-v1.5/src/master"
DEST="$(dirname "$0")/../data/raw/cantt"
SET="${1:-set_01}"
SUBSETS="train_01_attack_free train_02_with_attacks test_01_known_vehicle_known_attack test_02_unknown_vehicle_known_attack test_06_masquerade"
PY="$(dirname "$0")/../.venv/Scripts/python.exe"

for sub in $SUBSETS; do
  mkdir -p "$DEST/$SET/$sub"
  files=$(curl -s "$API/$SET/$sub/?pagelen=100" | "$PY" -c "
import sys,json; d=json.load(sys.stdin)
print(' '.join(v['path'].split('/')[-1] for v in d.get('values',[]) if v['type']=='commit_file'))")
  for f in $files; do
    out="$DEST/$SET/$sub/$f"
    for attempt in 1 2 3 4 5; do
      curl -sSL -C - --connect-timeout 20 --speed-limit 2000 --speed-time 60 -o "$out" "$REPO/$SET/$sub/$f"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 33 ] && break
      echo "retry $attempt $sub/$f (rc=$rc)"; sleep 5
    done
    echo "$sub/$f $(stat -c %s "$out") bytes"
  done
done
du -sh "$DEST/$SET"
