#!/usr/bin/env bash
# Resumable parallel download of the HCRL Car-Hacking dataset (official Dropbox share).
# Re-run safely at any time: completed files are skipped, partial files resume.
set -u
DEST="$(dirname "$0")/../data/raw/car_hacking"
mkdir -p "$DEST" && cd "$DEST" || exit 1
B="https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom"
K="rlkey=3h6zamu3kc262lrnipu5qden8&dl=1"

fetch() {
  local p="$1" f; f=$(basename "$p")
  for attempt in $(seq 1 40); do
    curl -sSL -C - --connect-timeout 20 --speed-limit 2000 --speed-time 60 -o "$f" "$B/$p?$K"
    rc=$?
    # 33 = server refused range (file complete or restart needed); 0 = done
    if [ $rc -eq 0 ] || [ $rc -eq 33 ]; then echo "done $f $(stat -c %s "$f") bytes"; return 0; fi
    echo "retry $attempt $f (curl rc=$rc, have $(stat -c %s "$f" 2>/dev/null || echo 0) bytes)"; sleep 5
  done
  echo "FAIL $f"
}

fetch "AFCn22TGN0tRVpNMDCq2F2Y/DoS_dataset.csv" &
fetch "AGI0rMPrcuyOlcLjJHVtj1I/gear_dataset.csv" &
fetch "AOyhvxvDP-I8aMo05PPyLxw/RPM_dataset.csv" &
fetch "AFFgJ9-R0At-wHfKV-VT8-A/normal_run_data.7z" &
wait
ls -la
