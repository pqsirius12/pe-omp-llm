#!/bin/sh
# overnight2.sh - measurement run #2 (Low Power Mode OFF, Mac idle).
#   1. import any new web replies (Claude bundle) -> variants/*/claude-web__*.c
#   2. histogram: re-measure EVERY variant with the -fno-vectorize fix
#      (Apple clang vectorizer miscompile; flag now set per kernel in run.py)
#   3. new Claude variants + serial + expert baselines on all kernels, so the
#      new variants are compared against baselines measured in the same session
# Output: results/timings_run2.csv. CPU power is logged by the caller (see chat).
set -u
cd "$(dirname "$0")"

lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
[ "$lpm" = "0" ] || { echo "ABORT: Low Power Mode is on (lowpowermode=$lpm)"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "ABORT: plug in the charger"; exit 1; }

python3 import_web.py
echo "claude variants: $(ls variants/*/claude-web__*.c 2>/dev/null | wc -l)"

echo "[$(date)] starting in 60 s - please leave the Mac alone (lid open)"
sleep 60

echo "[$(date)] histogram, all variants (vectorizer fix)"
python3 run.py run --kernels histogram --out timings_run2.csv --skip-done

echo "[$(date)] Claude variants + baselines, all kernels"
python3 run.py run --variants '^(claude-web|serial|expert|calib)' --out timings_run2.csv --skip-done

echo "[$(date)] RUN2_DONE"
