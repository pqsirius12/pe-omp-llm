#!/bin/sh
# overnight4.sh - measurement run #4 (Low Power Mode OFF, Mac idle):
# DeepSeek API answers (deepseek-flash = DeepSeek-V4.1-Flash-1017, thinking disabled)
# with serial/expert/calibrated baselines measured in the same session.
set -u
cd "$(dirname "$0")"
lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
[ "$lpm" = "0" ] || { echo "ABORT: Low Power Mode is on (lowpowermode=$lpm)"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "ABORT: plug in the charger"; exit 1; }
echo "deepseek-api variants: $(ls variants/*/deepseek-api__*.c 2>/dev/null | wc -l)"
echo "[$(date)] starting in 60 s - please leave the Mac alone (lid open)"
sleep 60
python3 run.py run --variants '^(deepseek-api|serial|expert|calib)' --out timings_run4.csv --skip-done
echo "[$(date)] RUN4_DONE"
