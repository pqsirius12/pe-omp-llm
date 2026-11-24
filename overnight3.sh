#!/bin/sh
# overnight3.sh - measurement run #3 (Low Power Mode OFF, Mac idle).
#   1. import any new web replies
#   2. feedback-round (fb) and explicit-request (hwx) answers, with fresh
#      serial/expert/calibrated baselines measured in the same session
#   3. oversubscription: every variant at 12, 16 and 20 threads (more threads
#      than the 10 cores), 3 repetitions
# Gemma answers that are still being collected are simply measured next time
# (--skip-done makes every run resumable). CPU power is logged by the caller.
set -u
cd "$(dirname "$0")"

lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
[ "$lpm" = "0" ] || { echo "ABORT: Low Power Mode is on (lowpowermode=$lpm)"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "ABORT: plug in the charger"; exit 1; }

python3 import_web.py
echo "fb variants: $(ls variants/*/*__fb__*.c 2>/dev/null | wc -l)  hwx variants: $(ls variants/*/*__hwx__*.c 2>/dev/null | wc -l)"

echo "[$(date)] starting in 60 s - please leave the Mac alone (lid open)"
sleep 60

echo "[$(date)] feedback + explicit-request answers with baselines"
python3 run.py run --variants '(__fb__|__hwx__|^serial$|^expert$|^calib$)' --out timings_run3.csv --skip-done

echo "[$(date)] oversubscription (12/16/20 threads)"
python3 run.py run --threads 12,16,20 --reps 3 --out timings_oversub.csv --skip-done

echo "[$(date)] RUN3_DONE"
