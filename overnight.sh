#!/bin/sh
# overnight.sh - Low-Power-Mode-OFF measurement session (run with the Mac idle).
#   1. safety checks (LPM off, on charger)
#   2. idle re-run of the core-speed probes      -> results/*_lpmoff.*
#   3. full benchmark, all variants, 5 reps      -> results/timings_lpmoff.csv
# CPU power is logged separately by powermetrics, started by the caller under sudo
# (see the command in chat); this script never asks for a password itself.
set -u
cd "$(dirname "$0")"
R=results

lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
[ "$lpm" = "0" ] || { echo "ABORT: Low Power Mode is on (lowpowermode=$lpm)"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "ABORT: plug in the charger"; exit 1; }

echo "[$(date)] starting in 60 s - please leave the Mac alone (lid open)"
sleep 60

echo "[$(date)] probes"
sh probe.sh  && mv $R/probe.txt  $R/probe_lpmoff.txt && mv $R/diag_hist.txt $R/diag_hist_lpmoff.txt
sh probe2.sh && mv $R/probe2.txt $R/probe2_lpmoff.txt
python3 pe_speed.py && mv $R/pe_speed.csv $R/pe_speed_lpmoff.csv

echo "[$(date)] benchmark"
python3 run.py run --out timings_lpmoff.csv --skip-done

echo "[$(date)] OVERNIGHT_DONE"
