#!/bin/sh
# probe.sh - build probe.c and run it for 1..10 threads, 3 rounds each.
# Also re-runs the histogram diagnostic, saving its output this time.
# Output: results/probe.txt and results/diag_hist.txt. About 3 minutes; Mac idle.
cd "$(dirname "$0")"
mkdir -p results build
sh diag_hist.sh > results/diag_hist.txt 2>&1
echo "histogram diagnostic saved to results/diag_hist.txt"
P=$(brew --prefix libomp)
clang -O2 -Xpreprocessor -fopenmp -I"$P/include" probe.c -o build/probe -L"$P/lib" -lomp || exit 1
{
  sysctl -n machdep.cpu.brand_string
  for round in 1 2 3; do
    for t in 1 2 3 4 5 6 7 8 9 10; do
      OMP_NUM_THREADS=$t ./build/probe
      sleep 1
    done
  done
  echo "--- same, forced to background QoS (E-cores only):"
  for t in 1 6; do OMP_NUM_THREADS=$t taskpolicy -b ./build/probe; done
} | tee results/probe.txt
