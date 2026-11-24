#!/bin/sh
# probe2.sh - fixed-work probe under three QoS classes, 1/2/4/6/8/10 threads,
# 3 rounds. About 2 minutes; Mac idle. Output: results/probe2.txt
cd "$(dirname "$0")"
mkdir -p results build
P=$(brew --prefix libomp)
{
  clang -O2 -Xpreprocessor -fopenmp -I"$P/include" probe2.c -o build/probe2 -L"$P/lib" -lomp 2>&1 \
    || { echo "COMPILE FAILED (see errors above)"; exit 1; }
  for round in 1 2 3; do
    for mode in none ui util; do
      for t in 1 2 4 6 8 10; do
        OMP_NUM_THREADS=$t ./build/probe2 $mode
        sleep 1
      done
    done
  done
} | tee results/probe2.txt
