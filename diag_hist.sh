#!/bin/sh
# diag_hist.sh - find out why the histogram expert version is wrong on the M4.
# Each line compiles the SAME source a different way and runs it once.
# The serial reference checksum (correct answer) is 5.140274176600e+11 for
# the full size and 5.140413612000e+09 for the small size.
cd "$(dirname "$0")"
P=$(brew --prefix libomp)
CC="clang -ffp-contract=off -Xpreprocessor -fopenmp -I$P/include -Ikernels"
LIB="-L$P/lib -lomp -lm"
SRC=variants/histogram/expert.c
G=variants/histogram/gemma-4-31b-it__hw__s1.c
mkdir -p build/diag
run() {   # label, flags, source, env assignments..., args
  label=$1; flags=$2; src=$3; shift 3
  $CC $flags "$src" -o build/diag/x $LIB 2>&1 | head -2
  printf '%-44s ' "$label"; env "$@" ./build/diag/x $ARGS | sed 's/.*checksum=//'
}
echo "== full size (200M x 20); correct = 5.140274176600e+11"
ARGS="200000000 20"
run "expert -O2 (as benchmarked), 1 thread"   "-O2" $SRC OMP_NUM_THREADS=1 OMP_SCHEDULE=static
run "expert -O2, no auto-vectorization"       "-O2 -fno-vectorize -fno-slp-vectorize" $SRC OMP_NUM_THREADS=1 OMP_SCHEDULE=static
run "expert -O0 (no optimization)"            "-O0" $SRC OMP_NUM_THREADS=1 OMP_SCHEDULE=static
run "expert -O2, 4 threads"                   "-O2" $SRC OMP_NUM_THREADS=4 OMP_SCHEDULE=static
run "expert -O2, 4 thr, forced critical reduce" "-O2" $SRC OMP_NUM_THREADS=4 OMP_SCHEDULE=static KMP_FORCE_REDUCTION=critical
run "gemma hw_s1 -O2, 1 thread"               "-O2" $G OMP_NUM_THREADS=1
run "gemma hw_s1 -O2, no auto-vectorization"  "-O2 -fno-vectorize -fno-slp-vectorize" $G OMP_NUM_THREADS=1
echo "== small size (20M x 2); correct = 5.140413612000e+09"
ARGS="20000000 2"
run "expert -O2, 1 thread"                    "-O2" $SRC OMP_NUM_THREADS=1 OMP_SCHEDULE=static
run "expert -O2, no auto-vectorization"       "-O2 -fno-vectorize -fno-slp-vectorize" $SRC OMP_NUM_THREADS=1 OMP_SCHEDULE=static
clang --version | head -1
