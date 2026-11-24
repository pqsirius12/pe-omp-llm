/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define CACHE_BITS 20
#define CACHE_SIZE (1u << CACHE_BITS)
#define CACHE_MASK (CACHE_SIZE - 1u)

static void kernel(long n, int *steps) {
    #pragma omp parallel
    {
        /* Per-thread direct-mapped cache of (value, steps) pairs.
         * Tag 0 means empty (value 0 is never stored). */
        unsigned int *ctag = (unsigned int *)calloc(CACHE_SIZE, sizeof(unsigned int));
        int *cval = (int *)malloc(CACHE_SIZE * sizeof(int));
        if (!ctag || !cval) {
            /* Fall back to no-cache path if allocation fails. */
            #pragma omp for schedule(guided, 64)
            for (long i = 1; i < n; i++) {
                unsigned long v = (unsigned long)i;
                int s = 0;
                while (v != 1) {
                    v = (v & 1) ? 3 * v + 1 : v / 2;
                    s++;
                }
                steps[i] = s;
            }
        } else {
            #pragma omp for schedule(guided, 64)
            for (long i = 1; i < n; i++) {
                unsigned long v = (unsigned long)i;
                int s = 0;
                for (;;) {
                    if (v < CACHE_SIZE) {
                        unsigned int tag = (unsigned int)v;
                        unsigned int idx = tag & CACHE_MASK;
                        if (ctag[idx] == tag) {
                            s += cval[idx];
                            break;
                        }
                        if (v == 1) break; /* 1 has 0 steps */
                    } else if (v == 1) {
                        break;
                    }
                    /* Fuse even steps: divide by 2 while even. */
                    if ((v & 1UL) == 0) {
                        unsigned long w = v >> 1;
                        int k = 1;
                        while ((w & 1UL) == 0) { w >>= 1; k++; }
                        v = w;
                        s += k;
                    } else {
                        v = 3 * v + 1;
                        s += 1;
                    }
                }
                steps[i] = s;
                /* Store final count in cache if start value was small. */
                if ((unsigned long)i < CACHE_SIZE) {
                    unsigned int tag = (unsigned int)i;
                    unsigned int idx = tag & CACHE_MASK;
                    ctag[idx] = tag;
                    cval[idx] = s;
                }
            }
            free(ctag);
            free(cval);
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int *steps = xmalloc(n * sizeof *steps);
    steps[0] = 0;

    double t0 = now_sec();
    kernel(n, steps);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 1; i < n; i++) sum += steps[i];
    report("collatz", t1 - t0, sum);
    free(steps);
    return 0;
}
