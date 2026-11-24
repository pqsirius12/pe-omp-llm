/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define CK 10
#define CMASK ((1u << CK) - 1u)

typedef struct { uint32_t m, a, s, pad; } ctab_t;
static ctab_t ctab[1u << CK];

static void kernel(long n, int *steps) {
    if (n <= 1) return;

    /* Build k-step table: for v = a*2^K + b, after K shortcut steps
     * T(x) = x even ? x/2 : (3x+1)/2, v becomes a*3^c + T^K(b),
     * which corresponds to K + c ordinary Collatz steps. */
    for (uint32_t b = 0; b <= CMASK; b++) {
        uint64_t x = b;
        uint32_t c = 0, m = 1;
        for (int k = 0; k < CK; k++) {
            if (x & 1) { x = (3 * x + 1) >> 1; c++; m *= 3; }
            else x >>= 1;
        }
        ctab[b].m = m;
        ctab[b].a = (uint32_t)x;
        ctab[b].s = CK + c;
        ctab[b].pad = 0;
    }

    /* Serial memoized prefix: stop once the trajectory drops below i. */
    long base = n < 4096 ? n : 4096;
    steps[1] = 0;
    for (long i = 2; i < base; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v >= (unsigned long)i) {
            if (v & 1) {
                v = (3 * v + 1) >> 1;
                s += 2;
            } else {
                int t = __builtin_ctzl(v);
                v >>= t;
                s += t;
            }
        }
        steps[i] = s + steps[v];
    }
    if (base >= n) return;

    /* Doubling phases: in phase [lo, hi) all values < lo are known. */
    #pragma omp parallel
    {
        for (long lo = base; lo < n; lo *= 2) {
            long hi = (lo * 2 < n) ? lo * 2 : n;
            const unsigned long ulo = (unsigned long)lo;
            long jlo = lo >> 1;          /* lo is even */
            long jhi = (hi + 1) >> 1;

            #pragma omp for schedule(dynamic, 2048)
            for (long j = jlo; j < jhi; j++) {
                long i = 2 * j;
                /* even: one halving step, j < lo already known */
                steps[i] = steps[j] + 1;

                long o = i + 1;
                if (o < hi) {
                    unsigned long v = (unsigned long)o;
                    int s = 0;
                    /* v >= lo >= 4096 at each block start, so a K-step
                     * block never passes through 1 (min value >= 4). */
                    do {
                        const ctab_t e = ctab[v & CMASK];
                        v = (v >> CK) * e.m + e.a;
                        s += (int)e.s;
                    } while (v >= ulo);
                    steps[o] = s + steps[v];
                }
            }
            /* implicit barrier: phase results visible before next phase */
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
