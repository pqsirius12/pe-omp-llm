/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;
    if (n < 2) return 0;

    /* Split [2, n) into chunks. Dynamic scheduling balances the
     * increasing cost per candidate. Each thread accumulates locally. */
    long chunk = 4096;
    long nchunks = (n - 2 + chunk - 1) / chunk;

    #pragma omp parallel
    {
        long local = 0;
        #pragma omp for schedule(dynamic, 1) nowait
        for (long c = 0; c < nchunks; c++) {
            long lo = 2 + c * chunk;
            long hi = lo + chunk;
            if (hi > n) hi = n;
            for (long i = lo; i < hi; i++) {
                int prime = 1;
                for (long d = 2; d * d <= i; d++)
                    if (i % d == 0) { prime = 0; break; }
                local += prime;
            }
        }
        #pragma omp atomic
        count += local;
    }
    return count;
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 6000000);

    double t0 = now_sec();
    long c = kernel(n);
    double t1 = now_sec();

    report("primes", t1 - t0, (double)c);
    return 0;
}
