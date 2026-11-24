/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;
    if (n <= 2) return 0;

    /* Static block partitioning would leave the slowest (largest-i) chunk
     * on one thread. Schedule with small chunks so the load balancer can
     * spread the increasingly expensive large-i work across all threads.
     * Each iteration is independent so reduction is exact (integer sum). */
    #pragma omp parallel for schedule(dynamic, 4096) reduction(+:count)
    for (long i = 2; i < n; i++) {
        int prime = 1;
        long lim = (long)__builtin_sqrt((double)i);
        for (long d = 2; d <= lim; d++)
            if (i % d == 0) { prime = 0; break; }
        count += prime;
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
