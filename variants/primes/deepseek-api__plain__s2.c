/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;
    if (n < 3) return 0;
    long limit = n - 1;  /* test i in [2, n-1] */

    /* Handle i=2 (the only even prime) separately. */
    if (limit >= 2) count = 1;

#pragma omp parallel
    {
        long local = 0;
#pragma omp for schedule(dynamic, 64)
        for (long i = 3; i <= limit; i += 2) {
            int prime = 1;
            for (long d = 3; d * d <= i; d += 2)
                if (i % d == 0) { prime = 0; break; }
            local += prime;
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
