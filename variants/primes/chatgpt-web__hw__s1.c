/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;

    #pragma omp parallel
    {
        long local_count = 0;

        #pragma omp for schedule(guided, 2048)
        for (long i = 2; i < n; i++) {
            int prime = 1;
            for (long d = 2; d * d <= i; d++) {
                if (i % d == 0) {
                    prime = 0;
                    break;
                }
            }
            local_count += prime;
        }

        #pragma omp atomic
        count += local_count;
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
