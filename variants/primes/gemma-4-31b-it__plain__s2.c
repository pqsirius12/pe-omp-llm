/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;
    /* 
     * The workload is irregular because:
     * 1. The cost of checking if 'i' is prime increases as i increases (up to sqrt(i)).
     * 2. Composites with small factors are found very quickly, while primes take longer.
     * 
     * 'schedule(guided)' is used because it handles workloads that are both
     * monotonically increasing in cost and irregular, reducing synchronization 
     * overhead compared to 'schedule(dynamic, 1)'.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 2; i < n; i++) {
        int prime = 1;
        for (long d = 2; d * d <= i; d++) {
            if (i % d == 0) {
                prime = 0;
                break;
            }
        }
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
