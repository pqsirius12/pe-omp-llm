/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;

    /* Start count at 1 to include the prime number 2. */
    long count = 1;

    /* 
     * We parallelize the outer loop starting from 3, skipping even numbers.
     * 'reduction(+:count)' safely aggregates the prime counts from all threads.
     * 'schedule(dynamic, 64)' is used instead of static or guided because:
     * 1. Workload is irregular: primes take much longer to verify than composites.
     * 2. Workload increases: the cost of verifying i increases as i grows.
     * 3. Heterogeneous Cores: The Apple M4 has P-cores and E-cores. Dynamic 
     *    scheduling ensures that faster P-cores process more chunks of work 
     *    than the slower E-cores, preventing the slower cores from becoming 
     *    the bottleneck.
     */
    #pragma omp parallel for reduction(+:count) schedule(dynamic, 64)
    for (long i = 3; i < n; i += 2) {
        int prime = 1;
        /* Since i is odd, we only need to test odd divisors starting from 3. */
        for (long d = 3; d * d <= i; d += 2) {
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
