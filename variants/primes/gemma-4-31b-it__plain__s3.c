/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;
    
    // Start count at 1 to account for the number 2, the only even prime.
    long count = 1;

    /* 
     * We parallelize the outer loop using OpenMP.
     * - reduction(+:count): Accumulates prime counts from each thread safely.
     * - schedule(guided): Since the computational cost of trial division 
     *   increases as 'i' grows, guided scheduling starts with large chunks
     *   and decreases them to maintain load balance across threads.
     * - i += 2: We only iterate over odd numbers to double the performance.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 3; i < n; i += 2) {
        int prime = 1;
        // Only check odd divisors starting from 3.
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
