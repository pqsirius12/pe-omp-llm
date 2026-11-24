/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;
    if (n == 3) return 1;

    /* 
     * To optimize trial division, we precompute all primes up to sqrt(n).
     * Testing divisibility only by primes is significantly faster than
     * testing all odd numbers, while remaining within the logic of trial division.
     */
    long limit = 0;
    while ((limit + 1) * (limit + 1) <= n) limit++;

    // Simple Sieve of Eratosthenes to find primes up to sqrt(n)
    char *is_p = (char *)xmalloc(limit + 1);
    for (long i = 0; i <= limit; i++) is_p[i] = 1;
    is_p[0] = is_p[1] = 0;
    for (long p = 2; p * p <= limit; p++) {
        if (is_p[p]) {
            for (long i = p * p; i <= limit; i += p) is_p[i] = 0;
        }
    }

    // Store only odd primes to avoid checking divisibility by 2 in the parallel loop
    long num_odd_primes = 0;
    for (long i = 3; i <= limit; i++) if (is_p[i]) num_odd_primes++;

    long *primes = (long *)xmalloc(num_odd_primes * sizeof(long));
    long p_idx = 0;
    for (long i = 3; i <= limit; i++) {
        if (is_p[i]) primes[p_idx++] = i;
    }
    free(is_p);

    long count = 1; // Start with 1 to account for the prime number 2

    /* 
     * Parallelize the outer loop.
     * - reduction(+:count): Thread-safe accumulation of the prime count.
     * - schedule(guided): The workload increases as 'i' increases (sqrt(i) checks),
     *   and is irregular based on primality. Guided scheduling balances this load.
     * - i += 2: Skip all even numbers to halve the outer loop iterations.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 3; i < n; i += 2) {
        int prime = 1;
        for (long j = 0; j < num_odd_primes; j++) {
            long p = primes[j];
            if (p * p > i) break;
            if (i % p == 0) {
                prime = 0;
                break;
            }
        }
        count += prime;
    }

    free(primes);
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
