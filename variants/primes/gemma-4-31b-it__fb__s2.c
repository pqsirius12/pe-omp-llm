/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;
    
    /* We start with count = 1 to account for the prime number 2.
     * We then only iterate through odd numbers from 3 up to n-1. */
    long count = 1;
    
    /* 
     * The workload is irregular for two reasons:
     * 1. As 'i' increases, the worst-case number of divisions (sqrt(i)) increases.
     * 2. Composite numbers are eliminated quickly, while primes require full trial division.
     * 
     * 'schedule(guided)' is ideal here: it starts with large chunks to reduce overhead 
     * and finishes with smaller chunks to ensure load balancing as the per-iteration 
     * cost increases.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
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
