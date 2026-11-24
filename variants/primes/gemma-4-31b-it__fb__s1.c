/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;
    if (n <= 3) return 1;
    
    // Start count at 2 to account for primes 2 and 3.
    long count = 2;
    
    /* 
     * Trial division is irregular: primes take the most time (O(sqrt(i))), 
     * while most composites are eliminated quickly. The cost increases as i increases.
     * 'guided' scheduling is ideal as it starts with larger chunks and decreases
     * them, ensuring load balance among threads while minimizing synchronization.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 5; i < n; i += 2) {
        // Skip multiples of 3 to further reduce the number of trial divisions.
        if (i % 3 == 0) continue;
        
        int prime = 1;
        /* 
         * Wheel Factorization (Wheel of 6):
         * All primes greater than 3 are of the form 6k +/- 1.
         * We only need to check divisors d = 5, 7, 11, 13, 17, 19...
         * This reduces the inner loop iterations by 1/3 compared to checking all odd numbers.
         */
        for (long d = 5; d * d <= i; d += 6) {
            if (i % d == 0 || i % (d + 2) == 0) {
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
