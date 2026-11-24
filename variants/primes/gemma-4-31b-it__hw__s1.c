/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"

/* === KERNEL BEGIN === */
#include <omp.h>

static long kernel(long n) {
    if (n <= 2) return 0;
    
    // We handle 2 separately to allow the loop to skip all even numbers,
    // which significantly reduces the workload (approx. 4x speedup).
    long count = 1; 

    /* 
     * The workload is highly irregular:
     * 1. The cost of trial division increases as i increases (O(sqrt(i))).
     * 2. Prime numbers take significantly longer to verify than composite numbers.
     * 3. The target machine (Apple M4) has asymmetric cores (P-cores and E-cores).
     *
     * 'schedule(guided)' is chosen because it starts with large chunks and decreases 
     * them over time. This effectively balances the increasing cost of larger 'i' 
     * and allows faster P-cores to take on more work than slower E-cores.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 3; i < n; i += 2) {
        int prime = 1;
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

