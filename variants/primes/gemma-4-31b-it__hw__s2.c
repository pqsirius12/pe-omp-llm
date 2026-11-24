/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    long count = 0;
    /* 
     * The workload is irregular for two reasons:
     * 1. Trial division takes longer for larger i (up to sqrt(i)).
     * 2. Primes take significantly longer to verify than composite numbers.
     * 3. The target hardware has asymmetric cores (P-cores and E-cores).
     * 
     * 'schedule(guided)' is ideal here because it starts with large chunks to 
     * reduce overhead and decreases chunk size to 1 as it progresses. 
     * This balances the increasing workload of larger i and allows faster 
     * P-cores to pick up more work than slower E-cores.
     */
    #pragma omp parallel for reduction(+:count) schedule(guided)
    for (long i = 2; i < n; i++) {
        int prime = 1;
        for (long d = 2; d * d <= i; d++)
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
