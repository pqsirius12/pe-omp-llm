/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;

    /* Small trial-division prefilter to speed up per-candidate work:
       checking a few small primes first lets us skip most composites fast
       and shrinks the inner loop trip count for the survivors. */
    static const int sprimes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                                   37, 41, 43, 47, 53, 59, 61, 67, 71, 73};
    const int nspr = (int)(sizeof(sprimes) / sizeof(sprimes[0]));

    long total = 0;

    #pragma omp parallel
    {
        long local = 0;

        /* Static schedule with big chunks: monotone work per i means
           contiguous ranges cost roughly the same, so static balancing is
           good and avoids atomic overhead. Give each thread one large
           contiguous block; the tail-most (heaviest) region may go to one
           thread — mitigate with a small over-decomposition via chunksize. */
        long chunk = (n - 2) / ((long)omp_get_num_threads() * 8) + 1;

        #pragma omp for schedule(static, chunk) nowait
        for (long i = 2; i < n; i++) {
            int prime = 1;
            /* Handle 2 separately: mark prime only for even i==2. */
            if ((i & 1) == 0) {
                if (i == 2) { local += 1; }
                continue;
            }
            /* small-prime prefilter */
            for (int k = 0; k < nspr; k++) {
                int p = sprimes[k];
                if (p * (long)p > i) goto is_prime;
                if (i % p == 0) { primality_check_fail: prime = 0; break; }
            }
            if (prime) {
                /* full trial division by odd divisors beyond the prefilter */
                for (long d = 79; d * d <= i; d += 2) {
                    if (i % d == 0) { prime = 0; break; }
                }
            }
        is_prime:
            local += prime;
            (void)0;
        }

        #pragma omp atomic
        total += local;
    }

    return total + 1; /* +1 for the prime 2 */
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
