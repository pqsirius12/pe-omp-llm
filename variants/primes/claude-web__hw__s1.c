/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>
#include <stdint.h>

/* === KERNEL BEGIN === */

/* Trial division for odd u >= 3 using only odd divisors (32-bit path). */
static inline int is_prime_odd32(uint32_t u) {
    for (uint32_t d = 3; (uint64_t)d * d <= u; d += 2)
        if (u % d == 0) return 0;
    return 1;
}

/* Same, 64-bit path for large n. */
static inline int is_prime_odd64(long i) {
    for (long d = 3; d * d <= i; d += 2)
        if (i % d == 0) return 0;
    return 1;
}

static long kernel(long n) {
    if (n <= 2) return 0;

    long count = 1;              /* the prime 2 (n > 2 here) */
    long m = (n - 2) / 2;        /* number of odd i in [3, n-1] */

    /* Iterate from the largest (most expensive) candidates down to the
     * smallest, so the cheap work fills in at the end and the P-/E-core
     * imbalance shrinks. Dynamic scheduling lets faster cores take more. */
    if (n - 1 <= (long)UINT32_MAX) {
        #pragma omp parallel for schedule(dynamic, 512) reduction(+:count)
        for (long j = 0; j < m; j++) {
            uint32_t i = (uint32_t)(3 + 2 * (m - 1 - j));
            count += is_prime_odd32(i);
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 512) reduction(+:count)
        for (long j = 0; j < m; j++) {
            long i = 3 + 2 * (m - 1 - j);
            count += is_prime_odd64(i);
        }
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
