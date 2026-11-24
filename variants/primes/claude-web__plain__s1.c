/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>
#include <stdint.h>

/* === KERNEL BEGIN === */
/* Primality of an odd number i >= 3 using odd divisors only (32-bit path). */
static inline int is_prime_odd32(uint32_t i) {
    for (uint32_t d = 3; (uint64_t)d * d <= i; d += 2)
        if (i % d == 0) return 0;
    return 1;
}

/* Same test with 64-bit arithmetic, for large i. */
static inline int is_prime_odd64(long i) {
    for (long d = 3; d * d <= i; d += 2)
        if (i % d == 0) return 0;
    return 1;
}

static long kernel(long n) {
    if (n <= 2) return 0;
    long count = 1;              /* the prime 2 (n > 2 here) */
    long kmax = n / 2;           /* odd i = 2k+1 with 1 <= k < kmax  <=>  3 <= i < n */

    if (n <= 4294967295L) {
        #pragma omp parallel for schedule(dynamic, 256) reduction(+:count)
        for (long k = 1; k < kmax; k++)
            count += is_prime_odd32((uint32_t)(2 * k + 1));
    } else {
        #pragma omp parallel for schedule(dynamic, 256) reduction(+:count)
        for (long k = 1; k < kmax; k++)
            count += is_prime_odd64(2 * k + 1);
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
