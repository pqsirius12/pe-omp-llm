/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n <= 2) return 0;

    long count = 1;  /* Prime 2 */

    #pragma omp parallel for reduction(+:count) schedule(guided, 64)
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
