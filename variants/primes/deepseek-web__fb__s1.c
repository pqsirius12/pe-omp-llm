/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>
#include <string.h>
#include <math.h>

/* === KERNEL BEGIN === */
static long kernel(long n) {
    if (n < 3) return (n >= 2) ? 1 : 0;
    long root = (long)sqrt((double)(n - 1));
    while ((root + 1) * (root + 1) <= n - 1) root++;
    while (root * root > n - 1) root--;

    /* Small primes up to sqrt(n-1) via simple sieve (serial, cheap). */
    char *small = (char *)xmalloc((size_t)(root + 1));
    memset(small, 1, (size_t)(root + 1));
    if (root >= 0) small[0] = 0;
    if (root >= 1) small[1] = 0;
    for (long i = 2; i * i <= root; i++)
        if (small[i])
            for (long j = i * i; j <= root; j += i) small[j] = 0;

    long nsmall = 0;
    for (long i = 2; i <= root; i++) nsmall += small[i];

    /* Segmented sieve over [root+1, n). */
    long seg_lo_base = root + 1;
    long total = n - seg_lo_base;
    if (total < 0) total = 0;
    long segsize = 32768;
    long nsegs = (total + segsize - 1) / segsize;

    long total_primes = nsmall;
    long seg_primes = 0;

    #pragma omp parallel reduction(+:seg_primes)
    {
        char *seg = (char *)xmalloc((size_t)segsize);
        #pragma omp for schedule(static)
        for (long s = 0; s < nsegs; s++) {
            long lo = seg_lo_base + s * segsize;
            long hi = lo + segsize;
            if (hi > n) hi = n;
            long len = hi - lo;
            memset(seg, 1, (size_t)len);
            for (long p = 2; p <= root; p++) {
                if (!small[p]) continue;
                long start = (lo + p - 1) / p * p;
                if (start < p * p) start = p * p;
                for (long j = start; j < hi; j += p) seg[j - lo] = 0;
            }
            long c = 0;
            for (long k = 0; k < len; k++) c += seg[k];
            seg_primes += c;
        }
        free(seg);
    }

    free(small);
    total_primes += seg_primes;
    return total_primes;
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
