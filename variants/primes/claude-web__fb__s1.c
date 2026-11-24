/* k11_primes: count primes below n by trial division.
 * IRREGULAR: testing i costs up to sqrt(i) divisions for primes but stops
 * early for most composites, so cost rises with i and jumps around. */
#include "common.h"
#include <omp.h>
#include <stdint.h>
#include <string.h>

/* === KERNEL BEGIN === */
/* Largest r with r*r < n (n >= 3). */
static long isqrt_below(long n) {
    long lo = 0, hi = 3037000499L;          /* hi*hi fits in signed 64-bit */
    while (lo < hi) {
        long mid = lo + (hi - lo + 1) / 2;
        if (mid * mid < n) lo = mid; else hi = mid - 1;
    }
    return lo;
}

static long kernel(long n) {
    if (n <= 2) return 0;

    /* Every composite i < n has a prime factor p with p*p <= i < n, i.e. p <= r. */
    long r = isqrt_below(n);

    /* Small serial sieve: odd primes up to r. */
    long nprimes = 0;
    long *pr = NULL;
    if (r >= 3) {
        unsigned char *s = (unsigned char *)calloc((size_t)r + 1, 1);
        if (!s) { fprintf(stderr, "out of memory\n"); exit(1); }
        for (long i = 3; i * i <= r; i += 2)
            if (!s[i])
                for (long j = i * i; j <= r; j += 2 * i) s[j] = 1;
        pr = (long *)xmalloc(sizeof(long) * (size_t)(r / 2 + 1));
        for (long i = 3; i <= r; i += 2)
            if (!s[i]) pr[nprimes++] = i;
        free(s);
    }

    /* Odd values 2j+1 < n  <=>  0 <= j < J.  j = 0 is the value 1 (not prime). */
    const long J = n / 2;
    const long SEG = 16384;                  /* bytes per segment: fits in L1 */
    const long nseg = (J + SEG - 1) / SEG;

    long count = 1;                          /* the prime 2 */

    #pragma omp parallel reduction(+:count)
    {
        unsigned char *buf = (unsigned char *)xmalloc((size_t)SEG);

        #pragma omp for schedule(dynamic, 1)
        for (long sgi = 0; sgi < nseg; sgi++) {
            long lo = sgi * SEG;
            long hi = lo + SEG;
            if (hi > J) hi = J;
            long len = hi - lo;

            memset(buf, 1, (size_t)len);

            long L = 2 * lo + 1;             /* smallest value in segment */
            long maxv = 2 * hi - 1;          /* largest value in segment */
            for (long k = 0; k < nprimes; k++) {
                long p = pr[k];
                long pp = p * p;
                if (pp > maxv) break;
                long m;
                if (pp >= L) {
                    m = pp;
                } else {
                    m = ((L + p - 1) / p) * p;
                    if (!(m & 1)) m += p;    /* first odd multiple >= L */
                }
                for (long j = (m - 1) / 2 - lo; j < len; j += p)
                    buf[j] = 0;
            }
            if (lo == 0) buf[0] = 0;         /* 1 is not prime */

            long c = 0;
            for (long j = 0; j < len; j++) c += buf[j];
            count += c;
        }

        free(buf);
    }

    free(pr);
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
