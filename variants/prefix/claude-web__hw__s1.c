/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>
#include <string.h>
#include <stdint.h>

/* === KERNEL BEGIN === */
#define SCAN_CHUNK 16384L          /* 128 KB of longs: stays cache-resident */
#define ST_NONE 0ULL
#define ST_AGG  1ULL
#define ST_INCL 2ULL

static inline void cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

static inline uint64_t pack_flag(long v, uint64_t st) {
    return ((uint64_t)v << 2) | st;
}
static inline long flag_val(uint64_t f) { return (long)((int64_t)f >> 2); }

static void serial_scan(long n, const long *x, long *out) {
    long run = 0;
    for (long i = 0; i < n; i++) {
        run += x[i];
        out[i] = run;
    }
}

static void kernel(long n, int reps, const long *x, long *out) {
    int nt = omp_get_max_threads();
    long nch = (n + SCAN_CHUNK - 1) / SCAN_CHUNK;

    if (nt <= 1 || nch <= 1) {
        for (int r = 0; r < reps; r++) serial_scan(n, x, out);
        return;
    }

    uint64_t *flags = (uint64_t *)xmalloc((size_t)nch * sizeof(uint64_t));
    long counter = 0;

    #pragma omp parallel
    {
        for (int r = 0; r < reps; r++) {
            #pragma omp single
            {
                memset(flags, 0, (size_t)nch * sizeof(uint64_t));
                __atomic_store_n(&counter, 0L, __ATOMIC_RELAXED);
            } /* implicit barrier */

            for (;;) {
                long k = __atomic_fetch_add(&counter, 1L, __ATOMIC_RELAXED);
                if (k >= nch) break;

                long lo = k * SCAN_CHUNK;
                long hi = lo + SCAN_CHUNK;
                if (hi > n) hi = n;

                /* 1) local aggregate (brings chunk of x into cache) */
                long agg = 0;
                for (long i = lo; i < hi; i++) agg += x[i];

                /* 2) publish aggregate, look back for exclusive prefix */
                long excl = 0;
                if (k == 0) {
                    __atomic_store_n(&flags[0], pack_flag(agg, ST_INCL),
                                     __ATOMIC_RELEASE);
                } else {
                    __atomic_store_n(&flags[k], pack_flag(agg, ST_AGG),
                                     __ATOMIC_RELEASE);
                    long j = k - 1;
                    for (;;) {
                        uint64_t f = __atomic_load_n(&flags[j], __ATOMIC_ACQUIRE);
                        uint64_t st = f & 3ULL;
                        if (st == ST_NONE) { cpu_relax(); continue; }
                        excl += flag_val(f);
                        if (st == ST_INCL) break;
                        j--;
                    }
                    __atomic_store_n(&flags[k], pack_flag(excl + agg, ST_INCL),
                                     __ATOMIC_RELEASE);
                }

                /* 3) final scan of the (cached) chunk with the offset */
                long run = excl;
                for (long i = lo; i < hi; i++) {
                    run += x[i];
                    out[i] = run;
                }
            }

            #pragma omp barrier
        }
    }

    free(flags);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 100000000);
    int reps = (int)arg_long(argc, argv, 2, 5);
    long *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(10);
    for (long i = 0; i < n; i++) x[i] = (long)(rng_next() % 100);

    double t0 = now_sec();
    kernel(n, reps, x, out);
    double t1 = now_sec();

    double sum = 0.0;   /* sample every 1000th entry plus the last one */
    for (long i = 0; i < n; i += 1000) sum += (double)out[i];
    sum += (double)out[n - 1];
    report("prefix", t1 - t0, sum);
    free(x); free(out);
    return 0;
}
