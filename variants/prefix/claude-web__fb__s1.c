/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
/* Single-pass block scan with decoupled look-back + streaming stores.
 *  - Blocks of PS_BLOCK elements are claimed dynamically (atomic ticket), so
 *    threads never wait on a per-tile barrier and fast cores take more work.
 *  - Per block: sum x (brings it into cache), publish the aggregate, look back
 *    over predecessors' published aggregates/inclusive prefixes to get the
 *    exclusive offset, publish the inclusive prefix, then scan the (cache-hot)
 *    block writing out with non-temporal stores (no write-allocate traffic).
 *  - Flags encode the repetition number, so no reset is needed between reps.
 * Integer arithmetic keeps the result identical to the serial version. */
#define PS_BLOCK 16384L   /* elements per block (128 KB of longs) */

typedef long ps_v4 __attribute__((vector_size(32)));

#if defined(__x86_64__) || defined(__i386__)
#define PS_RELAX() __builtin_ia32_pause()
#define PS_FENCE() __builtin_ia32_sfence()
#elif defined(__aarch64__)
#define PS_RELAX() __asm__ __volatile__("yield" ::: "memory")
#define PS_FENCE() __asm__ __volatile__("dmb ish" ::: "memory")
#else
#define PS_RELAX() ((void)0)
#define PS_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif

typedef struct {
    long flag;      /* 4*r+1: agg valid, 4*r+2: incl valid (rep r) */
    long agg;
    long incl;
    long pad[5];    /* one cache line per block descriptor */
} ps_desc;

static inline void ps_scan_block(const long *restrict x, long *restrict out,
                                 long lo, long hi, long run) {
    long i = lo;
    while (i < hi && (((uintptr_t)(out + i)) & 31u)) {
        run += x[i];
        out[i] = run;
        i++;
    }
    for (; i + 4 <= hi; i += 4) {
        long a0 = run + x[i];
        long a1 = a0 + x[i + 1];
        long a2 = a1 + x[i + 2];
        long a3 = a2 + x[i + 3];
        run = a3;
        ps_v4 v = {a0, a1, a2, a3};
        __builtin_nontemporal_store(v, (ps_v4 *)(out + i));
    }
    for (; i < hi; i++) {
        run += x[i];
        out[i] = run;
    }
}

static void kernel(long n, int reps, const long *x, long *out) {
    int maxt = omp_get_max_threads();
    if (maxt <= 1 || n < 4 * PS_BLOCK) {
        for (int r = 0; r < reps; r++)
            ps_scan_block(x, out, 0, n, 0);
        PS_FENCE();
        return;
    }

    const long nb = (n + PS_BLOCK - 1) / PS_BLOCK;
    ps_desc *desc = calloc((size_t)nb, sizeof *desc);
    long *ctrs = calloc((size_t)(reps > 0 ? reps : 1) * 8, sizeof *ctrs);
    if (!desc || !ctrs) { fprintf(stderr, "out of memory\n"); exit(1); }

    #pragma omp parallel
    {
        for (int r = 0; r < reps; r++) {
            long *ctr = &ctrs[(size_t)r * 8];
            const long st_agg = 4L * r + 1;
            const long st_inc = 4L * r + 2;

            for (;;) {
                long b = __atomic_fetch_add(ctr, 1L, __ATOMIC_RELAXED);
                if (b >= nb) break;
                long lo = b * PS_BLOCK;
                long hi = lo + PS_BLOCK;
                if (hi > n) hi = n;

                long s = 0;
                for (long i = lo; i < hi; i++) s += x[i];

                long excl = 0;
                if (b == 0) {
                    desc[0].incl = s;
                    __atomic_store_n(&desc[0].flag, st_inc, __ATOMIC_RELEASE);
                } else {
                    desc[b].agg = s;
                    __atomic_store_n(&desc[b].flag, st_agg, __ATOMIC_RELEASE);
                    long j = b - 1;
                    for (;;) {
                        long f = __atomic_load_n(&desc[j].flag, __ATOMIC_ACQUIRE);
                        if (f == st_inc) { excl += desc[j].incl; break; }
                        if (f == st_agg) { excl += desc[j].agg; j--; continue; }
                        PS_RELAX();
                    }
                    desc[b].incl = excl + s;
                    __atomic_store_n(&desc[b].flag, st_inc, __ATOMIC_RELEASE);
                }

                ps_scan_block(x, out, lo, hi, excl);
            }
            PS_FENCE();          /* order streaming stores before the barrier */
            #pragma omp barrier  /* keep reps cleanly separated */
        }
    }

    free(ctrs);
    free(desc);
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
