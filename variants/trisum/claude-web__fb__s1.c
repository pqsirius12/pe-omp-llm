/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TS_X86 1
#elif defined(__aarch64__)
#include <arm_neon.h>
#define TS_NEON 1
#endif

/* === KERNEL BEGIN === */
#define RB 16

/* Exact serial computation of rows i0 .. i0+cnt-1. */
static void rows_exact(long i0, long cnt, const double *x, double *out) {
    for (long k = 0; k < cnt; k++) {
        long i = i0 + k;
        double s = 0.0;
        for (long j = 0; j <= i; j++)
            s += x[j] / (double)(1 + i - j);
        out[i] = s;
    }
}

/* Row i0+k additionally needs j = i0+1 .. i0+k, in order, exact division. */
static void tails(long i0, const double *x, double *s) {
    for (int k = 1; k < RB; k++) {
        long i = i0 + k;
        double sk = s[k];
        for (long j = i0 + 1; j <= i; j++)
            sk += x[j] / (double)(1 + i - j);
        s[k] = sk;
    }
}

/* Common part j = 0..i0 of rows i0..i0+RB-1. Returns 1 on success,
 * 0 if the exactness proof failed somewhere (caller recomputes exactly). */
typedef int (*block_fn)(long, const double *, const double *, const double *, double *);

/* Generic path: true division (bit-identical by construction). */
static int block_div(long i0, const double *x, const double *r,
                     const double *dt, double *s) {
    (void)r;
    double acc[RB];
    for (int k = 0; k < RB; k++) acc[k] = 0.0;
    for (long j = 0; j <= i0; j++) {
        double xj = x[j];
        const double *dp = dt + (1 + i0 - j);
        #pragma omp simd
        for (int k = 0; k < RB; k++)
            acc[k] += xj / dp[k];
    }
    for (int k = 0; k < RB; k++) s[k] = acc[k];
    return 1;
}

#ifdef TS_X86
/* q1 = correctly rounded a/b via reciprocal + FMA correction;
 * proven exact iff fma(fma(-q1,b,a), y, q1) == q1. */
#define TS_STEP(ACC, OFF) do {                                        \
        __m256d y  = _mm256_loadu_pd(rp + (OFF));                     \
        __m256d b  = _mm256_loadu_pd(dp + (OFF));                     \
        __m256d q  = _mm256_mul_pd(a, y);                             \
        __m256d e  = _mm256_fnmadd_pd(q, b, a);                       \
        __m256d q1 = _mm256_fmadd_pd(e, y, q);                        \
        __m256d e2 = _mm256_fnmadd_pd(q1, b, a);                      \
        __m256d q2 = _mm256_fmadd_pd(e2, y, q1);                      \
        bad = _mm256_or_pd(bad, _mm256_cmp_pd(q2, q1, _CMP_NEQ_UQ));  \
        ACC = _mm256_add_pd(ACC, q1);                                 \
    } while (0)

__attribute__((target("avx2,fma")))
static int block_avx2(long i0, const double *x, const double *r,
                      const double *dt, double *s) {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
    __m256d bad = _mm256_setzero_pd();
    for (long j = 0; j <= i0; j++) {
        __m256d a = _mm256_broadcast_sd(x + j);
        long d = 1 + i0 - j;
        const double *rp = r + d, *dp = dt + d;
        TS_STEP(s0, 0);
        TS_STEP(s1, 4);
        TS_STEP(s2, 8);
        TS_STEP(s3, 12);
    }
    if (!_mm256_testz_pd(bad, bad)) return 0;
    _mm256_storeu_pd(s + 0, s0);
    _mm256_storeu_pd(s + 4, s1);
    _mm256_storeu_pd(s + 8, s2);
    _mm256_storeu_pd(s + 12, s3);
    return 1;
}
#endif

#ifdef TS_NEON
static int block_neon(long i0, const double *x, const double *r,
                      const double *dt, double *s) {
    float64x2_t acc[RB / 2];
    uint64x2_t ok = vdupq_n_u64(~0ULL);
    for (int m = 0; m < RB / 2; m++) acc[m] = vdupq_n_f64(0.0);
    for (long j = 0; j <= i0; j++) {
        float64x2_t a = vdupq_n_f64(x[j]);
        long d = 1 + i0 - j;
        const double *rp = r + d, *dp = dt + d;
        #pragma clang loop unroll(full)
        for (int m = 0; m < RB / 2; m++) {
            float64x2_t y  = vld1q_f64(rp + 2 * m);
            float64x2_t b  = vld1q_f64(dp + 2 * m);
            float64x2_t q  = vmulq_f64(a, y);
            float64x2_t e  = vfmsq_f64(a, q, b);     /* a - q*b, fused */
            float64x2_t q1 = vfmaq_f64(q, e, y);     /* q + e*y, fused */
            float64x2_t e2 = vfmsq_f64(a, q1, b);
            float64x2_t q2 = vfmaq_f64(q1, e2, y);
            ok = vandq_u64(ok, vceqq_f64(q2, q1));
            acc[m] = vaddq_f64(acc[m], q1);
        }
    }
    if ((vgetq_lane_u64(ok, 0) & vgetq_lane_u64(ok, 1)) != ~0ULL) return 0;
    for (int m = 0; m < RB / 2; m++) vst1q_f64(s + 2 * m, acc[m]);
    return 1;
}
#endif

static void kernel(long n, const double *x, double *out) {
    if (n <= 0) return;
    long nb = n / RB;          /* full blocks of RB rows */
    long rem_start = nb * RB;  /* leftover rows handled exactly */

    double *r  = xmalloc((size_t)(n + 1) * sizeof *r);   /* ~1/d   */
    double *dt = xmalloc((size_t)(n + 1) * sizeof *dt);  /* (double)d */

    block_fn fn = block_div;
#ifdef TS_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        fn = block_avx2;
#endif
#ifdef TS_NEON
    fn = block_neon;
#endif

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long d = 0; d <= n; d++) {
            dt[d] = (double)d;
            r[d] = d ? 1.0 / (double)d : 0.0;
        }

        /* Largest blocks first, dynamic for P/E-core balance. */
        #pragma omp for schedule(dynamic, 1) nowait
        for (long bb = 0; bb < nb; bb++) {
            long i0 = (nb - 1 - bb) * RB;
            double s[RB];
            if (fn(i0, x, r, dt, s)) {
                tails(i0, x, s);
                for (int k = 0; k < RB; k++) out[i0 + k] = s[k];
            } else {
                rows_exact(i0, RB, x, out);   /* proof failed: exact redo */
            }
        }

        /* Remaining rows (fewer than RB), each done exactly as serial. */
        #pragma omp for schedule(dynamic, 1)
        for (long i = rem_start; i < n; i++)
            rows_exact(i, 1, x, out);
    }

    free(r);
    free(dt);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 40000);
    double *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(4);
    for (long i = 0; i < n; i++) x[i] = rng_unit();

    double t0 = now_sec();
    kernel(n, x, out);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += out[i];
    report("trisum", t1 - t0, sum);
    free(x); free(out);
    return 0;
}
