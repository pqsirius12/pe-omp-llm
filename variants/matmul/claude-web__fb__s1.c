/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define NR 8     /* columns per packed B panel (2 x v4d)            */
#define KC 256   /* k-block: one panel slice = KC*NR*8 B = 16 KB    */
#define MB 32    /* rows per task tile                              */
#define NB 256   /* columns per task tile (multiple of NR)          */

typedef double v4d __attribute__((vector_size(32), aligned(8)));
#define LDV(p)    (*(const v4d *)(p))
#define STV(p, v) (*(v4d *)(p) = (v))
#define UNROLL _Pragma("clang loop unroll(full)")

/* Micro-kernel: MRV rows x 8 cols of C kept in registers over kc steps.
 * Per element the order is identical to serial: 0.0, then += a*b, k ascending. */
#define DEF_MICRO(NAME, MRV, ATTR)                                           \
static inline ATTR void NAME(int kc, const double *restrict a, long lda,     \
                             const double *restrict bp,                      \
                             double *restrict c, long ldc, int first) {      \
    v4d acc[MRV][2];                                                         \
    if (first) {                                                             \
        UNROLL for (int r = 0; r < MRV; r++) {                               \
            acc[r][0] = (v4d){0.0, 0.0, 0.0, 0.0};                           \
            acc[r][1] = (v4d){0.0, 0.0, 0.0, 0.0};                           \
        }                                                                    \
    } else {                                                                 \
        UNROLL for (int r = 0; r < MRV; r++) {                               \
            acc[r][0] = LDV(c + r * ldc);                                    \
            acc[r][1] = LDV(c + r * ldc + 4);                                \
        }                                                                    \
    }                                                                        \
    for (int k = 0; k < kc; k++) {                                           \
        v4d b0 = LDV(bp + (long)k * NR);                                     \
        v4d b1 = LDV(bp + (long)k * NR + 4);                                 \
        UNROLL for (int r = 0; r < MRV; r++) {                               \
            double s = a[r * lda + k];                                       \
            v4d av = {s, s, s, s};                                           \
            acc[r][0] += av * b0;                                            \
            acc[r][1] += av * b1;                                            \
        }                                                                    \
    }                                                                        \
    UNROLL for (int r = 0; r < MRV; r++) {                                   \
        STV(c + r * ldc, acc[r][0]);                                         \
        STV(c + r * ldc + 4, acc[r][1]);                                     \
    }                                                                        \
}

/* One task tile: rows [i0,imax), columns [j0,jmax). */
#define DEF_TILE(NAME, MF, M1, MRV, ATTR)                                    \
static ATTR __attribute__((noinline)) void NAME(int n, const double *A,      \
        const double *Bp, double *C, int i0, int imax, int j0, int jmax) {   \
    for (int k0 = 0; k0 < n; k0 += KC) {                                     \
        int kc = (n - k0 < KC) ? n - k0 : KC;                                \
        int first = (k0 == 0);                                               \
        for (int jp = j0; jp < jmax; jp += NR) {                             \
            int nc = (jmax - jp < NR) ? jmax - jp : NR;                      \
            const double *bp = Bp + (long)(jp / NR) * n * NR + (long)k0 * NR;\
            int i = i0;                                                      \
            if (nc == NR) {                                                  \
                for (; i + MRV <= imax; i += MRV)                            \
                    MF(kc, A + (long)i * n + k0, n, bp,                      \
                       C + (long)i * n + jp, n, first);                      \
                for (; i < imax; i++)                                        \
                    M1(kc, A + (long)i * n + k0, n, bp,                      \
                       C + (long)i * n + jp, n, first);                      \
            } else {                                                         \
                double tmp[NR];                                              \
                for (; i < imax; i++) {                                      \
                    double *cr = C + (long)i * n + jp;                       \
                    for (int c = 0; c < NR; c++)                             \
                        tmp[c] = (c < nc && !first) ? cr[c] : 0.0;           \
                    M1(kc, A + (long)i * n + k0, n, bp, tmp, NR, first);     \
                    for (int c = 0; c < nc; c++) cr[c] = tmp[c];             \
                }                                                            \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

/* Baseline build (SSE2 on x86, NEON on AArch64). */
#if defined(__aarch64__)
#define MR_BASE 4
#else
#define MR_BASE 2
#endif
DEF_MICRO(micro_base_full, MR_BASE, )
DEF_MICRO(micro_base_one, 1, )
DEF_TILE(tile_base, micro_base_full, micro_base_one, MR_BASE, )

/* AVX build on x86 (no FMA enabled -> same rounding as the serial build). */
#if defined(__x86_64__) || defined(__i386__)
#define HAVE_AVX 1
DEF_MICRO(micro_avx_full, 4, __attribute__((target("avx"))))
DEF_MICRO(micro_avx_one, 1, __attribute__((target("avx"))))
DEF_TILE(tile_avx, micro_avx_full, micro_avx_one, 4, __attribute__((target("avx"))))
#else
#define HAVE_AVX 0
#endif

static void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    int npan = (n + NR - 1) / NR;
    double *Bp = xmalloc((size_t)npan * (size_t)n * NR * sizeof(double));

    int use_avx = 0;
#if HAVE_AVX
    __builtin_cpu_init();
    use_avx = __builtin_cpu_supports("avx");
#endif

    int nib = (n + MB - 1) / MB;
    int njb = (n + NB - 1) / NB;

    #pragma omp parallel
    {
        /* Pack B into column panels: Bp[p][k][0..NR-1], zero-padded. */
        #pragma omp for schedule(static)
        for (int k = 0; k < n; k++) {
            const double *br = B + (long)k * n;
            for (int p = 0; p < npan; p++) {
                double *dst = Bp + ((long)p * n + k) * NR;
                int j0 = p * NR;
                if (j0 + NR <= n) {
                    for (int c = 0; c < NR; c++) dst[c] = br[j0 + c];
                } else {
                    for (int c = 0; c < NR; c++)
                        dst[c] = (j0 + c < n) ? br[j0 + c] : 0.0;
                }
            }
        }
        /* implicit barrier: Bp complete */

        #pragma omp for collapse(2) schedule(static)
        for (int ib = 0; ib < nib; ib++) {
            for (int jb = 0; jb < njb; jb++) {
                int i0 = ib * MB, j0 = jb * NB;
                int imax = (i0 + MB < n) ? i0 + MB : n;
                int jmax = (j0 + NB < n) ? j0 + NB : n;
#if HAVE_AVX
                if (use_avx) tile_avx(n, A, Bp, C, i0, imax, j0, jmax);
                else         tile_base(n, A, Bp, C, i0, imax, j0, jmax);
#else
                (void)use_avx;
                tile_base(n, A, Bp, C, i0, imax, j0, jmax);
#endif
            }
        }
    }
    free(Bp);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int n = (int)arg_long(argc, argv, 1, 1024);
    long nn = (long)n * n;
    double *A = xmalloc(nn * sizeof *A), *B = xmalloc(nn * sizeof *B);
    double *C = xmalloc(nn * sizeof *C);
    rng_seed(2);
    for (long i = 0; i < nn; i++) { A[i] = rng_unit(); B[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, A, B, C);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < nn; i++) sum += C[i];
    report("matmul", t1 - t0, sum);
    free(A); free(B); free(C);
    return 0;
}
