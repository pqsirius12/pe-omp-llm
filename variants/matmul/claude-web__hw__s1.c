/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* === KERNEL BEGIN === */
/*
 * Parallel, cache-blocked matmul that is bit-identical to the serial loop.
 *
 * Every C[i][j] starts at 0.0 and gets a*b added for k = 0, 1, ..., n-1 in
 * increasing order, each step as one fused multiply-add. That is exactly what
 * the serial code computes; clang contracts "C += a*B" into an FMA by default.
 * Blocking over i and j and splitting k into consecutive blocks (with C stored
 * and reloaded between them) never reorders the additions to any element.
 *
 * Work is cut into many (MB rows x NB cols) tiles handed out dynamically, so
 * the faster P-cores take more tiles than the E-cores and none sit idle.
 */
#define MB 32    /* rows per tile    */
#define NB 128   /* cols per tile    */
#define KB 256   /* k-block (packed) */

/* 4x8 register-blocked micro-kernel over kc consecutive k values.
 * bp is a packed B panel laid out as [k][8]. */
static inline void micro4x8(int kc, const double *A, long lda,
                            const double *bp, double *C, long ldc) {
#ifdef __aarch64__
    float64x2_t c00 = vld1q_f64(C + 0*ldc + 0), c01 = vld1q_f64(C + 0*ldc + 2),
                c02 = vld1q_f64(C + 0*ldc + 4), c03 = vld1q_f64(C + 0*ldc + 6);
    float64x2_t c10 = vld1q_f64(C + 1*ldc + 0), c11 = vld1q_f64(C + 1*ldc + 2),
                c12 = vld1q_f64(C + 1*ldc + 4), c13 = vld1q_f64(C + 1*ldc + 6);
    float64x2_t c20 = vld1q_f64(C + 2*ldc + 0), c21 = vld1q_f64(C + 2*ldc + 2),
                c22 = vld1q_f64(C + 2*ldc + 4), c23 = vld1q_f64(C + 2*ldc + 6);
    float64x2_t c30 = vld1q_f64(C + 3*ldc + 0), c31 = vld1q_f64(C + 3*ldc + 2),
                c32 = vld1q_f64(C + 3*ldc + 4), c33 = vld1q_f64(C + 3*ldc + 6);
    const double *a0p = A, *a1p = A + lda, *a2p = A + 2*lda, *a3p = A + 3*lda;
    for (int k = 0; k < kc; k++) {
        const double *b = bp + (long)k * 8;
        float64x2_t b0 = vld1q_f64(b), b1 = vld1q_f64(b + 2),
                    b2 = vld1q_f64(b + 4), b3 = vld1q_f64(b + 6);
        double a0 = a0p[k], a1 = a1p[k], a2 = a2p[k], a3 = a3p[k];
        c00 = vfmaq_n_f64(c00, b0, a0); c01 = vfmaq_n_f64(c01, b1, a0);
        c02 = vfmaq_n_f64(c02, b2, a0); c03 = vfmaq_n_f64(c03, b3, a0);
        c10 = vfmaq_n_f64(c10, b0, a1); c11 = vfmaq_n_f64(c11, b1, a1);
        c12 = vfmaq_n_f64(c12, b2, a1); c13 = vfmaq_n_f64(c13, b3, a1);
        c20 = vfmaq_n_f64(c20, b0, a2); c21 = vfmaq_n_f64(c21, b1, a2);
        c22 = vfmaq_n_f64(c22, b2, a2); c23 = vfmaq_n_f64(c23, b3, a2);
        c30 = vfmaq_n_f64(c30, b0, a3); c31 = vfmaq_n_f64(c31, b1, a3);
        c32 = vfmaq_n_f64(c32, b2, a3); c33 = vfmaq_n_f64(c33, b3, a3);
    }
    vst1q_f64(C + 0*ldc + 0, c00); vst1q_f64(C + 0*ldc + 2, c01);
    vst1q_f64(C + 0*ldc + 4, c02); vst1q_f64(C + 0*ldc + 6, c03);
    vst1q_f64(C + 1*ldc + 0, c10); vst1q_f64(C + 1*ldc + 2, c11);
    vst1q_f64(C + 1*ldc + 4, c12); vst1q_f64(C + 1*ldc + 6, c13);
    vst1q_f64(C + 2*ldc + 0, c20); vst1q_f64(C + 2*ldc + 2, c21);
    vst1q_f64(C + 2*ldc + 4, c22); vst1q_f64(C + 2*ldc + 6, c23);
    vst1q_f64(C + 3*ldc + 0, c30); vst1q_f64(C + 3*ldc + 2, c31);
    vst1q_f64(C + 3*ldc + 4, c32); vst1q_f64(C + 3*ldc + 6, c33);
#else
    double acc[4][8];
    for (int r = 0; r < 4; r++)
        for (int jj = 0; jj < 8; jj++) acc[r][jj] = C[r * ldc + jj];
    for (int k = 0; k < kc; k++)
        for (int r = 0; r < 4; r++) {
            double a = A[r * lda + k];
            for (int jj = 0; jj < 8; jj++)
                acc[r][jj] += a * bp[(long)k * 8 + jj];
        }
    for (int r = 0; r < 4; r++)
        for (int jj = 0; jj < 8; jj++) C[r * ldc + jj] = acc[r][jj];
#endif
}

/* Generic edge case: mr rows (<=4) x w cols (<=8), same k order. */
static inline void micro_edge(int mr, int w, int kc, const double *A, long lda,
                              const double *bp, double *C, long ldc) {
    for (int r = 0; r < mr; r++)
        for (int jj = 0; jj < w; jj++) {
            double c = C[r * ldc + jj];
            for (int k = 0; k < kc; k++)
                c += A[r * lda + k] * bp[(long)k * 8 + jj];
            C[r * ldc + jj] = c;
        }
}

static void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    const long ln = n;
    const int nti = (n + MB - 1) / MB;
    const int ntj = (n + NB - 1) / NB;
    const long ntiles = (long)nti * ntj;

    #pragma omp parallel
    {
        double *bp = (double *)aligned_alloc(64, (size_t)KB * NB * sizeof(double));
        if (!bp) { fprintf(stderr, "out of memory\n"); exit(1); }

        #pragma omp for schedule(dynamic, 1)
        for (long t = 0; t < ntiles; t++) {
            int i0 = (int)(t / ntj) * MB, j0 = (int)(t % ntj) * NB;
            int i1 = i0 + MB < n ? i0 + MB : n;
            int j1 = j0 + NB < n ? j0 + NB : n;
            int nc = j1 - j0;

            for (int i = i0; i < i1; i++)
                for (int j = j0; j < j1; j++) C[i * ln + j] = 0.0;

            for (int k0 = 0; k0 < n; k0 += KB) {
                int kc = k0 + KB < n ? KB : n - k0;

                /* Pack B[k0:k0+kc, j0:j1] into 8-wide panels [p][k][8]. */
                for (int k = 0; k < kc; k++) {
                    const double *src = B + (long)(k0 + k) * ln + j0;
                    for (int p = 0; p < nc; p += 8) {
                        double *dst = bp + (long)(p / 8) * kc * 8 + (long)k * 8;
                        int w = nc - p < 8 ? nc - p : 8;
                        int jj = 0;
                        for (; jj < w; jj++) dst[jj] = src[p + jj];
                        for (; jj < 8; jj++) dst[jj] = 0.0;
                    }
                }

                for (int p = 0; p < nc; p += 8) {
                    int w = nc - p < 8 ? nc - p : 8;
                    const double *bpp = bp + (long)(p / 8) * kc * 8;
                    int i = i0;
                    if (w == 8) {
                        for (; i + 4 <= i1; i += 4)
                            micro4x8(kc, A + i * ln + k0, ln, bpp,
                                     C + i * ln + j0 + p, ln);
                    }
                    for (; i < i1; i += 4) {
                        int mr = i1 - i < 4 ? i1 - i : 4;
                        micro_edge(mr, w, kc, A + i * ln + k0, ln, bpp,
                                   C + i * ln + j0 + p, ln);
                    }
                }
            }
        }
        free(bp);
    }
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
