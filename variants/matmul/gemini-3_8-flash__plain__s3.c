/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < n; i0 += 8) {
        int imax = i0 + 8 <= n ? i0 + 8 : n;
        for (int i = i0; i < imax; i++) {
            double * restrict Ci = &C[(long)i * n];
            for (int j = 0; j < n; j++) Ci[j] = 0.0;
        }

        if (imax == i0 + 8) {
            double * restrict C0 = &C[(long)(i0 + 0) * n];
            double * restrict C1 = &C[(long)(i0 + 1) * n];
            double * restrict C2 = &C[(long)(i0 + 2) * n];
            double * restrict C3 = &C[(long)(i0 + 3) * n];
            double * restrict C4 = &C[(long)(i0 + 4) * n];
            double * restrict C5 = &C[(long)(i0 + 5) * n];
            double * restrict C6 = &C[(long)(i0 + 6) * n];
            double * restrict C7 = &C[(long)(i0 + 7) * n];

            const double * restrict A0 = &A[(long)(i0 + 0) * n];
            const double * restrict A1 = &A[(long)(i0 + 1) * n];
            const double * restrict A2 = &A[(long)(i0 + 2) * n];
            const double * restrict A3 = &A[(long)(i0 + 3) * n];
            const double * restrict A4 = &A[(long)(i0 + 4) * n];
            const double * restrict A5 = &A[(long)(i0 + 5) * n];
            const double * restrict A6 = &A[(long)(i0 + 6) * n];
            const double * restrict A7 = &A[(long)(i0 + 7) * n];

            for (int k = 0; k < n; k++) {
                double a0 = A0[k], a1 = A1[k], a2 = A2[k], a3 = A3[k];
                double a4 = A4[k], a5 = A5[k], a6 = A6[k], a7 = A7[k];
                const double * restrict Bk = &B[(long)k * n];

                #pragma omp simd
                for (int j = 0; j < n; j++) {
                    double b = Bk[j];
                    C0[j] += a0 * b;
                    C1[j] += a1 * b;
                    C2[j] += a2 * b;
                    C3[j] += a3 * b;
                    C4[j] += a4 * b;
                    C5[j] += a5 * b;
                    C6[j] += a6 * b;
                    C7[j] += a7 * b;
                }
            }
        } else {
            for (int i = i0; i < imax; i++) {
                double * restrict Ci = &C[(long)i * n];
                const double * restrict Ai = &A[(long)i * n];
                for (int k = 0; k < n; k++) {
                    double a = Ai[k];
                    const double * restrict Bk = &B[(long)k * n];

                    #pragma omp simd
                    for (int j = 0; j < n; j++) {
                        Ci[j] += a * Bk[j];
                    }
                }
            }
        }
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
