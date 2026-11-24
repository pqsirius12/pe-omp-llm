/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <string.h>
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    int num_blocks = (n + 7) / 8;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < num_blocks; b++) {
        int i0 = b * 8;
        int imax = (i0 + 8 <= n) ? i0 + 8 : n;
        int rem = imax - i0;

        if (rem == 8) {
            long i0_n = (long)i0 * n;
            double * restrict c0 = &C[i0_n];
            double * restrict c1 = c0 + n;
            double * restrict c2 = c1 + n;
            double * restrict c3 = c2 + n;
            double * restrict c4 = c3 + n;
            double * restrict c5 = c4 + n;
            double * restrict c6 = c5 + n;
            double * restrict c7 = c6 + n;

            const double *a0_ptr = &A[i0_n];
            const double *a1_ptr = a0_ptr + n;
            const double *a2_ptr = a1_ptr + n;
            const double *a3_ptr = a2_ptr + n;
            const double *a4_ptr = a3_ptr + n;
            const double *a5_ptr = a4_ptr + n;
            const double *a6_ptr = a5_ptr + n;
            const double *a7_ptr = a6_ptr + n;

            memset(c0, 0, (size_t)8 * n * sizeof(double));

            for (int jb = 0; jb < n; jb += 256) {
                int j_end = (jb + 256 <= n) ? jb + 256 : n;
                for (int k = 0; k < n; k++) {
                    double a0 = a0_ptr[k];
                    double a1 = a1_ptr[k];
                    double a2 = a2_ptr[k];
                    double a3 = a3_ptr[k];
                    double a4 = a4_ptr[k];
                    double a5 = a5_ptr[k];
                    double a6 = a6_ptr[k];
                    double a7 = a7_ptr[k];
                    const double * restrict bk = &B[(long)k * n];

                    #pragma clang loop vectorize(enable)
                    for (int j = jb; j < j_end; j++) {
                        double b = bk[j];
                        c0[j] += a0 * b;
                        c1[j] += a1 * b;
                        c2[j] += a2 * b;
                        c3[j] += a3 * b;
                        c4[j] += a4 * b;
                        c5[j] += a5 * b;
                        c6[j] += a6 * b;
                        c7[j] += a7 * b;
                    }
                }
            }
        } else {
            memset(&C[(long)i0 * n], 0, (size_t)rem * n * sizeof(double));
            for (int i = i0; i < imax; i++) {
                long in = (long)i * n;
                double * restrict c = &C[in];
                const double * restrict a_row = &A[in];

                for (int jb = 0; jb < n; jb += 256) {
                    int j_end = (jb + 256 <= n) ? jb + 256 : n;
                    for (int k = 0; k < n; k++) {
                        double a = a_row[k];
                        const double * restrict bk = &B[(long)k * n];

                        #pragma clang loop vectorize(enable)
                        for (int j = jb; j < j_end; j++) {
                            c[j] += a * bk[j];
                        }
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
