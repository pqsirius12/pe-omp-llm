/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    int num_blocks = (n + 3) / 4;
    int BK = 64;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int b = 0; b < num_blocks; b++) {
            int i_start = b * 4;
            int i_end = (i_start + 4 <= n) ? i_start + 4 : n;
            for (int i = i_start; i < i_end; i++) {
                double *Ci = &C[(long)i * n];
                for (int j = 0; j < n; j++) {
                    Ci[j] = 0.0;
                }
            }
        }

        for (int kk = 0; kk < n; kk += BK) {
            int kend = kk + BK < n ? kk + BK : n;

            #pragma omp for schedule(static)
            for (int b = 0; b < num_blocks; b++) {
                int i = b * 4;
                if (i + 3 < n) {
                    double * restrict Ci0 = &C[(long)(i + 0) * n];
                    double * restrict Ci1 = &C[(long)(i + 1) * n];
                    double * restrict Ci2 = &C[(long)(i + 2) * n];
                    double * restrict Ci3 = &C[(long)(i + 3) * n];

                    for (int k = kk; k < kend; k++) {
                        double a0 = A[(long)(i + 0) * n + k];
                        double a1 = A[(long)(i + 1) * n + k];
                        double a2 = A[(long)(i + 2) * n + k];
                        double a3 = A[(long)(i + 3) * n + k];
                        const double * restrict Bk = &B[(long)k * n];

                        #pragma clang loop vectorize(enable)
                        for (int j = 0; j < n; j++) {
                            double b_val = Bk[j];
                            Ci0[j] += a0 * b_val;
                            Ci1[j] += a1 * b_val;
                            Ci2[j] += a2 * b_val;
                            Ci3[j] += a3 * b_val;
                        }
                    }
                } else {
                    for (int r = i; r < n; r++) {
                        double * restrict Cir = &C[(long)r * n];
                        for (int k = kk; k < kend; k++) {
                            double a = A[(long)r * n + k];
                            const double * restrict Bk = &B[(long)k * n];

                            #pragma clang loop vectorize(enable)
                            for (int j = 0; j < n; j++) {
                                Cir[j] += a * Bk[j];
                            }
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
