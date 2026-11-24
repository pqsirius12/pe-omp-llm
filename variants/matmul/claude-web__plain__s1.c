/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define IB 32   /* rows per task tile        */
#define JB 256  /* columns per task tile     */
#define KB 128  /* k-block (B tile in cache) */

static void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i0 = 0; i0 < n; i0 += IB) {
        for (int j0 = 0; j0 < n; j0 += JB) {
            int imax = i0 + IB < n ? i0 + IB : n;
            int jmax = j0 + JB < n ? j0 + JB : n;
            int jlen = jmax - j0;

            for (int i = i0; i < imax; i++) {
                double *restrict c = C + (long)i * n + j0;
                for (int j = 0; j < jlen; j++) c[j] = 0.0;
            }

            for (int k0 = 0; k0 < n; k0 += KB) {
                int kmax = k0 + KB < n ? k0 + KB : n;
                int i = i0;
                for (; i + 4 <= imax; i += 4) {
                    double *restrict c0 = C + (long)(i    ) * n + j0;
                    double *restrict c1 = C + (long)(i + 1) * n + j0;
                    double *restrict c2 = C + (long)(i + 2) * n + j0;
                    double *restrict c3 = C + (long)(i + 3) * n + j0;
                    const double *a0 = A + (long)(i    ) * n;
                    const double *a1 = A + (long)(i + 1) * n;
                    const double *a2 = A + (long)(i + 2) * n;
                    const double *a3 = A + (long)(i + 3) * n;
                    for (int k = k0; k < kmax; k++) {
                        const double *restrict b = B + (long)k * n + j0;
                        double x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
                        for (int j = 0; j < jlen; j++) {
                            double bj = b[j];
                            c0[j] += x0 * bj;
                            c1[j] += x1 * bj;
                            c2[j] += x2 * bj;
                            c3[j] += x3 * bj;
                        }
                    }
                }
                for (; i < imax; i++) {
                    double *restrict c = C + (long)i * n + j0;
                    const double *ar = A + (long)i * n;
                    for (int k = k0; k < kmax; k++) {
                        const double *restrict b = B + (long)k * n + j0;
                        double a = ar[k];
                        for (int j = 0; j < jlen; j++)
                            c[j] += a * b[j];
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
