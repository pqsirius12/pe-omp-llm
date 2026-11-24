/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <string.h>
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < n; i++) {
        long in = (long)i * n;
        double * restrict c = &C[in];
        memset(c, 0, (size_t)n * sizeof(double));
        const double * restrict ai = &A[in];

        int k = 0;
        for (; k + 3 < n; k += 4) {
            double a0 = ai[k];
            double a1 = ai[k + 1];
            double a2 = ai[k + 2];
            double a3 = ai[k + 3];
            const double * restrict b0 = &B[(long)(k + 0) * n];
            const double * restrict b1 = &B[(long)(k + 1) * n];
            const double * restrict b2 = &B[(long)(k + 2) * n];
            const double * restrict b3 = &B[(long)(k + 3) * n];

            #pragma clang loop vectorize(enable) interleave(enable)
            for (int j = 0; j < n; j++) {
                double cj = c[j];
                cj += a0 * b0[j];
                cj += a1 * b1[j];
                cj += a2 * b2[j];
                cj += a3 * b3[j];
                c[j] = cj;
            }
        }
        for (; k < n; k++) {
            double a = ai[k];
            const double * restrict b = &B[(long)k * n];

            #pragma clang loop vectorize(enable)
            for (int j = 0; j < n; j++) {
                c[j] += a * b[j];
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
