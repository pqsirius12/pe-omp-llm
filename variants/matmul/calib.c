/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include "pe_split.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    pe_calibrate();
    #pragma omp parallel
    {
    long b, e;
    pe_range(0, n, &b, &e);
    for (int i = (int)b; i < (int)e; i++) {
        for (int j = 0; j < n; j++) C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++)
                C[(long)i * n + j] += a * B[(long)k * n + j];
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
