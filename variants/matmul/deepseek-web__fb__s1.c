/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /*
     * i-k-j loop order with an explicitly transposed copy of B (Bt[j*n+k] = B[k*n+j]).
     * This turns the inner loop into a dot-product over contiguous memory for both
     * A's row and Bt's row, which the compiler auto-vectorizes cleanly (no gathers,
     * no false sharing on C, no strided access).
     *
     * Strategy:
     *   1. Transpose B into Bt in parallel (memory-bound, cheap relative to n^3).
     *   2. Parallelize over rows of C (uniform cost), compute each row as n dot
     *      products of length n over contiguous data.
     */
    long nn = (long)n * n;
    double *Bt = (double *)xmalloc((size_t)nn * sizeof *Bt);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Bk = &B[(long)k * n];
        for (int j = 0; j < n; j++)
            Bt[(long)j * n + k] = Bk[j];
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *Ai = &A[(long)i * n];
        double *Ci = &C[(long)i * n];
        for (int j = 0; j < n; j++) {
            const double *Bj = &Bt[(long)j * n];
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int k = 0; k < n; k++)
                s += Ai[k] * Bj[k];
            Ci[j] = s;
        }
    }

    free(Bt);
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
