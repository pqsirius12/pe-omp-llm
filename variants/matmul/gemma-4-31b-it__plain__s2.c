/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The matrix multiplication is implemented using the i-k-j loop order.
     * This order is highly cache-efficient as the innermost loop (j) 
     * accesses memory linearly for both matrix B and matrix C.
     * Each row of C is computed independently, making the outer loop 'i'
     * the ideal candidate for parallelization.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        long i_off = (long)i * n;
        double *ci = &C[i_off];
        const double *ai = &A[i_off];

        /* Initialize the target row of C to zero */
        for (int j = 0; j < n; j++) {
            ci[j] = 0.0;
        }

        /* Compute the row result using a linear scan of B */
        for (int k = 0; k < n; k++) {
            double a = ai[k];
            const double *bk = &B[(long)k * n];
            /* The compiler (clang -O2) will vectorize this inner loop */
            for (int j = 0; j < n; j++) {
                ci[j] += a * bk[j];
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
