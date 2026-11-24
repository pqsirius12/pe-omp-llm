/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The matrix multiplication is implemented using the IKJ loop order.
     * This order is significantly more cache-friendly than the standard IJK order
     * because the innermost loop (j) accesses matrices B and C sequentially.
     * Since each row of the result matrix C is independent, we parallelize the
     * outermost loop (i) using OpenMP.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        long i_offset = (long)i * n;
        double *Ci = &C[i_offset];

        /* Initialize the current row of C to zero */
        for (int j = 0; j < n; j++) {
            Ci[j] = 0.0;
        }

        /* Perform the multiplication using the IKJ pattern */
        for (int k = 0; k < n; k++) {
            double a = A[i_offset + k];
            const double *Bk = &B[(long)k * n];
            
            /* The following loop is a simple multiply-accumulate on contiguous memory,
               which Clang's -O2 will typically vectorize using SIMD instructions. */
            for (int j = 0; j < n; j++) {
                Ci[j] += a * Bk[j];
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
