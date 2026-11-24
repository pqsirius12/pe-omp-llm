/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The target machine has a heterogeneous architecture (P-cores and E-cores).
     * Static scheduling would result in a bottleneck where the faster P-cores 
     * wait for the slower E-cores to complete their equal shares.
     * 'schedule(guided)' is used to dynamically balance the load, starting with 
     * large chunks and reducing them to handle the tail end of the computation 
     * efficiently across varying core speeds.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        long row_i = (long)i * n;
        
        /* Initialize the destination row to zero */
        for (int j = 0; j < n; j++) {
            C[row_i + j] = 0.0;
        }
        
        /* 
         * The loop order IKJ is used to ensure that the innermost loop 
         * accesses both B and C sequentially in memory, maximizing 
         * cache line utilization and enabling SIMD auto-vectorization.
         */
        for (int k = 0; k < n; k++) {
            double a = A[row_i + k];
            long row_k = (long)k * n;
            for (int j = 0; j < n; j++) {
                C[row_i + j] += a * B[row_k + j];
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
