/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The problem is compute-bound with a uniform cost per row.
     * To maximize performance on the Apple M4's heterogeneous architecture (4 P-cores, 6 E-cores),
     * we parallelize the outermost loop (i) using 'guided' scheduling.
     * This balances the load across cores of different speeds while keeping overhead low.
     * The (i, k, j) loop order is used to ensure sequential memory access for matrices B and C,
     * which allows Clang to effectively auto-vectorize the innermost loop.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        long i_off = (long)i * n;
        
        /* Initialize row C[i] to 0.0 */
        for (int j = 0; j < n; j++) {
            C[i_off + j] = 0.0;
        }

        for (int k = 0; k < n; k++) {
            double a = A[i_off + k];
            long k_off = (long)k * n;
            
            /* The inner loop iterates over contiguous memory (row-major),
             * providing optimal cache locality and enabling SIMD vectorization. */
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                C[i_off + j] += a * B[k_off + j];
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

