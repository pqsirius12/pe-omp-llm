/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The loop order is i, k, j. This is the most cache-friendly layout for 
     * row-major matrices as the innermost loop iterates sequentially 
     * over both C and B.
     * 
     * Given the Apple M4's hybrid architecture (4 P-cores, 6 E-cores), 
     * 'schedule(guided)' is used to balance the workload. It prevents 
     * faster P-cores from idling while slower E-cores finish their chunks.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        long i_off = (long)i * n;
        
        // Initialize the current row of C to zero.
        for (int j = 0; j < n; j++) {
            C[i_off + j] = 0.0;
        }
        
        for (int k = 0; k < n; k++) {
            double a = A[i_off + k];
            long k_off = (long)k * n;
            
            // The compiler can auto-vectorize this loop since it's a 
            // contiguous memory access pattern (SIMD).
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

