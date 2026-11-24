/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * Use OpenMP to parallelize the outer loop. 
     * On a machine with mixed P-cores and E-cores (like Apple M4), 
     * static scheduling would lead to load imbalance because the E-cores 
     * would take longer to finish their assigned chunks.
     * 'schedule(guided)' starts with large chunks and reduces them over time, 
     * allowing faster P-cores to process more rows while reducing scheduling 
     * overhead compared to 'schedule(dynamic)'.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        long row_C = (long)i * n;
        
        /* Initialize the row of C to zero */
        for (int j = 0; j < n; j++) {
            C[row_C + j] = 0.0;
        }
        
        /* 
         * Reordered loops (i, k, j) to ensure row-major memory access patterns.
         * This ensures that the innermost loop over 'j' iterates linearly 
         * across rows of both B and C, maximizing cache hits and enabling 
         * the compiler to effectively use SIMD (vector) instructions.
         */
        for (int k = 0; k < n; k++) {
            double a = A[row_C + k];
            long row_B = (long)k * n;
            for (int j = 0; j < n; j++) {
                C[row_C + j] += a * B[row_B + j];
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
