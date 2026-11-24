/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    /* 
     * The matrix multiplication is implemented using the i-k-j loop order.
     * This order is highly cache-efficient because the innermost loop (j) 
     * accesses memory linearly for both matrix B and matrix C.
     * 
     * Each row of C is independent, so the outer loop 'i' is parallelized.
     * We unroll the 'k' loop by a factor of 4 to reduce the number of loads/stores 
     * to matrix C and to increase instruction-level parallelism (ILP).
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double * restrict ci = &C[(long)i * n];
        const double * restrict ai = &A[(long)i * n];

        /* Initialize the target row of C to zero */
        for (int j = 0; j < n; j++) {
            ci[j] = 0.0;
        }

        int k = 0;
        /* Unroll the k-loop to reduce memory traffic for C and enable better FMA usage */
        for (; k <= n - 4; k += 4) {
            double a0 = ai[k];
            double a1 = ai[k + 1];
            double a2 = ai[k + 2];
            double a3 = ai[k + 3];
            
            const double * restrict bk0 = &B[(long)k * n];
            const double * restrict bk1 = &B[(long)(k + 1) * n];
            const double * restrict bk2 = &B[(long)(k + 2) * n];
            const double * restrict bk3 = &B[(long)(k + 3) * n];

            #pragma omp simd
            for (int j = 0; j < n; j++) {
                ci[j] += a0 * bk0[j] + a1 * bk1[j] + a2 * bk2[j] + a3 * bk3[j];
            }
        }
        
        /* Handle remaining k elements */
        for (; k < n; k++) {
            double a = ai[k];
            const double * restrict bk = &B[(long)k * n];
            #pragma omp simd
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
