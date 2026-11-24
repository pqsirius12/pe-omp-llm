/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double * restrict A, const double * restrict B, double * restrict C) {
    /*
     * To achieve maximum performance, we use several optimization techniques:
     * 1. Parallelization: We distribute the outermost loop (i) using OpenMP.
     * 2. Loop Order (IKJ): We use the IKJ order instead of the traditional IJK.
     *    This ensures that the innermost loop (j) accesses the memory of B and C 
     *    linearly, which is cache-friendly and enables efficient SIMD vectorization.
     * 3. Register Tiling/Unrolling: We unroll the 'k' loop by a factor of 4.
     *    This reduces the number of times we load and store elements of C[i, j] 
     *    by a factor of 4, increasing the arithmetic intensity and utilizing more 
     *    CPU registers.
     * 4. Aliasing Hints: The 'restrict' keyword informs the compiler that the 
     *    pointers A, B, and C do not overlap, allowing for more aggressive optimizations.
     * 5. SIMD: We use '#pragma omp simd' to explicitly encourage the compiler 
     *    to use AVX/FMA instructions for the innermost loop.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double * restrict Ci = &C[(long)i * n];

        /* Initialize the row of C to zero */
        for (int j = 0; j < n; j++) {
            Ci[j] = 0.0;
        }

        int k = 0;
        /* Unroll the k-loop by 4 to reuse the row of C across multiple rows of B */
        for (; k <= n - 4; k += 4) {
            double a0 = A[(long)i * n + k];
            double a1 = A[(long)i * n + (k + 1)];
            double a2 = A[(long)i * n + (k + 2)];
            double a3 = A[(long)i * n + (k + 3)];

            const double * restrict B0 = &B[(long)(k + 0) * n];
            const double * restrict B1 = &B[(long)(k + 1) * n];
            const double * restrict B2 = &B[(long)(k + 2) * n];
            const double * restrict B3 = &B[(long)(k + 3) * n];

            #pragma omp simd
            for (int j = 0; j < n; j++) {
                Ci[j] += a0 * B0[j] + a1 * B1[j] + a2 * B2[j] + a3 * B3[j];
            }
        }

        /* Handle the remaining k iterations if n is not a multiple of 4 */
        for (; k < n; k++) {
            double a = A[(long)i * n + k];
            const double * restrict Bk = &B[(long)k * n];
            #pragma omp simd
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

