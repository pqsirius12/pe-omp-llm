/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    #pragma omp parallel
    {
        /* Tile output columns to improve cache reuse and limit working-set size. */
        const int TILE = 64;

        #pragma omp for schedule(static)
        for (int ii = 0; ii < n; ii += 1) {
            const double *restrict a = A + (long)ii * n;
            double *restrict c = C + (long)ii * n;

            for (int j = 0; j < n; j++)
                c[j] = 0.0;

            for (int k = 0; k < n; k++) {
                const double av = a[k];
                const double *restrict b = B + (long)k * n;

                #pragma omp simd
                for (int j = 0; j < n; j++)
                    c[j] += av * b[j];
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
