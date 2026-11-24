/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* 
     * Division is significantly more expensive than multiplication.
     * Since the denominators are always integers from 1 to n, 
     * we precompute the reciprocals once to convert the O(n^2) divisions 
     * into O(n^2) multiplications.
     */
    double *inv = xmalloc((n + 1) * sizeof(double));
    for (long k = 1; k <= n; k++) {
        inv[k] = 1.0 / (double)k;
    }

    /* 
     * The computational load increases linearly with i.
     * schedule(guided) is ideal here: it allocates large chunks initially 
     * and smaller chunks as i increases, ensuring that the most expensive 
     * iterations (at the end of the loop) are distributed finely across 
     * all available threads for optimal load balancing.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < n; i++) {
        double s = 0.0;
        /* 
         * The inner loop is a simple dot product of x[0...i] and 
         * inv[i+1...1]. Using a local accumulator s and precomputed 
         * reciprocals allows the compiler to vectorize the loop 
         * using SIMD instructions (e.g., AVX2/AVX-512).
         */
        #pragma omp simd reduction(+:s)
        for (long j = 0; j <= i; j++) {
            s += x[j] * inv[i + 1 - j];
        }
        out[i] = s;
    }
    
    free(inv);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 40000);
    double *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(4);
    for (long i = 0; i < n; i++) x[i] = rng_unit();

    double t0 = now_sec();
    kernel(n, x, out);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += out[i];
    report("trisum", t1 - t0, sum);
    free(x); free(out);
    return 0;
}

