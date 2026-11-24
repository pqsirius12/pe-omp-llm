/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* 
     * 1. Replace division with multiplication:
     *    Computing (1.0 / k) once for k = 1..n replaces n(n+1)/2 divisions
     *    with n(n+1)/2 multiplications. Divisions are significantly slower.
     * 2. Use 'guided' schedule:
     *    Since the workload for iteration i grows linearly (i+1), a 
     *    guided schedule balances the load by starting with larger 
     *    chunks and decreasing them, ensuring that threads are not 
     *    idle while others finish heavy iterations at the end.
     * 3. SIMD vectorization:
     *    The inner loop is a dot product of a sequence and a reversed 
     *    sequence. Using #pragma omp simd encourages the compiler to 
     *    use AVX/SIMD instructions to process multiple elements per cycle.
     */
    double *inv = (double *)xmalloc((n + 1) * sizeof(double));
    for (long k = 1; k <= n; k++) {
        inv[k] = 1.0 / (double)k;
    }

    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < n; i++) {
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (long j = 0; j <= i; j++) {
            s += x[j] * inv[i - j + 1];
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

