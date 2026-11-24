/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original algorithm streams arrays x and y from memory 'reps' times.
     * To maximize performance, we perform loop interchange: we process each 
     * element of x and y completely before moving to the next. This reduces 
     * memory traffic from O(reps * n) to O(n).
     *
     * We use '#pragma omp parallel for' to distribute the outer loop 'i' 
     * across threads. We use a static schedule since the workload per 
     * element is uniform. 
     *
     * We avoid using 'omp simd' explicitly here because Clang's -O2 
     * optimizer is highly effective at vectorizing simple streaming 
     * loops automatically. We use local variables for x[i] and y[i] to 
     * ensure the compiler keeps the intermediate values in registers 
     * during the 'reps' loop, minimizing memory access.
     */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double ax = a * xi;
        
        /* 
         * We keep the addition loop instead of using multiplication (reps * ax)
         * to maintain bit-for-bit numerical identity with the serial version,
         * as floating-point addition is not associative and multiplication 
         * can produce different rounding errors.
         */
        for (int r = 0; r < reps; r++) {
            yi += ax;
        }
        y[i] = yi;
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int reps = (int)arg_long(argc, argv, 2, 20);
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    rng_seed(1);
    for (long i = 0; i < n; i++) { x[i] = rng_unit(); y[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, reps, 0.5, x, y);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += y[i];
    report("saxpy", t1 - t0, sum);
    free(x); free(y);
    return 0;
}

