/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    double s = 0.0;

    /* One parallel region for all repetitions avoids fork/join overhead per rep.
     * Each thread always processes the same contiguous block (good NUMA/cache
     * locality across reps) and accumulates privately; partial sums are
     * combined once at the end via the reduction clause. */
    #pragma omp parallel reduction(+:s)
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();
        long chunk = n / nt;
        long rem   = n % nt;
        long lo = tid * chunk + (tid < rem ? tid : rem);
        long hi = lo + chunk + (tid < rem ? 1 : 0);

        double local = 0.0;
        for (int r = 0; r < reps; r++) {
            double t = 0.0;
            #pragma omp simd reduction(+:t)
            for (long i = lo; i < hi; i++)
                t += sqrt(fabs(x[i]));
            local += t;
        }
        s += local;
    }
    return s;
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 50000000);
    int reps = (int)arg_long(argc, argv, 2, 6);
    double *x = xmalloc(n * sizeof *x);
    rng_seed(9);
    for (long i = 0; i < n; i++) x[i] = rng_unit() - 0.5;

    double t0 = now_sec();
    double s = kernel(n, reps, x);
    double t1 = now_sec();

    report("norm", t1 - t0, s);
    free(x);
    return 0;
}
