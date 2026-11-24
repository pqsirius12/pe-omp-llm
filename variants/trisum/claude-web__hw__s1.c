/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
typedef double v2d __attribute__((vector_size(16)));

#define RB 16            /* rows per block (8 vectors of 2 doubles) */
#define NV (RB / 2)

static void kernel(long n, const double *x, double *out) {
    long nb = n / RB;          /* full blocks */
    long rem0 = nb * RB;       /* first leftover row */

    #pragma omp parallel
    {
        /* Leftover rows are the most expensive ones: do them first. */
        #pragma omp for schedule(dynamic, 1) nowait
        for (long i = n - 1; i >= rem0; i--) {
            double s = 0.0;
            for (long j = 0; j <= i; j++)
                s += x[j] / (double)(1 + i - j);
            out[i] = s;
        }

        /* Full blocks, largest (most work) first, handed out dynamically
           so fast P-cores and slower E-cores both stay busy to the end. */
        #pragma omp for schedule(dynamic, 1)
        for (long b = nb - 1; b >= 0; b--) {
            long i0 = b * RB;
            v2d s[NV], d[NV];
            const v2d one = {1.0, 1.0};
            for (int k = 0; k < NV; k++) {
                s[k] = (v2d){0.0, 0.0};
                /* denominator for row i0+r at j=0 is 1 + i0 + r */
                d[k] = (v2d){(double)(1 + i0 + 2 * k), (double)(2 + i0 + 2 * k)};
            }

            /* Common range j = 0..i0, valid for all rows of the block.
               Each lane accumulates in the same order as the serial code. */
            for (long j = 0; j <= i0; j++) {
                double xj = x[j];
                v2d xv = {xj, xj};
                for (int k = 0; k < NV; k++) {
                    s[k] += xv / d[k];
                    d[k] -= one;
                }
            }

            double part[RB];
            for (int k = 0; k < NV; k++) {
                part[2 * k]     = s[k][0];
                part[2 * k + 1] = s[k][1];
            }

            /* Triangular tail: row i0+r still needs j = i0+1 .. i0+r. */
            for (int r = 0; r < RB; r++) {
                long i = i0 + r;
                double acc = part[r];
                for (long j = i0 + 1; j <= i; j++)
                    acc += x[j] / (double)(1 + i - j);
                out[i] = acc;
            }
        }
    }
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
