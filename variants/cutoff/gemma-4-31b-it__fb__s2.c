#include <math.h>
#include <omp.h>
#include <string.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, double r, const double *x, const double *y, int *nb) {
    double r2 = r * r;
    int nt = omp_get_max_threads();
    
    /* 
     * To avoid race conditions and minimize synchronization overhead, we use
     * thread-local buffers for counting neighbors. Each thread maintains its own 
     * count array for all points. 
     * The total memory is nt * n * sizeof(int), which for n=400,000 and nt=10
     * is ~16MB, well within typical limits.
     */
    int *nb_local = (int *)calloc((size_t)nt * n, sizeof(int));
    if (!nb_local) return; // Should not happen with calloc for this size

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *my_nb = &nb_local[(size_t)tid * n];
        
        /* 
         * The original algorithm counts neighbors by scanning both forward and backward.
         * This is redundant because if i is a neighbor of j, then j is a neighbor of i.
         * We only scan forward (j > i) and update both i and j, halving the total
         * distance calculations.
         */
        #pragma omp for schedule(guided)
        for (long i = 0; i < n; i++) {
            double xi = x[i];
            double yi = y[i];
            
            for (long j = i + 1; j < n; j++) {
                double dx = x[j] - xi;
                if (dx > r) break; // Since x is sorted, we can stop early
                
                double dy = y[j] - yi;
                if (dx * dx + dy * dy <= r2) {
                    my_nb[i]++;
                    my_nb[j]++;
                }
            }
        }
    }

    /* 
     * Sum the local counts into the final nb array. 
     * We iterate through each thread's buffer sequentially to maximize cache efficiency.
     */
    memset(nb, 0, (size_t)n * sizeof(int));
    for (int t = 0; t < nt; t++) {
        int *local_ptr = &nb_local[(size_t)t * n];
        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            nb[i] += local_ptr[i];
        }
    }

    free(nb_local);
}
/* === KERNEL END === */

static int cmp_x(const void *a, const void *b) {
    double d = ((const double *)a)[0] - ((const double *)b)[0];
    return (d > 0) - (d < 0);
}

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 400000);
    double r = 0.002;
    double *xy = xmalloc(2 * n * sizeof *xy);
    rng_seed(15);
    /* 70% of points in 8 tight Gaussian-ish clusters, 30% spread uniformly */
    for (long i = 0; i < n; i++) {
        if (rng_unit() < 0.7) {
            int c = (int)(rng_next() % 8);
            double cx = 0.1 + 0.1 * c, cy = 0.5;
            double a = rng_unit() + rng_unit() + rng_unit() - 1.5;   /* ~bell */
            double b = rng_unit() + rng_unit() + rng_unit() - 1.5;
            xy[2 * i] = cx + 0.01 * a; xy[2 * i + 1] = cy + 0.01 * b;
        } else { xy[2 * i] = rng_unit(); xy[2 * i + 1] = rng_unit(); }
    }
    qsort(xy, n, 2 * sizeof *xy, cmp_x);   /* sort (x,y) pairs by x */
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    for (long i = 0; i < n; i++) { x[i] = xy[2 * i]; y[i] = xy[2 * i + 1]; }
    int *nb = xmalloc(n * sizeof *nb);

    double t0 = now_sec();
    kernel(n, r, x, y, nb);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += nb[i];
    report("cutoff", t1 - t0, sum);
    free(xy); free(x); free(y); free(nb);
    return 0;
}

