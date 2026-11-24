/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* 
     * Use 'guided' scheduling to balance the workload across the P-cores and E-cores.
     * The workload is highly irregular, with the center rows taking significantly 
     * more time than the edges. Guided scheduling starts with large chunks and 
     * decreases them over time, which minimizes overhead while ensuring load balance 
     * on heterogeneous CPU architectures.
     */
    #pragma omp parallel for schedule(guided)
    for (int py = 0; py < h; py++) {
        double ci = -1.5 + 3.0 * py / h;
        for (int px = 0; px < w; px++) {
            double cr = -2.0 + 3.0 * px / w;
            double zr = 0.0, zi = 0.0;
            double zr2 = 0.0, zi2 = 0.0;
            int k = 0;
            
            /* 
             * Optimized inner loop to reduce redundant multiplications.
             * pre-calculating zr2 and zi2 inside the loop reduces 
             * the number of operations per iteration.
             */
            while (k < max_iter && zr2 + zi2 <= 4.0) {
                zi = 2.0 * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                zr2 = zr * zr;
                zi2 = zi * zi;
                k++;
            }
            iters[(long)py * w + px] = k;
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int w = (int)arg_long(argc, argv, 1, 2048);
    int h = (int)arg_long(argc, argv, 2, 2048);
    int max_iter = (int)arg_long(argc, argv, 3, 1000);
    int *iters = xmalloc((long)w * h * sizeof *iters);

    double t0 = now_sec();
    kernel(w, h, max_iter, iters);
    double t1 = now_sec();

    double sum = 0.0;   /* integer counts: exact regardless of thread order */
    for (long k = 0; k < (long)w * h; k++) sum += iters[k];
    report("mandelbrot", t1 - t0, sum);
    free(iters);
    return 0;
}

