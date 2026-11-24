/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* 
     * The workload is highly irregular (some pixels take much longer than others),
     * and the machine has heterogeneous cores (4 P-cores, 6 E-cores).
     * 'schedule(dynamic, 1)' is used to ensure load balancing: faster P-cores 
     * will process more rows, and no thread is stuck with a disproportionate 
     * amount of "heavy" rows in the center of the set.
     */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < h; py++) {
        double ci = -1.5 + 3.0 * (double)py / h;
        for (int px = 0; px < w; px++) {
            double cr = -2.0 + 3.0 * (double)px / w;
            double zr = 0.0, zi = 0.0;
            int k = 0;
            while (k < max_iter && zr * zr + zi * zi <= 4.0) {
                double t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
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
