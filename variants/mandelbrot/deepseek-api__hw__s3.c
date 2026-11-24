/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* Dynamic scheduling: work per pixel varies wildly (pixels inside the
     * set run to max_iter, others escape early), and the heavy middle rows
     * are not uniformly spread. With a heterogeneous 4P+6E machine where the
     * OS schedules threads freely, a static split leaves fast cores idle or
     * overloaded. Dynamic chunks let the runtime rebalance as E-cores lag.
     * Chunk = one row (w pixels) keeps coalesced writes per row while giving
     * enough granularity to balance across all threads. */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < h; py++) {
        int *row = iters + (long)py * w;
        double ci = -1.5 + 3.0 * py / h;
        for (int px = 0; px < w; px++) {
            double cr = -2.0 + 3.0 * px / w;
            double zr = 0.0, zi = 0.0;
            int k = 0;
            while (k < max_iter && zr * zr + zi * zi <= 4.0) {
                double t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                k++;
            }
            row[px] = k;
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
