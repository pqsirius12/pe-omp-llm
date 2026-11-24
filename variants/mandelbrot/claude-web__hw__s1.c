/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define MB_BLOCK 128   /* pixels per work item (one row segment) */

static inline int mb_one(double cr, double ci, int max_iter) {
    double zr = 0.0, zi = 0.0;
    int k = 0;
    while (k < max_iter && zr * zr + zi * zi <= 4.0) {
        double t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        k++;
    }
    return k;
}

static void kernel(int w, int h, int max_iter, int *iters) {
    int nb = (w + MB_BLOCK - 1) / MB_BLOCK;

    /* Many small tiles handed out dynamically: copes with the irregular
     * per-pixel cost and with the mix of fast P-cores and slow E-cores. */
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int py = 0; py < h; py++) {
        for (int b = 0; b < nb; b++) {
            int x0 = b * MB_BLOCK;
            int x1 = x0 + MB_BLOCK;
            if (x1 > w) x1 = w;
            double ci = -1.5 + 3.0 * py / h;
            int *row = iters + (long)py * w;

            int px = x0;
            /* Two pixels at a time: independent dependency chains for ILP.
             * Each pixel uses exactly the serial expressions, so results
             * are bit-identical. */
            for (; px + 1 < x1; px += 2) {
                double cr0 = -2.0 + 3.0 * px / w;
                double cr1 = -2.0 + 3.0 * (px + 1) / w;
                double zr0 = 0.0, zi0 = 0.0, zr1 = 0.0, zi1 = 0.0;
                int k = 0;
                while (k < max_iter &&
                       zr0 * zr0 + zi0 * zi0 <= 4.0 &&
                       zr1 * zr1 + zi1 * zi1 <= 4.0) {
                    double t0 = zr0 * zr0 - zi0 * zi0 + cr0;
                    double t1 = zr1 * zr1 - zi1 * zi1 + cr1;
                    zi0 = 2.0 * zr0 * zi0 + ci;
                    zi1 = 2.0 * zr1 * zi1 + ci;
                    zr0 = t0;
                    zr1 = t1;
                    k++;
                }
                int k0 = k, k1 = k;
                /* Finish whichever pixel(s) are still running. */
                while (k0 < max_iter && zr0 * zr0 + zi0 * zi0 <= 4.0) {
                    double t = zr0 * zr0 - zi0 * zi0 + cr0;
                    zi0 = 2.0 * zr0 * zi0 + ci;
                    zr0 = t;
                    k0++;
                }
                while (k1 < max_iter && zr1 * zr1 + zi1 * zi1 <= 4.0) {
                    double t = zr1 * zr1 - zi1 * zi1 + cr1;
                    zi1 = 2.0 * zr1 * zi1 + ci;
                    zr1 = t;
                    k1++;
                }
                row[px] = k0;
                row[px + 1] = k1;
            }
            for (; px < x1; px++) {
                double cr = -2.0 + 3.0 * px / w;
                row[px] = mb_one(cr, ci, max_iter);
            }
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
