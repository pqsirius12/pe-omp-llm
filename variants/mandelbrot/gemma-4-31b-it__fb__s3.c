/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* 
     * The Mandelbrot set workload is highly irregular. The central region 
     * (the set itself) requires the maximum number of iterations, while 
     * peripheral pixels escape quickly. 
     * 
     * To optimize:
     * 1. We use 'schedule(guided)' on the outer loop. Guided scheduling 
     *    starts with large chunks and reduces them over time, which 
     *    effectively balances the load across threads with lower overhead 
     *    than 'dynamic' scheduling for this range of h (2048) and threads (1-10).
     * 2. We move the calculation of 'ci' outside the inner loop since it 
     *    is constant for all pixels in a row.
     * 3. Inside the while-loop, we precompute zr*zr and zi*zi. This reduces 
     *    the number of multiplications per iteration from 4 to 3.
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
             * Optimization: By tracking the squares of zr and zi, we only 
             * need to calculate them once per iteration to satisfy both 
             * the escape condition and the calculation of the next zr.
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

