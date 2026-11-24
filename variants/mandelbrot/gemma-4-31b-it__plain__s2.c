/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* The Mandelbrot workload is highly irregular. Pixels in the center of the set 
       take significantly longer to compute than those on the edges. 
       'schedule(dynamic)' or 'schedule(guided)' is used to ensure load balancing 
       across threads. We parallelize the outer loop, which provides enough 
       granularity (h iterations) for the threads without excessive overhead. */
    #pragma omp parallel for schedule(dynamic)
    for (int py = 0; py < h; py++) {
        /* ci only depends on py, so we calculate it once per row. */
        double ci = -1.5 + 3.0 * py / h;
        for (int px = 0; px < w; px++) {
            double cr = -2.0 + 3.0 * px / w;
            double zr = 0.0, zi = 0.0;
            int k = 0;
            
            /* Inner loop optimization: minimize multiplications. 
               Note: we maintain the original logic to ensure identical results. */
            while (k < max_iter) {
                double zr2 = zr * zr;
                double zi2 = zi * zi;
                if (zr2 + zi2 > 4.0) break;
                
                double t = zr2 - zi2 + cr;
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
