/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* The Mandelbrot workload is highly irregular because pixels inside the set 
       take the maximum number of iterations, while pixels outside escape quickly.
       We use 'schedule(guided)' to provide excellent load balancing across threads 
       while reducing the synchronization overhead associated with 'schedule(dynamic)'. */
    #pragma omp parallel for schedule(guided)
    for (int py = 0; py < h; py++) {
        /* ci only depends on py, so we compute it once per row. */
        double ci = -1.5 + 3.0 * (double)py / h;
        long row_offset = (long)py * w;
        
        for (int px = 0; px < w; px++) {
            /* cr depends on px. */
            double cr = -2.0 + 3.0 * (double)px / w;
            double zr = 0.0, zi = 0.0;
            int k = 0;
            
            /* The loop is the most computationally expensive part. 
               We cache zr*zr and zi*zi to avoid redundant multiplications 
               in every iteration. */
            while (k < max_iter) {
                double zr2 = zr * zr;
                double zi2 = zi * zi;
                
                /* Escape condition: |z|^2 > 4 */
                if (zr2 + zi2 > 4.0) break;
                
                /* z = z^2 + c
                   zr_next = zr^2 - zi^2 + cr
                   zi_next = 2 * zr * zi + ci */
                double next_zr = zr2 - zi2 + cr;
                zi = 2.0 * zr * zi + ci;
                zr = next_zr;
                k++;
            }
            iters[row_offset + px] = k;
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

