/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int w, int h, int max_iter, int *iters) {
    /* 
     * The Mandelbrot workload is highly irregular: center rows are much 
     * more expensive than edges. Additionally, the target machine has 
     * heterogeneous P-cores and E-cores. 
     * 
     * 'schedule(dynamic)' is used to ensure load balancing. Threads that 
     * finish their assigned rows (likely the P-cores or those working on 
     * the edges) can request new work, while the E-cores or those working 
     * on the center can take their time without bottlenecking the rest.
     */
    #pragma omp parallel for schedule(dynamic)
    for (int py = 0; py < h; py++) {
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
