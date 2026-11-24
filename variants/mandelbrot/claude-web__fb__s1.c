/* k05_mandelbrot: escape-time iteration count per pixel.
 * IRREGULAR and UNPREDICTABLE: pixels inside the set run to max_iter,
 * pixels outside escape early, and the costly rows sit in the middle. */
#include "common.h"
#include <omp.h>
#include <string.h>

/* === KERNEL BEGIN === */

/* Conservative interior test: main cardioid shrunk by 0.97 toward the
 * origin (the cardioid is star-shaped about 0) and period-2 bulb with a
 * slightly reduced radius.  Points passing this test have a strongly
 * attracting cycle, so the floating-point orbit can never escape and the
 * serial loop would run to max_iter. */
static inline int in_interior(double cr, double ci) {
    double b = cr + 1.0;
    if (b * b + ci * ci < 0.0625 * 0.95) return 1;      /* period-2 bulb */
    const double s = 1.0 / 0.97;                        /* shrink cardioid */
    double x = cr * s - 0.25, y = ci * s;
    double q = x * x + y * y;
    return q * (q + x) < 0.25 * y * y;                  /* main cardioid */
}

/* Same iteration as the serial code (identical expressions), plus exact
 * periodicity detection: if z repeats bit-for-bit, the orbit is periodic
 * in floating point forever and every state on it already passed the
 * escape test, so the serial loop would reach max_iter. */
static inline int mb_pixel(double cr, double ci, int max_iter) {
    double zr = 0.0, zi = 0.0;
    double sr = 0.0, si = 0.0;   /* saved state (state at k = 0) */
    int lim = 8;
    int k = 0;
    while (k < max_iter && zr * zr + zi * zi <= 4.0) {
        double t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        k++;
        if (zr == sr && zi == si) return max_iter;
        if (k == lim) { sr = zr; si = zi; lim <<= 1; }
    }
    return k;
}

static void kernel(int w, int h, int max_iter, int *iters) {
    /* Rows have very uneven cost, so hand them out dynamically.
     * Row h-py mirrors row py when its ci is exactly -ci: the orbit is then
     * the exact conjugate, giving identical counts, so one copy suffices. */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < h; py++) {
        const double ci = -1.5 + 3.0 * py / h;
        const int m = h - py;
        int mirror = 0;
        if (m >= 1 && m < h && m != py) {
            const double cim = -1.5 + 3.0 * m / h;
            if (cim == -ci) {
                if (m < py) continue;      /* computed by row m's iteration */
                mirror = 1;
            }
        }
        int *row = iters + (long)py * w;
        for (int px = 0; px < w; px++) {
            double cr = -2.0 + 3.0 * px / w;
            row[px] = (max_iter > 0 && in_interior(cr, ci))
                          ? max_iter
                          : mb_pixel(cr, ci, max_iter);
        }
        if (mirror)
            memcpy(iters + (long)m * w, row, (size_t)w * sizeof *row);
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
