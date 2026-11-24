/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;

    /* Cache the filter in registers/stack so inner loops don't re-read it
     * (it is 225 floats = 900 bytes, fits easily in L1). */
    float fl[KS * KS];
    for (int k = 0; k < KS * KS; k++) fl[k] = filt[k];

    /* Parallelize over output rows.  Each thread handles a contiguous band
     * of rows.  The row-inner loop keeps the same accumulation order per
     * output pixel as the serial version, so results are bit-identical. */
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int y = 0; y < oh; y++) {
            const float *restrict img_row = img + (long)y * w;
            float *restrict out_row = out + (long)y * ow;
            for (int x = 0; x < ow; x++) {
                float s = 0.0f;
                for (int ky = 0; ky < KS; ky++) {
                    const float *restrict p = img_row + (long)ky * w + x;
                    const float *restrict f = fl + ky * KS;
                    #pragma omp simd reduction(+:s)
                    for (int kx = 0; kx < KS; kx++)
                        s += p[kx] * f[kx];
                }
                out_row[x] = s;
            }
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int h = (int)arg_long(argc, argv, 1, 6144), w = (int)arg_long(argc, argv, 2, 6144);
    long oh = h - KS + 1, ow = w - KS + 1;
    float *img = xmalloc((long)h * w * sizeof *img), filt[KS * KS];
    float *out = xmalloc(oh * ow * sizeof *out);
    rng_seed(13);
    for (long k = 0; k < (long)h * w; k++) img[k] = (float)rng_unit();
    for (int k = 0; k < KS * KS; k++) filt[k] = (float)rng_unit() / (KS * KS);

    double t0 = now_sec();
    kernel(h, w, img, filt, out);
    double t1 = now_sec();

    double sum = 0.0;   /* each pixel is computed in a fixed order: exact */
    for (long k = 0; k < oh * ow; k++) sum += out[k];
    report("conv2d", t1 - t0, sum);
    free(img); free(out);
    return 0;
}
