/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    const int oh = h - KS + 1;
    const int ow = w - KS + 1;

    /* Cache the filter in a contiguous, aligned-friendly layout. */
    float f[KS * KS];
    for (int i = 0; i < KS * KS; i++)
        f[i] = filt[i];

    #pragma omp parallel
    {
        /* Each thread processes a contiguous block of output rows.
           This gives good cache locality and minimizes scheduling overhead. */
        #pragma omp for schedule(static)
        for (int y = 0; y < oh; y++) {
            const long outrow = (long)y * ow;
            const long imgrow = (long)y * w;

            for (int x = 0; x < ow; x++) {
                float s = 0.0f;

                /* Explicitly unroll the inner filter dimension. */
                for (int ky = 0; ky < KS; ky++) {
                    const float *row = img + imgrow + (long)ky * w + x;
                    const float *frow = f + ky * KS;

                    #pragma omp simd reduction(+:s)
                    for (int kx = 0; kx < KS; kx++)
                        s += row[kx] * frow[kx];
                }

                out[outrow + x] = s;
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
