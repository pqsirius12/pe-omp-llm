/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;

    /* 
     * Parallelize the outermost loop to distribute the rows of the output 
     * image across available threads.
     */
    #pragma omp parallel for
    for (int y = 0; y < oh; y++) {
        /* Pointer to the start of the current output row. */
        float *out_row = &out[(long)y * ow];

        /* 
         * Reorder loops from (y, x, ky, kx) to (y, ky, kx, x) to maximize 
         * SIMD efficiency. By making 'x' the innermost loop, we transform 
         * the operation into a series of AXPY-like updates: 
         * out_row[x] += img_ptr[x] * f_val.
         *
         * Because 'ow' (6130) is large, the 'x' loop is easily vectorized.
         * The 'out_row' (approx 24.5 KB) fits comfortably in the L1 cache,
         * reducing memory traffic.
         */
        for (int ky = 0; ky < KS; ky++) {
            const float *img_row = &img[(long)(y + ky) * w];
            const float *filt_row = &filt[ky * KS];
            for (int kx = 0; kx < KS; kx++) {
                float f_val = filt_row[kx];
                const float *img_ptr = img_row + kx;

                if (ky == 0 && kx == 0) {
                    /* Use initialization for the first filter element to avoid 
                       a separate memset/zeroing loop. */
                    #pragma omp simd
                    for (int x = 0; x < ow; x++) {
                        out_row[x] = img_ptr[x] * f_val;
                    }
                } else {
                    /* Accumulate the result for subsequent filter elements. */
                    #pragma omp simd
                    for (int x = 0; x < ow; x++) {
                        out_row[x] += img_ptr[x] * f_val;
                    }
                }
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

