/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;
    
    /* 
     * Each output pixel (y, x) is computed independently, making it a perfectly parallel problem.
     * We parallelize the outer loop 'y' as it provides a large enough iteration space for 
     * any number of threads (up to 10) and ensures that work is distributed in contiguous 
     * chunks of rows, which is cache-friendly for the output array.
     * 
     * The filter is small (15x15 = 225 floats), easily fitting in L1 cache.
     * The access pattern for 'img' is also cache-friendly: for a fixed 'y', as 'x' increments,
     * we reuse the same set of 15 rows of the image.
     */
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            float s = 0.0f;
            for (int ky = 0; ky < KS; ky++) {
                /* Prefetch row pointers to assist the compiler in vectorization */
                const float *img_ptr = &img[(long)(y + ky) * w + x];
                const float *filt_ptr = &filt[ky * KS];
                
                /* The innermost loop is a dot product of length 15. 
                 * Modern compilers with -O2 will vectorize this automatically. */
                for (int kx = 0; kx < KS; kx++) {
                    s += img_ptr[kx] * filt_ptr[kx];
                }
            }
            out[(long)y * ow + x] = s;
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

