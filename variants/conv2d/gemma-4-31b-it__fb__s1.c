/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;

    /* 
     * To maximize performance while ensuring bitwise identical results to the serial version, 
     * we preserve the exact summation order: for each output pixel (y, x), the accumulation 
     * happens sequentially across ky and then kx.
     * 
     * We use a technique called "blocking" or "strip-mining" on the x-dimension. 
     * By processing a small block of output pixels (e.g., 16) at once, we can keep the 
     * partial sums for those pixels in registers (or L1 cache), transforming the 
     * innermost loop into a highly efficient vector-scalar multiply-add (AXPY) operation.
     */
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < oh; y++) {
        float *out_row = &out[(long)y * ow];
        
        for (int x = 0; x < ow; x += 16) {
            int x_limit = (x + 16 <= ow) ? 16 : (ow - x);
            float sums[16] = {0.0f};
            
            for (int ky = 0; ky < KS; ky++) {
                const float *img_row = &img[(long)(y + ky) * w + x];
                const float *filt_row = &filt[ky * KS];
                
                for (int kx = 0; kx < KS; kx++) {
                    float f = filt_row[kx];
                    /* This innermost loop is a simple sequential scan over 'i'. 
                     * Modern compilers (clang -O2) will vectorize this loop using 
                     * SIMD instructions (e.g., AVX/SSE), providing massive speedup. */
                    for (int i = 0; i < x_limit; i++) {
                        sums[i] += img_row[i + kx] * f;
                    }
                }
            }
            
            for (int i = 0; i < x_limit; i++) {
                out_row[x + i] = sums[i];
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

