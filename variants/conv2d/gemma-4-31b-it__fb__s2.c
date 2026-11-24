/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;
    
    /* Local restrict pointers inform the compiler that the memory regions 
       do not overlap, enabling better SIMD vectorization. */
    const float * __restrict p_img = img;
    const float * __restrict p_filt = filt;
    float * __restrict p_out = out;

    /* Parallelize the outermost loop (y) to distribute work across threads.
       With oh = 6130, this provides excellent granularity for 1-10 threads. */
    #pragma omp parallel for
    for (int y = 0; y < oh; y++) {
        float * __restrict out_row = &p_out[(long)y * ow];
        
        /* Initialize output row. The cost is O(ow), which is negligible
           compared to the O(ow * KS^2) computation. */
        for (int x = 0; x < ow; x++) {
            out_row[x] = 0.0f;
        }

        /* Reorganizing the loop order from (y, x, ky, kx) to (y, ky, kx, x)
           transforms the innermost loop into a long, linear scan.
           This is far more efficient for SIMD vectorization and 
           hardware prefetching. */
        for (int ky = 0; ky < KS; ky++) {
            const float * __restrict img_row = &p_img[(long)(y + ky) * w];
            const float * __restrict filt_row = &p_filt[ky * KS];
            for (int kx = 0; kx < KS; kx++) {
                float f_val = filt_row[kx];
                
                /* The inner loop now processes 'ow' (6130) elements linearly,
                   perfect for SIMD. We use #pragma omp simd to hint 
                   the compiler to vectorize the operation. */
                #pragma omp simd
                for (int x = 0; x < ow; x++) {
                    out_row[x] += img_row[x + kx] * f_val;
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

