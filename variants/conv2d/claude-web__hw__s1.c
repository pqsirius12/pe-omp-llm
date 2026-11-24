/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
typedef float v4  __attribute__((vector_size(16)));
typedef float v4u __attribute__((vector_size(16), aligned(4)));   /* unaligned access */

#define RB 4   /* output rows per block        */
#define CV 4   /* 4-float vectors per block    */
#define CB (4 * CV)  /* output columns per block */

/* Exactly the serial per-pixel computation (same order, same expression). */
static inline float pixel_scalar(int w, const float *img, const float *filt, int y, int x) {
    float s = 0.0f;
    for (int ky = 0; ky < KS; ky++)
        for (int kx = 0; kx < KS; kx++)
            s += img[(long)(y + ky) * w + (x + kx)] * filt[ky * KS + kx];
    return s;
}

static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;
    if (oh <= 0 || ow <= 0) return;

    v4 fb[KS * KS];
    for (int k = 0; k < KS * KS; k++) {
        float f = filt[k];
        fb[k] = (v4){f, f, f, f};
    }

    int nb = (oh + RB - 1) / RB;
    int xv = ow - ow % CB;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nb; b++) {
        int y = b * RB;
        if (y + RB <= oh) {
            for (int x0 = 0; x0 < xv; x0 += CB) {
                v4 acc[RB][CV];
                #pragma unroll
                for (int j = 0; j < RB; j++)
                    #pragma unroll
                    for (int c = 0; c < CV; c++)
                        acc[j][c] = (v4){0.0f, 0.0f, 0.0f, 0.0f};

                for (int r = 0; r < KS + RB - 1; r++) {
                    const float *row = img + (long)(y + r) * w + x0;
                    for (int kx = 0; kx < KS; kx++) {
                        v4 v[CV];
                        #pragma unroll
                        for (int c = 0; c < CV; c++)
                            v[c] = *(const v4u *)(row + kx + 4 * c);
                        #pragma unroll
                        for (int j = 0; j < RB; j++) {
                            int ky = r - j;
                            if (ky >= 0 && ky < KS) {
                                v4 f = fb[ky * KS + kx];
                                #pragma unroll
                                for (int c = 0; c < CV; c++)
                                    acc[j][c] += v[c] * f;
                            }
                        }
                    }
                }

                #pragma unroll
                for (int j = 0; j < RB; j++)
                    #pragma unroll
                    for (int c = 0; c < CV; c++)
                        *(v4u *)(out + (long)(y + j) * ow + x0 + 4 * c) = acc[j][c];
            }
            /* leftover columns */
            for (int j = 0; j < RB; j++)
                for (int x = xv; x < ow; x++)
                    out[(long)(y + j) * ow + x] = pixel_scalar(w, img, filt, y + j, x);
        } else {
            /* leftover rows */
            for (int yy = y; yy < oh; yy++)
                for (int x = 0; x < ow; x++)
                    out[(long)yy * ow + x] = pixel_scalar(w, img, filt, yy, x);
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
