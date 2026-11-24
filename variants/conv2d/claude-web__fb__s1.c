/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>
#include <stdint.h>
#include <sys/mman.h>

#define KS 15

/* === KERNEL BEGIN === */
#define XCH 1024

#define CONV_ROW_BODY                \
    s += r[x + 0]  * f0;             \
    s += r[x + 1]  * f1;             \
    s += r[x + 2]  * f2;             \
    s += r[x + 3]  * f3;             \
    s += r[x + 4]  * f4;             \
    s += r[x + 5]  * f5;             \
    s += r[x + 6]  * f6;             \
    s += r[x + 7]  * f7;             \
    s += r[x + 8]  * f8;             \
    s += r[x + 9]  * f9;             \
    s += r[x + 10] * f10;            \
    s += r[x + 11] * f11;            \
    s += r[x + 12] * f12;            \
    s += r[x + 13] * f13;            \
    s += r[x + 14] * f14;

#define LOAD_FILTER_ROW(f)                                            \
    const float f0 = f[0],  f1 = f[1],  f2 = f[2],  f3 = f[3];        \
    const float f4 = f[4],  f5 = f[5],  f6 = f[6],  f7 = f[7];        \
    const float f8 = f[8],  f9 = f[9],  f10 = f[10], f11 = f[11];     \
    const float f12 = f[12], f13 = f[13], f14 = f[14];

/* First filter row: start from 0.0f exactly like the serial code. */
static inline void row_first(float *restrict o, const float *restrict r,
                             const float *restrict f, int len) {
    LOAD_FILTER_ROW(f)
    #pragma omp simd
    for (int x = 0; x < len; x++) {
        float s = 0.0f;
        CONV_ROW_BODY
        o[x] = s;
    }
}

/* Subsequent filter rows: continue the running sum kept in the output. */
static inline void row_next(float *restrict o, const float *restrict r,
                            const float *restrict f, int len) {
    LOAD_FILTER_ROW(f)
    #pragma omp simd
    for (int x = 0; x < len; x++) {
        float s = o[x];
        CONV_ROW_BODY
        o[x] = s;
    }
}

static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;
    if (oh <= 0 || ow <= 0) return;

#ifdef MADV_HUGEPAGE
    /* Fewer page faults when first touching the (fresh) output buffer. */
    {
        uintptr_t b = (uintptr_t)out;
        uintptr_t e = b + (uintptr_t)oh * (uintptr_t)ow * sizeof(float);
        uintptr_t pb = (b + 4095) & ~(uintptr_t)4095;
        uintptr_t pe = e & ~(uintptr_t)4095;
        if (pe > pb) madvise((void *)pb, pe - pb, MADV_HUGEPAGE);
    }
#endif

    float fl[KS * KS];
    for (int k = 0; k < KS * KS; k++) fl[k] = filt[k];

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < oh; y++) {
        float *orow = out + (long)y * ow;
        for (int x0 = 0; x0 < ow; x0 += XCH) {
            int len = ow - x0 < XCH ? ow - x0 : XCH;
            float *o = orow + x0;
            row_first(o, img + (long)y * w + x0, fl, len);
            for (int ky = 1; ky < KS; ky++)
                row_next(o, img + (long)(y + ky) * w + x0, fl + ky * KS, len);
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
