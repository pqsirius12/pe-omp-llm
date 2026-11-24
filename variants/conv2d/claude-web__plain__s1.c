/* k13_conv2d: 2-D convolution of an image with a 15x15 filter (valid region).
 * UNIFORM cost per output pixel, compute-bound: the core of CNN layers. */
#include "common.h"
#include <omp.h>

#define KS 15

/* === KERNEL BEGIN === */
#define XCH 512

static void kernel(int h, int w, const float *img, const float *filt, float *out) {
    int oh = h - KS + 1, ow = w - KS + 1;
    if (oh <= 0 || ow <= 0) return;

    float fl[KS * KS];
    for (int k = 0; k < KS * KS; k++) fl[k] = filt[k];

    #pragma omp parallel firstprivate(fl)
    {
        float acc[XCH];
        #pragma omp for schedule(static)
        for (int y = 0; y < oh; y++) {
            float *orow = out + (long)y * ow;
            for (int x0 = 0; x0 < ow; x0 += XCH) {
                int len = ow - x0 < XCH ? ow - x0 : XCH;
                for (int x = 0; x < len; x++) acc[x] = 0.0f;
                for (int ky = 0; ky < KS; ky++) {
                    const float *restrict r = img + (long)(y + ky) * w + x0;
                    const float f0 = fl[ky*KS+0],  f1 = fl[ky*KS+1],  f2 = fl[ky*KS+2];
                    const float f3 = fl[ky*KS+3],  f4 = fl[ky*KS+4],  f5 = fl[ky*KS+5];
                    const float f6 = fl[ky*KS+6],  f7 = fl[ky*KS+7],  f8 = fl[ky*KS+8];
                    const float f9 = fl[ky*KS+9],  f10 = fl[ky*KS+10], f11 = fl[ky*KS+11];
                    const float f12 = fl[ky*KS+12], f13 = fl[ky*KS+13], f14 = fl[ky*KS+14];
                    #pragma omp simd
                    for (int x = 0; x < len; x++) {
                        float s = acc[x];
                        s += r[x + 0]  * f0;
                        s += r[x + 1]  * f1;
                        s += r[x + 2]  * f2;
                        s += r[x + 3]  * f3;
                        s += r[x + 4]  * f4;
                        s += r[x + 5]  * f5;
                        s += r[x + 6]  * f6;
                        s += r[x + 7]  * f7;
                        s += r[x + 8]  * f8;
                        s += r[x + 9]  * f9;
                        s += r[x + 10] * f10;
                        s += r[x + 11] * f11;
                        s += r[x + 12] * f12;
                        s += r[x + 13] * f13;
                        s += r[x + 14] * f14;
                        acc[x] = s;
                    }
                }
                for (int x = 0; x < len; x++) orow[x0 + x] = acc[x];
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
