/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    if (n <= 0 || reps <= 0) return;

#if defined(__ARM_NEON)
    const float64x2_t va = vdupq_n_f64(a);
    const long CHUNK = 8192;
    const long num_chunks = (n + CHUNK - 1) / CHUNK;

    #pragma omp parallel for schedule(dynamic, 1)
    for (long ci = 0; ci < num_chunks; ci++) {
        long start = ci * CHUNK;
        long end = start + CHUNK;
        if (end > n) end = n;

        long i = start;
        for (; i + 15 < end; i += 16) {
            float64x2_t vy0 = vld1q_f64(y + i);
            float64x2_t vy1 = vld1q_f64(y + i + 2);
            float64x2_t vy2 = vld1q_f64(y + i + 4);
            float64x2_t vy3 = vld1q_f64(y + i + 6);
            float64x2_t vy4 = vld1q_f64(y + i + 8);
            float64x2_t vy5 = vld1q_f64(y + i + 10);
            float64x2_t vy6 = vld1q_f64(y + i + 12);
            float64x2_t vy7 = vld1q_f64(y + i + 14);

            float64x2_t vx0 = vld1q_f64(x + i);
            float64x2_t vx1 = vld1q_f64(x + i + 2);
            float64x2_t vx2 = vld1q_f64(x + i + 4);
            float64x2_t vx3 = vld1q_f64(x + i + 6);
            float64x2_t vx4 = vld1q_f64(x + i + 8);
            float64x2_t vx5 = vld1q_f64(x + i + 10);
            float64x2_t vx6 = vld1q_f64(x + i + 12);
            float64x2_t vx7 = vld1q_f64(x + i + 14);

            for (int r = 0; r < reps; r++) {
                vy0 = vfmaq_f64(vy0, vx0, va);
                vy1 = vfmaq_f64(vy1, vx1, va);
                vy2 = vfmaq_f64(vy2, vx2, va);
                vy3 = vfmaq_f64(vy3, vx3, va);
                vy4 = vfmaq_f64(vy4, vx4, va);
                vy5 = vfmaq_f64(vy5, vx5, va);
                vy6 = vfmaq_f64(vy6, vx6, va);
                vy7 = vfmaq_f64(vy7, vx7, va);
            }

            vst1q_f64(y + i, vy0);
            vst1q_f64(y + i + 2, vy1);
            vst1q_f64(y + i + 4, vy2);
            vst1q_f64(y + i + 6, vy3);
            vst1q_f64(y + i + 8, vy4);
            vst1q_f64(y + i + 10, vy5);
            vst1q_f64(y + i + 12, vy6);
            vst1q_f64(y + i + 14, vy7);
        }

        for (; i < end; i++) {
            double yi = y[i];
            double xi = x[i];
            for (int r = 0; r < reps; r++) {
                yi = a * xi + yi;
            }
            y[i] = yi;
        }
    }
#else
    const long CHUNK = 8192;
    const long num_chunks = (n + CHUNK - 1) / CHUNK;

    #pragma omp parallel for schedule(dynamic, 1)
    for (long ci = 0; ci < num_chunks; ci++) {
        long start = ci * CHUNK;
        long end = start + CHUNK;
        if (end > n) end = n;

        for (long i = start; i < end; i++) {
            double yi = y[i];
            double xi = x[i];
            for (int r = 0; r < reps; r++) {
                yi = a * xi + yi;
            }
            y[i] = yi;
        }
    }
#endif
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int reps = (int)arg_long(argc, argv, 2, 20);
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    rng_seed(1);
    for (long i = 0; i < n; i++) { x[i] = rng_unit(); y[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, reps, 0.5, x, y);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += y[i];
    report("saxpy", t1 - t0, sum);
    free(x); free(y);
    return 0;
}
