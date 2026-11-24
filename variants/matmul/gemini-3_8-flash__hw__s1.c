/* k02_matmul: C = A*B (n x n). UNIFORM cost per row, compute-bound. */
#include "common.h"
#include <omp.h>
#include <arm_neon.h>

/* === KERNEL BEGIN === */
static void kernel(int n, const double *A, const double *B, double *C) {
    int n4 = n & ~3;
    int n8 = n & ~7;

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n4; i += 4) {
        const double *a_ptr0 = A + (long)(i + 0) * n;
        const double *a_ptr1 = A + (long)(i + 1) * n;
        const double *a_ptr2 = A + (long)(i + 2) * n;
        const double *a_ptr3 = A + (long)(i + 3) * n;

        int j = 0;
        for (; j < n8; j += 8) {
            float64x2_t c00 = vdupq_n_f64(0.0);
            float64x2_t c01 = vdupq_n_f64(0.0);
            float64x2_t c02 = vdupq_n_f64(0.0);
            float64x2_t c03 = vdupq_n_f64(0.0);

            float64x2_t c10 = vdupq_n_f64(0.0);
            float64x2_t c11 = vdupq_n_f64(0.0);
            float64x2_t c12 = vdupq_n_f64(0.0);
            float64x2_t c13 = vdupq_n_f64(0.0);

            float64x2_t c20 = vdupq_n_f64(0.0);
            float64x2_t c21 = vdupq_n_f64(0.0);
            float64x2_t c22 = vdupq_n_f64(0.0);
            float64x2_t c23 = vdupq_n_f64(0.0);

            float64x2_t c30 = vdupq_n_f64(0.0);
            float64x2_t c31 = vdupq_n_f64(0.0);
            float64x2_t c32 = vdupq_n_f64(0.0);
            float64x2_t c33 = vdupq_n_f64(0.0);

            const double *b_row = B + j;
            for (int k = 0; k < n; k++) {
                float64x2_t a0 = vdupq_n_f64(a_ptr0[k]);
                float64x2_t a1 = vdupq_n_f64(a_ptr1[k]);
                float64x2_t a2 = vdupq_n_f64(a_ptr2[k]);
                float64x2_t a3 = vdupq_n_f64(a_ptr3[k]);

                float64x2_t b0 = vld1q_f64(b_row + 0);
                float64x2_t b1 = vld1q_f64(b_row + 2);
                float64x2_t b2 = vld1q_f64(b_row + 4);
                float64x2_t b3 = vld1q_f64(b_row + 6);

                c00 = vfmaq_f64(c00, b0, a0);
                c01 = vfmaq_f64(c01, b1, a0);
                c02 = vfmaq_f64(c02, b2, a0);
                c03 = vfmaq_f64(c03, b3, a0);

                c10 = vfmaq_f64(c10, b0, a1);
                c11 = vfmaq_f64(c11, b1, a1);
                c12 = vfmaq_f64(c12, b2, a1);
                c13 = vfmaq_f64(c13, b3, a1);

                c20 = vfmaq_f64(c20, b0, a2);
                c21 = vfmaq_f64(c21, b1, a2);
                c22 = vfmaq_f64(c22, b2, a2);
                c23 = vfmaq_f64(c23, b3, a2);

                c30 = vfmaq_f64(c30, b0, a3);
                c31 = vfmaq_f64(c31, b1, a3);
                c32 = vfmaq_f64(c32, b2, a3);
                c33 = vfmaq_f64(c33, b3, a3);

                b_row += n;
            }

            double *c_ptr0 = C + (long)(i + 0) * n + j;
            double *c_ptr1 = C + (long)(i + 1) * n + j;
            double *c_ptr2 = C + (long)(i + 2) * n + j;
            double *c_ptr3 = C + (long)(i + 3) * n + j;

            vst1q_f64(c_ptr0 + 0, c00);
            vst1q_f64(c_ptr0 + 2, c01);
            vst1q_f64(c_ptr0 + 4, c02);
            vst1q_f64(c_ptr0 + 6, c03);

            vst1q_f64(c_ptr1 + 0, c10);
            vst1q_f64(c_ptr1 + 2, c11);
            vst1q_f64(c_ptr1 + 4, c12);
            vst1q_f64(c_ptr1 + 6, c13);

            vst1q_f64(c_ptr2 + 0, c20);
            vst1q_f64(c_ptr2 + 2, c21);
            vst1q_f64(c_ptr2 + 4, c22);
            vst1q_f64(c_ptr2 + 6, c23);

            vst1q_f64(c_ptr3 + 0, c30);
            vst1q_f64(c_ptr3 + 2, c31);
            vst1q_f64(c_ptr3 + 4, c32);
            vst1q_f64(c_ptr3 + 6, c33);
        }

        for (; j < n; j++) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            const double *b_col = B + j;
            for (int k = 0; k < n; k++) {
                double b = *b_col;
                sum0 += a_ptr0[k] * b;
                sum1 += a_ptr1[k] * b;
                sum2 += a_ptr2[k] * b;
                sum3 += a_ptr3[k] * b;
                b_col += n;
            }
            C[(long)(i + 0) * n + j] = sum0;
            C[(long)(i + 1) * n + j] = sum1;
            C[(long)(i + 2) * n + j] = sum2;
            C[(long)(i + 3) * n + j] = sum3;
        }
    }

    for (int i = n4; i < n; i++) {
        for (int j = 0; j < n; j++) C[(long)i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
            double a = A[(long)i * n + k];
            for (int j = 0; j < n; j++)
                C[(long)i * n + j] += a * B[(long)k * n + j];
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int n = (int)arg_long(argc, argv, 1, 1024);
    long nn = (long)n * n;
    double *A = xmalloc(nn * sizeof *A), *B = xmalloc(nn * sizeof *B);
    double *C = xmalloc(nn * sizeof *C);
    rng_seed(2);
    for (long i = 0; i < nn; i++) { A[i] = rng_unit(); B[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, A, B, C);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < nn; i++) sum += C[i];
    report("matmul", t1 - t0, sum);
    free(A); free(B); free(C);
    return 0;
}
