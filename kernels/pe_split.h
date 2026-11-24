/* pe_split.h - calibrated uneven split of a loop across threads
 * (the idea behind AID and the AINA 2026 calibration baseline).
 *
 * pe_calibrate(): every thread runs the same small, fixed workload at the
 * same time and times itself. A thread that landed on a P-core finishes
 * sooner, so it gets a bigger share of the loop.
 * pe_range(): gives the calling thread its share [*b, *e) of [lo, hi).
 *
 * Known limits, reported in the paper: (1) macOS may move a thread between
 * P- and E-cores after calibration; (2) the calibration loop is
 * compute-bound, so for memory-bound loops the measured speed ratio is
 * only an estimate; (3) calibration time counts toward the kernel's time. */
#ifndef PE_SPLIT_H
#define PE_SPLIT_H
#include <omp.h>

#define PE_MAXT 256
static double pe_w[PE_MAXT + 1];   /* cumulative share: thread t owns [pe_w[t], pe_w[t+1]) */
static int pe_nt = 1;

static void pe_calibrate(void) {
    double speed[PE_MAXT];
    #pragma omp parallel
    {
        #pragma omp single
        pe_nt = omp_get_num_threads();        /* implied barrier: all start together */
        int t = omp_get_thread_num();
        volatile double x = 1.0;              /* volatile: the compiler cannot skip it */
        double t0 = omp_get_wtime();
        for (long i = 0; i < 5000000; i++) x = x * 1.0000001 + 1e-9;
        speed[t] = 1.0 / (omp_get_wtime() - t0);
    }
    double total = 0.0;
    for (int t = 0; t < pe_nt; t++) total += speed[t];
    pe_w[0] = 0.0;
    for (int t = 0; t < pe_nt; t++) pe_w[t + 1] = pe_w[t] + speed[t] / total;
}

static inline void pe_range(long lo, long hi, long *b, long *e) {
    int t = omp_get_thread_num();
    long n = hi - lo;
    *b = lo + (long)(pe_w[t] * n);
    *e = (t == pe_nt - 1) ? hi : lo + (long)(pe_w[t + 1] * n);   /* last thread takes the rest */
}
#endif
