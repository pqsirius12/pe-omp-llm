/* probe2.c - does the thread priority class (QoS) decide P- vs E-core speed?
 * Same fixed-work probe as probe.c (integer chain), but each thread first
 * sets its own QoS class:  ./probe2 <mode>
 *   none : leave the class as OpenMP/macOS gave it (also reported)
 *   ui   : QOS_CLASS_USER_INTERACTIVE (highest)
 *   util : QOS_CLASS_UTILITY (lower)
 * API verified in Apple's libpthread headers: pthread_set_qos_class_self_np,
 * pthread_get_qos_class_np (<pthread/qos.h>). macOS only. */
#include <omp.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

volatile unsigned long IA = 3, IB = 1;
double sink[256];

static int cmp(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

static const char *qname(qos_class_t q) {
    switch (q) {
    case QOS_CLASS_USER_INTERACTIVE: return "user-interactive";
    case QOS_CLASS_USER_INITIATED:   return "user-initiated";
    case QOS_CLASS_DEFAULT:          return "default";
    case QOS_CLASS_UTILITY:          return "utility";
    case QOS_CLASS_BACKGROUND:       return "background";
    case QOS_CLASS_UNSPECIFIED:      return "unspecified";
    default:                         return "other";
    }
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "none";
    long work = argc > 2 ? atol(argv[2]) : 300000000L;
    qos_class_t want = QOS_CLASS_UNSPECIFIED;
    if (!strcmp(mode, "ui")) want = QOS_CLASS_USER_INTERACTIVE;
    else if (!strcmp(mode, "util")) want = QOS_CLASS_UTILITY;

    double t[256];
    qos_class_t seen[256];
    int nt = 1, failed = 0;
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        if (want != QOS_CLASS_UNSPECIFIED && pthread_set_qos_class_self_np(want, 0) != 0) {
            #pragma omp atomic
            failed++;
        }
        int rel;
        pthread_get_qos_class_np(pthread_self(), &seen[id], &rel);
        #pragma omp single
        nt = omp_get_num_threads();           /* implied barrier: start together */
        double t0 = omp_get_wtime();
        unsigned long v = 27 + id, a = IA, b = IB;
        for (long i = 0; i < work; i++) v = (v & 1) ? a * v + b : (v >> 1) | 7;
        sink[id] = (double)v;
        t[id] = omp_get_wtime() - t0;
    }
    double chk = 0;
    for (int i = 0; i < nt; i++) chk += sink[i];
    qsort(t, nt, sizeof t[0], cmp);
    printf("mode=%-4s threads=%2d qos(main)=%s qos(last)=%s set_failures=%d times:",
           mode, nt, qname(seen[0]), qname(seen[nt - 1]), failed);
    for (int i = 0; i < nt; i++) printf(" %.3f", t[i]);
    printf("  (check %.3g)\n", chk);
    return 0;
}
