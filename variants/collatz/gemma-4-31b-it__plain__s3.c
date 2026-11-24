/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"

/* === KERNEL BEGIN === */
#include <omp.h>

static void kernel(long n, int *steps) {
    /* The Collatz sequence computation for each starting value is independent.
     * However, the number of steps is highly irregular, making static scheduling
     * inefficient. dynamic scheduling with a moderate chunk size ensures 
     * a good load balance across threads while minimizing scheduling overhead. */
    #pragma omp parallel for schedule(dynamic, 64)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            v = (v & 1) ? 3 * v + 1 : v / 2;
            s++;
        }
        steps[i] = s;
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int *steps = xmalloc(n * sizeof *steps);
    steps[0] = 0;

    double t0 = now_sec();
    kernel(n, steps);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 1; i < n; i++) sum += steps[i];
    report("collatz", t1 - t0, sum);
    free(steps);
    return 0;
}

