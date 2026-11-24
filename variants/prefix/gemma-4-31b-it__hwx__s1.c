#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    /* 
     * We use a block-based scan algorithm to parallelize the prefix sum.
     * We divide the array into blocks; P-cores will process more blocks than E-cores 
     * thanks to 'schedule(guided)'.
     */
    int num_blocks = num_threads * 64; 
    long block_size = (n + num_blocks - 1) / num_blocks;
    long *block_sums = (long *)xmalloc(num_blocks * sizeof(long));

    #pragma omp parallel
    {
        for (int r = 0; r < reps; r++) {
            /* 
             * Pass 1: Compute local inclusive prefix sums for each block.
             * Each thread computes the scan for its assigned block and stores the 
             * total sum of that block in block_sums[b].
             */
            #pragma omp for schedule(guided)
            for (int b = 0; b < num_blocks; b++) {
                long start = (long)b * block_size;
                if (start >= n) {
                    block_sums[b] = 0;
                    continue;
                }
                long end = start + block_size;
                if (end > n) end = n;

                long local_run = 0;
                for (long i = start; i < end; i++) {
                    local_run += x[i];
                    out[i] = local_run;
                }
                block_sums[b] = local_run;
            }

            /* 
             * Pass 2: Compute the global prefix sum of the block totals.
             * This is a small sequential step.
             */
            #pragma omp single
            {
                for (int b = 1; b < num_blocks; b++) {
                    block_sums[b] += block_sums[b - 1];
                }
            }

            /* 
             * Pass 3: Correct the local prefix sums.
             * For each block b > 0, we add the total sum of all previous blocks (0...b-1)
             * to every element in the current block.
             */
            #pragma omp for schedule(guided)
            for (int b = 1; b < num_blocks; b++) {
                long start = (long)b * block_size;
                if (start >= n) continue;
                long end = start + block_size;
                if (end > n) end = n;

                long offset = block_sums[b - 1];
                for (long i = start; i < end; i++) {
                    out[i] += offset;
                }
            }
        }
    }
    free(block_sums);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 100000000);
    int reps = (int)arg_long(argc, argv, 2, 5);
    long *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(10);
    for (long i = 0; i < n; i++) x[i] = (long)(rng_next() % 100);

    double t0 = now_sec();
    kernel(n, reps, x, out);
    double t1 = now_sec();

    double sum = 0.0;   /* sample every 1000th entry plus the last one */
    for (long i = 0; i < n; i += 1000) sum += (double)out[i];
    sum += (double)out[n - 1];
    report("prefix", t1 - t0, sum);
    free(x); free(out);
    return 0;
}

