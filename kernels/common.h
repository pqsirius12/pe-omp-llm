/* common.h - shared helpers for every kernel program.
 * Nothing here is parallel: timing, deterministic input generation,
 * argument parsing and the one-line result report the harness parses. */
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Wall-clock seconds from a monotonic clock (unaffected by clock changes). */
static inline double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* xorshift64: tiny deterministic generator, identical on Linux and macOS,
 * so every run and every LLM variant gets exactly the same input data. */
static uint64_t rng_state = 88172645463325252ULL;
static inline void rng_seed(uint64_t s) { rng_state = s ? s : 1; }
static inline uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
/* Uniform double in [0, 1): top 53 bits of the generator. */
static inline double rng_unit(void) {
    return (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline long arg_long(int argc, char **argv, int i, long def) {
    return argc > i ? atol(argv[i]) : def;
}

static inline void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

/* One machine-readable line per run; the harness parses exactly this. */
static inline void report(const char *name, double secs, double checksum) {
    printf("kernel=%s time_s=%.6f checksum=%.12e\n", name, secs, checksum);
}

#endif
