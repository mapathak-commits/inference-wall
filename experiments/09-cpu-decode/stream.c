/* Threaded STREAM triad — aggregate sustained DRAM bandwidth.
 * The decode ceiling is aggregate bandwidth across all cores, which numpy's
 * single-threaded element-wise ops can't reach. Compile:
 *   gcc -O3 -march=native -fopenmp src/stream.c -o stream
 * Run with OMP_NUM_THREADS set. Arrays sized past L3 (32 MiB) so every access hits DRAM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (96*1024*1024)   /* 96M doubles = 768 MiB per array, 3 arrays = 2.25 GiB */
#define NTIMES 20

static double a[N], b[N], c[N];

int main(void) {
    double scalar = 3.0;
    #pragma omp parallel for
    for (long i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double best = 1e30;
    for (int k = 0; k < NTIMES; k++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < N; i++) a[i] = b[i] + scalar * c[i];
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
        if (a[k] < 0) printf("x");  /* defeat dead-code elimination */
    }
    /* triad moves 3 arrays: read b, read c, write a */
    double bytes = 3.0 * (double)N * sizeof(double);
    int nt = omp_get_max_threads();
    printf("{\"threads\": %d, \"triad_gbps\": %.2f, \"array_mib\": %.0f}\n",
           nt, bytes / best / 1e9, (double)N * sizeof(double) / 1024 / 1024);
    return 0;
}
