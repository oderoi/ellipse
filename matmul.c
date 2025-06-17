#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

// Tunable tile size for cache efficiency (optimized for L1/L2 cache)
#define TILE_SIZE 64

// Matrix multiplication: C = A * B (n x n matrices)
void matmul(double *A, double *B, double *C, int n) {
    // Initialize C to zero
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    #pragma omp parallel for simd  collapse(2)
    for (int i = 0; i < n; i += TILE_SIZE) {
        for (int j = 0; j < n; j += TILE_SIZE) {
            for (int k = 0; k < n; k += TILE_SIZE) {
                int i_max = (i + TILE_SIZE < n) ? i + TILE_SIZE : n;
                int j_max = (j + TILE_SIZE < n) ? j + TILE_SIZE : n;
                int k_max = (k + TILE_SIZE < n) ? k + TILE_SIZE : n;

                for (int ii = i; ii < i_max; ii++) {
                    for (int jj = j; jj < j_max; jj += 4) { // Process 4 doubles (AVX2)
                        __m256d c_vec = _mm256_load_pd(&C[ii * n + jj]);
                        for (int kk = k; kk < k_max; kk++) {
                            __m256d a_vec = _mm256_broadcast_sd(&A[ii * n + kk]);
                            __m256d b_vec = _mm256_load_pd(&B[kk * n + jj]);
                            c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        }
                        _mm256_store_pd(&C[ii * n + jj], c_vec);
                    }
                    // Handle remaining elements (jj not divisible by 4)
                    for (int jj = j_max - (j_max % 4); jj < j_max; jj++) {
                        double sum = C[ii * n + jj];
                        for (int kk = k; kk < k_max; kk++) {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    int n = 1000;

    // Allocate aligned memory for AVX2 (32-byte alignment)
    double *A = (double *)_mm_malloc(n * n * sizeof(double), 32);
    double *B = (double *)_mm_malloc(n * n * sizeof(double), 32);
    double *C = (double *)_mm_malloc(n * n * sizeof(double), 32);

    // Initialize matrices (same as PyTorch test: fill with 1.0)
    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    // Warm-up run (to avoid cache effects)
    matmul(A, B, C, n);

    // Measure time
    double start = omp_get_wtime();
    matmul(A, B, C, n);
    double end = omp_get_wtime();

    printf("Time taken: %.3f ms\n", (end - start) * 1000);

    // Verify result (optional, for debugging)
    /*
    printf("Sample result (C[0][0]): %.2f\n", C[0]);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", C[i * n + j]);
        }
        printf("\n");
    }
    */

    // Free memory
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    return 0;
}
