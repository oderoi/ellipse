#include "ellipse.h"
#include <stdio.h>
#include <time.h>

// Cross-platform timing
#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
    #include <sys/time.h>
#endif

// Include OpenBLAS threading control
#ifdef NAN_USE_OPENBLAS
    #include <cblas.h>
#endif

// Portable wall-clock timing function
double get_time_in_seconds() {
    #ifdef _WIN32
        LARGE_INTEGER freq, t;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart / freq.QuadPart;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec / 1e9;
    #endif
}

int main() {
    Tensor* a = ones(FLOAT64, (int[]){1000,1000}, false);
    Tensor* b = zeros(FLOAT64, (int[]){1000,1000}, false);

    size_t n = a->size;

    for(int i = 0; i < n; i++) {
        b->data.float64[i] = (double)2.0;        
    }
    
    // Optimize OpenBLAS threading (set to number of CPU cores)
    #ifdef NAN_USE_OPENBLAS
        // Use environment variable or runtime setting (e.g., 4 cores)
        // Can be overridden with OPENBLAS_NUM_THREADS environment variable
        int num_threads = 8; // Adjust based on your CPU
        openblas_set_num_threads(num_threads);
        printf("OpenBLAS using %d threads\n", openblas_get_num_threads());
    #endif

    // Time matrix multiplication
    double start = get_time_in_seconds();
    Tensor* c = matmul(a, b);
    double end = get_time_in_seconds();

    double time = end - start;
    printf("a @ b: took %.6f sec\n", time);
   
    return 0;
}
