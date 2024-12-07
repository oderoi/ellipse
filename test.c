#include "torch.h"
// #include <stdio.h>
// #include <omp.h>

int main() {
    Tensor *A = randn(FLOAT32, (int[]){10, 50}, 2, true);
    Tensor *B = randn(FLOAT32, (int[]){50, 10}, 2, false);

    Tensor *C = matmul(A, B);

    print(C);

    return 0;
}

// int main(){
//     #pragma omp parallel
//     {
//         printf("Hello from thread %d out of %d threads\n",omp_get_thread_num(), omp_get_num_threads());
//     }
//     return 0;
// }