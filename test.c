#include "torch.h"
#include <stdio.h>

int main() {
    Tensor *x=tensor((double[]){1, 2, 3, 4, 5, 6}, FLOAT64, (int[]){2,3}, 2, true);
    Tensor *w=tensor((double[]){7, 8, 9, 10, 11, 12} ,FLOAT64, (int[]){3, 2}, 2, true);
    // Tensor *b = randn(FLOAT64, (int[]){2,2}, 2, true);
    Tensor *w_t = w->T(w);
    
    Tensor *t = matmul(x, w);
    Tensor *a = relu(t);
    Tensor *z = relu(a);
    grad_init(z);
    backward(z);

    print(x);

    // for (int i =0; i<z->size; i++){
    //     printf("z[%d]: %f,\n", i,(1- z->data.float32[i]));
    // }

    t_free(x);
    t_free(w);
    // t_free(b);
    t_free(t);
}