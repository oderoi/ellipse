#include "torch.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    Tensor *x = randn(FLOAT32, (int[]){2, 3}, true);
    Tensor *w = randn(FLOAT32, (int[]){2, 3}, true);

    // Tensor *yTrue = tensor((float[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, FLOAT32, (int[]){3, 3}, true);

    Tensor *yPred = matmul(w->T(w), x);

    // Tensor *loss = MSELoss(yTrue, yPred);
    // grad_init(loss);
    // backward(loss);

    print(w);
    print(yPred);

    t_free(x);
    t_free(w);
    t_free(yPred);
    // t_free(yPred);


    return 0;
}