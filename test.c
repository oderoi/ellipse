#include "torch.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    Tensor *x = randn(FLOAT32, (int[]){2, 3}, true);
    Tensor *w = randn(FLOAT32, (int[]){3, 2}, true);
    Tensor *yTrue = tensor((float[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, FLOAT32, (int[]){3, 3}, false);

    Tensor *yPred = matmul(w, x);

    Tensor *loss = MAELoss(yTrue, yPred);
    grad_init(loss);
    backward(loss);

    print(w);
    print(x);

    t_free(x);
    t_free(w);
    t_free(yTrue);
    // t_free(yPred);


    return 0;
}