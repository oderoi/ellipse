#include "torch.h"

int main() {
    Tensor *x = randn(FLOAT32, (int[]){2, 3}, 2, true);
    Tensor *w = randd(FLOAT32, (int[]){3, 2}, 2, true);
    Tensor *yTrue = tensor((float[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, FLOAT32, (int[]){3, 3}, 2, true);

    Tensor *yPred = matmul(w, x);

    Tensor *loss = MSELoss(yTrue, yPred);
    grad_init(loss);
    backward(loss);

    print(w);
    print(loss);

    return 0;
}