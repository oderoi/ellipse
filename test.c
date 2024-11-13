#include "torch.h"
#include <stdio.h>

int main() {

    Tensor *x = randn(FLOAT32, (int[]){5,2}, 2, false);
    Tensor *w = randn(FLOAT32, (int[]){2,5}, 2, true);
    Tensor *b = randd(FLOAT32, (int[]){2,2}, 2, true);

    float lr = 0.5;

    print(w);

    for (int i = 0; i < 10000; i++)
    {
        Tensor *z= add(matmul(w, x), b);
        Tensor *a= relu(z); 
        Tensor *ypred= softmax(a);
        Tensor *loss= mean(ypred);
        grad_init(loss);
        backward(loss);

        for (int i = 0; i < w->size; i++)
        {
            w->data.float32[i] -= w->grad.float32[i] * lr;
        }

        if (i%2000 == 0)
        {
            printf("loss: %f\n",loss->data.float32[0]);
        }
        
        t_free(z);
        t_free(a);
        t_free(ypred);
        t_free(loss);
    }

    print(w);

    t_free(x);
    t_free(w);
    t_free(b);
    return 0;
}