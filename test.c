#include "torch.h"

int main(){

    Tensor* x = tensor((int[]){1,2,3,4,5,6}, INT, (int[]){2,3}, false);
    // Tensor* t = ones(x->dtype, x->dims, false);
    Tensor* y = Pow(x, 3);
    print(x);
    // print(t);
    print(y);

    return 0;
}
