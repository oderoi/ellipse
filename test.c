#include "nan.h"

int main(){

    Tensor* x = tensor((double[]){1,2,3,4,5,6,7,8}, FLOAT64, (int[]){2,4}, true);
    // Tensor* t = ones(x->dtype, x->dims, false);
    // Tensor* y = Pow(x, 3);
    print(x);
    // print(t);
    // print(y);

    return 0;
}
