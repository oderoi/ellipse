#include "torch.h"

int main(){

    Tensor* x = tensor((int[]){1,2,3,4,5,6}, INT, (int[]){2,3}, true);
    print(x);

    return 0;
}