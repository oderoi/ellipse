#include "torch.h"

int main(){
    Tensor *t=tensor((float[]){1,2,3,4,5,6}, FLOAT32, (int[]){3,2}, true);
    print(t);

    return 0;
}