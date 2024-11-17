//An Array of pointer to structure
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#define MAX_PREVS 3

typedef union
{
    float* float32;
    double* float64;
    int32_t* int32;
    int64_t* int64;
    void* raw_data;
} Data;

typedef union
{
    float *float32;
    double *float64;
}Grad;

typedef enum{
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
}DType;

typedef enum{
    ADD,
    SUM,
    SUB,
    MUL,
    MATMUL,
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    MEAN,
    SOFTMAX,
    DIV,
    POW,
    EXP,
    MSE
}Op;

typedef enum{
    true,
    false
}bool;

typedef struct Tensor{
    Data data;
    DType dtype;
    double extra;
    int *dims;
    int ndim;
    int size;
    Op op;
    Grad grad;
    struct Tensor * prevs[MAX_PREVS];
    bool requires_grad;
    int num_prevs;
}Tensor;

static size_t dtype_size(DType dtype){
    switch(dtype){
        case FLOAT32: return sizeof(float);
        case FLOAT64: return sizeof(double);
        case INT32: return sizeof(int32_t);
        case INT64: return sizeof(int64_t);
        default: return 0;
    }
}

static int total_size(int * dims, int ndim){
    int size=1;
    for(int i=0; i<ndim; i++){
        size *= dims[i];
    }
    return size;
}

Tensor * tensor(void * data, DType dtype, int * dims, int ndim, bool requires_grad){
    if(!dims || ndim <= 0) return NULL;
    //allocate memory for the tensor structure
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if(!t){
        fprintf(stderr, "Memory allocation for tensor failed\n");
        return NULL;
    }
    t->dtype = dtype;
    t->size = total_size(dims, ndim);
    t->ndim = ndim;
    t->extra = 0;
    t->requires_grad = requires_grad;
    t->op = -1;
    t->num_prevs = 0;
    t->dims = (int *)malloc(ndim*sizeof(int));
    if(!t->dims){
        fprintf(stderr, "Memory allocation for dims failed\n");
        free(t);
        return NULL;
    }
    memcpy(t->dims, dims, ndim*sizeof(int));
    switch(dtype){
        case FLOAT32:
            t->data.float32 = (float*) calloc(t->size , sizeof(float));
            if(!t->data.float32){
                fprintf(stderr, "Memory allocation for data failed\n");
                free(t->dims);
                free(t);
                return NULL;
            }
            if(data){
                memcpy(t->data.float32, data, t->size*sizeof(float));
            }
            if(t->requires_grad==true){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            t->data.float64 = (double*) calloc(t->size, sizeof(double));
            if(!t->data.float64){
                fprintf(stderr, "Memory allocation for data failed\n");
                free(t->dims);
                free(t);
                return NULL;
            }
            if(data){
                memcpy(t->data.float64, data, t->size*sizeof(double));
            }
            if(t->requires_grad==true){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            t->data.int32 = (int32_t*) calloc(t->size, sizeof(int32_t));
            if(!t->data.int32){
                fprintf(stderr, "Memory allocation for data failed\n");
                free(t->dims);
                free(t);
                return NULL;
            }
            if(data){
                memcpy(t->data.int32, data, t->size*sizeof(int32_t));
            }
            break;
        case INT64:
            t->data.int64 = (int64_t*) calloc(t->size, sizeof(int64_t));
            if(!t->data.int64){
                fprintf(stderr, "Memory allocation for data failed\n");
                free(t->dims);
                free(t);
                return NULL;
            }
            if(data){
                memcpy(t->data.int64, data, t->size*sizeof(int64_t));
            }
            break;
        default:
        fprintf(stderr, "Unsupported data type\n");
        free(t->dims);
        free(t);
    }
    return t;
}

Tensor * zeros(DType dtype, int * dims, int ndim, bool requires_grad){
    Tensor *t = tensor(NULL, dtype, dims, ndim, requires_grad);
    if(!t)return NULL;
    switch(dtype){
        case FLOAT32:
            for (int i = 0; i < t->size; i++)
            {
               t->data.float32[i] = 0.0f;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t->size; i++){
                t->data.float64[i] = 0.0;
            }
            break;
        case INT32:
            for(int i=0; i<t->size; i++){
                t->data.int32[i] = 0;
            }
            break;
        case INT64:
            for(int i=0; i<t->size; i++){
                t->data.int64[i] = 0;
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    return t;
}
    
Tensor * ones(DType dtype, int * dims, int ndim, bool requires_grad) {
    Tensor *t = tensor(NULL, dtype, dims, ndim, requires_grad);
    if(!t) return NULL;
    //Fill with ones based on dtype
    switch (dtype){
        case FLOAT32:
            for (int i = 0; i < t->size; i++) t->data.float32[i] = 1.0f;
            break;
        case FLOAT64:
            for (int i = 0; i < t->size; i++) t->data.float64[i] = 1.0;
            break;
        case INT32:
            for (int i = 0; i < t->size; i++) t->data.int32[i] = 1;
            break;
        case INT64:
            for (int i = 0; i < t->size; i++) t->data.int64[i] = 1;
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    return t;
}

/*
In PyTorch, the function torch.randn() generates random numbers from a normal (Gaussian) distribution with a mean of 0 and a standard deviation of 1. The values typically range from around -3 to 3, but there’s no strict bound because normal distribution tails extend infinitely.

Here’s how it works:

	•	The values are centered around 0, with most numbers lying within approximately ±3 due to the distribution’s properties.
	•	About 68% of values will be within ±1 (one standard deviation from the mean).
	•	About 95% will be within ±2.
	•	Around 99.7% will fall within ±3.
*/

Tensor * randn(DType dtype, int *dims, int ndim, bool requires_grad){
    Tensor *t = tensor(NULL, dtype, dims, ndim, requires_grad);
    if (!t) return NULL;
    srand(time(NULL));
    switch (dtype){
        case FLOAT32:
            for (int i = 0; i < t->size; i++){
                float u1 = (float)rand() / (float)RAND_MAX;
                float u2 = (float)rand() / (float)RAND_MAX;
                t->data.float32[i] =(float) sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            }
            break;
        case FLOAT64:
            for (int i = 0; i < t->size; i++){
                double u1 = (double)rand() / (double)RAND_MAX;
                double u2 = (double)rand() / (double)RAND_MAX;
                t->data.float64[i] =(double) sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            }
            break;
        case INT32:
        case INT64:
            free(t);
            fprintf(stderr, " \"randn\" not implemented for \'int\' dtype \n"); 
            return NULL;
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    return t;
}

Tensor * randd(DType dtype, int * dims, int ndim, bool requires_grad){
    Tensor *t = tensor(NULL, dtype, dims, ndim, requires_grad);
    if (!t) return NULL;  
    srand(time(NULL));
    switch (dtype){
        case FLOAT32:
            for (int i = 0; i < t->size; i++) t->data.float32[i] = (float)rand() / (float)RAND_MAX;
            break;
        case FLOAT64:
            for (int i = 0; i < t->size; i++) t->data.float64[i] = (double)rand() / (double)RAND_MAX;
            break;
        case INT32:
        case INT64:
            free(t);
            fprintf(stderr," \"randd\" not implemented for \'int\' dtype \n"); 
            return NULL;
            // break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    return t;    
}

//Memory Management
void t_free(Tensor* t){
    if(t == NULL)return;
    if (t->data.float32 != NULL) {
        free(t->data.float32);
        t->data.float32 = NULL;
    }else if (t->data.float64 != NULL) {
        free(t->data.float64);
        t->data.float64 = NULL;
    }else if (t->data.int32 != NULL) {
        free(t->data.int32);
        t->data.int32 = NULL;
    }else if (t->data.int64 != NULL) {
        free(t->data.int64);
        t->data.int64 = NULL;
    }
    if(t->grad.float32 != NULL) {
        free(t->grad.float32);
        t->grad.float32 = NULL;
    } else if (t->grad.float64 != NULL) {
        free(t->grad.float64);
        t->grad.float64 = NULL;
    }
    if (t->dims!= NULL) {
        free(t->dims);
        t->dims = NULL;
    }
    free(t);
}

void grad_init(Tensor * loss){
    if(!loss) return;
    if(loss->dtype == FLOAT32){
        if(!loss->requires_grad){
            for (int i = 0; i < loss->size; i++)
            {
                loss->grad.float32[i] = 1.0f;
            }
        }
    } else if(loss->dtype == FLOAT64){
        if(!loss->requires_grad){
            for (int i = 0; i < loss->size; i++)
            {
                loss->grad.float64[i] = 1.0;
            }
        }
    }
}

// element-wise addition
Tensor * add(Tensor * t1, Tensor * t2){
    if (!t1 || !t2) return NULL;
    if(t1->ndim != t2->ndim || t1->dtype != t2->dtype) return NULL;
    for(int i=0; i<t1->ndim; i++){
        if(t1->dims[i] != t2->dims[i]) return NULL;
    }
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = t1->data.float32[i] + t2->data.float32[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            }else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = t1->data.float64[i] + t2->data.float64[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.int32[i] = t1->data.int32[i] + t2->data.int32[i];
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.int64[i] = t1->data.int64[i] + t2->data.int64[i];
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad = (t1->requires_grad == true || t2->requires_grad == true) ? true : false;
    t->op = ADD;
    t->prevs[0] = t1;
    t->prevs[1] = t2;
    t->num_prevs = 2;
    return t;
}

void add_backward(Tensor * out){
    if(!out) return;
    if(out->prevs[0]->requires_grad==true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float32[i] += (float)1.0 * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float64[i] += (double)1.0 * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
    if(out->prevs[1]->requires_grad==true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float32[i] += (float)1.0 * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float64[i] += (double)1.0 * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
}

//element-wise subtraction
Tensor * sub(Tensor * t1, Tensor * t2){
    if(!t1 || !t2) return NULL;
    if(t1->ndim != t2->ndim || t1->dtype != t2->dtype){
        return NULL;
    }
    for(int i =0; i < t1->ndim; i++){
        if(t1->dims[i]!= t2->dims[i]) return NULL;
    }
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = t1->data.float32[i] - t2->data.float32[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = t1->data.float64[i] - t2->data.float64[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.int32[i] = t1->data.int32[i] - t2->data.int32[i];
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.int64[i] = t1->data.int64[i] - t2->data.int64[i];
            }
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad = (t1->requires_grad == true || t2->requires_grad == true) ? true : false;
    t->op = SUB;
    t->prevs[0] = t1;
    t->prevs[1] = t2;
    t->num_prevs = 2;
    return t;
}

void sub_backward(Tensor * out){
    if(!out) return;
    if(out->prevs[0]->requires_grad==true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float32[i] += (float)1.0 * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float64[i] += (double)1.0 * out->grad.float64[i];
                }
                break;           
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
    if(out->prevs[1]->requires_grad==true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float32[i] += (float)-1.0 * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float64[i] += (double)-1.0 * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
}

//element-wise multiplication
Tensor * mul(Tensor *t1, Tensor *t2){
    if(!t1 || !t2) return NULL;
    if(t1->ndim != t2->ndim || t1->dtype != t2->dtype){
        return NULL;
    }
    for(int i =0; i<t1->ndim; i++){
        if(t1->dims[i] != t2->dims[i]){
            return NULL;
        }
    }
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = t1->data.float32[i] * t2->data.float32[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = t1->data.float64[i] * t2->data.float64[i];
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.int32[i] = t1->data.int32[i] * t2->data.int32[i];
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.int64[i] = t1->data.int64[i] * t2->data.int64[i];
            }
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad = (t1->requires_grad == true || t2->requires_grad == true) ? true : false;
    t->op = MUL;
    t->prevs[0] = t1;
    t->prevs[1] = t2;
    t->num_prevs = 2;
    return t;
}

void mul_backward(Tensor * out){
    if(!out) return;
    if(out->prevs[0]->requires_grad==true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float32[i] += out->prevs[1]->data.float32[i] * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[0]->grad.float64[i] += out->prevs[1]->data.float64[i] * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
    if(out->prevs[1]->requires_grad == true){
        switch (out->dtype){
            case FLOAT32:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float32[i] += out->prevs[0]->data.float32[i] * out->grad.float32[i];
                }
                break;
            case FLOAT64:
                for(int i=0; i<out->size; i++){
                    out->prevs[1]->grad.float64[i] += out->prevs[0]->data.float64[i] * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
        }
    }
}

//dot preoduct
Tensor * matmul(Tensor *t1, Tensor *t2){
    if(!t1 || !t2) return NULL;
    int m = t1->dims[0];
    int n = t2->dims[1];
    int l = t1->dims[1];
    int dims[]=(int[]){m, n};
    if(t1->dims[1]!= t2->dims[0] || t1->dtype != t2->dtype){
        return NULL;
    }
    Tensor * t = tensor(NULL, t1->dtype, dims, t1->ndim, false);
    if(!t) return NULL;
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    float sum=0.0;
                    for(int k=0; k<l; k++){
                        sum += t1->data.float32[i*l + k] * t2->data.float32[k*n + j];
                    }
                    t->data.float32[i*n + j] = sum;
                }
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    double sum=0.0;
                    for(int k=0; k<l; k++){
                        sum += t1->data.float64[i*l + k] * t2->data.float64[k*n + j];
                    }
                    t->data.float64[i*n + j] = sum;
                }
            }
            if(!t1->requires_grad || !t2->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    int32_t sum=0;
                    for(int k=0; k<l; k++){
                        sum += t1->data.int32[i*l + k] * t2->data.int32[k*n + j];
                    }
                    t->data.int32[i*n + j] = sum;
                }
            }
            break;
        case INT64:
            for(int i=0; i<m; i++){
                for(int j=0; j<n; j++){
                    int64_t sum=0;
                    for(int k=0; k<l; k++){
                        sum += t1->data.int64[i*l + k] * t2->data.int64[k*n + j];
                    }
                    t->data.int64[i*n + j] = sum;
                }
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad = (t1->requires_grad == true || t2->requires_grad == true) ? true : false;
    t->op = MATMUL;
    t->prevs[0] = t1;
    t->prevs[1] = t2;
    t->num_prevs = 2;
    return t;
}

void matmul_backward(Tensor * out){
    if(!out) return;
    int m = out->prevs[0]->dims[0];
    int n = out->prevs[1]->dims[1];
    int l = out->prevs[1]->dims[0];
    switch(out->dtype){
        case FLOAT32:
            if(out->prevs[0]->requires_grad == true){
                for(int i=0; i<m; i++){
                    for(int j=0; j<n; j++){
                        float grad_out = out->grad.float32[i*n + j];
                        float sum=0.0;
                        for(int k=0; k<l; k++){
                            sum += out->prevs[1]->data.float32[j*l + k] * grad_out;
                        }
                        out->prevs[0]->grad.float32[i*n + j] = sum;
                    }
                }
            }
            if(out->prevs[1]->requires_grad == true){
                for(int i=0; i<m; i++){
                    for(int j=0; j<n; j++){
                        float grad_out = out->grad.float32[i*n + j];
                        float sum=0.0;
                        for(int k=0; k<l; k++){
                            sum += out->prevs[0]->data.float32[i*l + k] * grad_out;
                        }
                        out->prevs[1]->grad.float32[i*n + j] = sum;
                    }
                }
            }
            break;
        case FLOAT64:
            if(out->prevs[0]->requires_grad == true){
                for(int i=0; i<m; i++){
                    for(int j=0; j<n; j++){
                        double grad_out = out->grad.float64[i*n + j];
                        double sum=0.0;
                        for(int k=0; k<l; k++){
                            sum += out->prevs[1]->data.float64[j*l + k] * grad_out;
                        }
                        out->prevs[0]->grad.float64[i*n + j] = sum;
                    }
                }
            }
            if(out->prevs[1]->requires_grad == true){
                for(int i=0; i<m; i++){
                    for(int j=0; j<n; j++){
                        double grad_out = out->grad.float64[i*n + j];
                        double sum=0.0;
                        for(int k=0; k<l; k++){
                            sum += out->prevs[0]->data.float64[i*l + k] * grad_out;
                        }
                        out->prevs[1]->grad.float64[i*n + j] = sum;
                    }
                }
            }
            break;
        default:
            free(out);
            fprintf(stderr, "Unsupported data type \n");
            return;
    }
}

Tensor * Div( Tensor * t1, Tensor *t2){
    if(!t1 || !t2) return NULL;
    if(t1->dtype != t2->dtype || t1->ndim != t2->ndim) return NULL;
    for (int i = 0; i < t1->ndim; i++)
    {
        if(t1->dims[i]!= t2->dims[i]) return NULL;
    }

    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);

    switch (t1->dtype)
    {
    case FLOAT32:
        for (int i = 0; i < t1->size; i++)
        {
            t->data.float32[i] = t1->data.float32[i] / t2->data.float32[i];
        }
        if(!t1->requires_grad || !t2->requires_grad){
            t->grad.float32 = (float*)calloc(t1->size, sizeof(float));
            if(!t1->grad.float32){
                fprintf(stderr, "Memory allocation for grad failed\n");
                free(t->data.float32);
                free(t->dims);
                free(t);
                return NULL;
            }
        }else
        {
            t->grad.float32 = NULL;
        }
        break;
    case FLOAT64:
        for (int i = 0; i < t1->size; i++){
            t->data.float64[i] = t1->data.float64[i] / t2->data.float64[i];
        }
        if(!t1->requires_grad || !t2->requires_grad){
            t->grad.float64 = (double*)calloc(t1->size, sizeof(double));
            if(!t1->grad.float64){
                fprintf(stderr, "Memory allocation for grad failed\n");
                free(t->data.float64);
                free(t->dims);
                free(t);
                return NULL;
            }
        }else{
            t->grad.float64 = NULL;
        }
        break;
    case INT32:
        for(int i=0; i<t1->size; i++){
            t->data.float32[i] = t1->data.int32[i] / t2->data.int32[i];
        }
        if(!t1->requires_grad || !t2->requires_grad){
            fprintf(stderr, "Only Tensors of floating point and complex dtype can require gradients\n");
            free(t->data.float32);
            free(t->dims);
            free(t);
            return NULL;
        }
        break;
    case INT64:
        for(int i=0; i<t1->size; i++){
            t->data.float64[i] = t1->data.int64[i] / t2->data.int64[i];
        }
        if(!t1->requires_grad || !t2->requires_grad){
            fprintf(stderr, "Only Tensors of floating point and complex dtype can require gradients\n");
            free(t->data.float64);
            free(t->dims);
            free(t);
            return NULL;
        }
        break;
    default:
        free(t);
        fprintf(stderr, "Unsupported data type \n");
        return NULL;
        break;
    }
    t->op=DIV;
    t->num_prevs=2;
    t->prevs[0]= t1;
    t->prevs[1]= t2;
    t->requires_grad = (!t1->requires_grad || !t2->requires_grad) ? true : false;
    
    return t;
}

void Div_backward(Tensor *out){
    if(!out) return;

    switch (out->dtype)
    {
    case FLOAT32:
        if(out->prevs[0]->requires_grad==true){
            for (int i = 0; i < out->size; i++)
            {
                out->prevs[0]->grad.float32[i] += out->grad.float32[i]/out->prevs[1]->data.float32[i]; 
            }
        }
        if(out->prevs[1]->requires_grad==true){
            for (int i = 0; i < out->size; i++)
            {
                out->prevs[1]->grad.float32[i] += out->grad.float32[i] * out->prevs[0]->data.float32[i] / pow(out->prevs[1]->data.float32[i] , 2);
            }
        }
        break;

    case FLOAT64:
        if(out->prevs[0]->requires_grad==true){
            for (int i = 0; i < out->size; i++)
            {
                out->prevs[0]->grad.float64[i] += out->grad.float64[i]/out->prevs[1]->data.float64[i]; 
            }
        }
        if(out->prevs[1]->requires_grad==true){
            for (int i = 0; i < out->size; i++)
            {
                out->prevs[1]->grad.float64[i] += (out->grad.float64[i] * out->prevs[0]->data.float64[i]) / pow(out->prevs[1]->data.float64[i] , 2);
            }  
        }
        break;    
    default:
        free(out);
        fprintf(stderr, "Unsupported data type \n");
        return;
        break;
    }
}

Tensor* Pow(Tensor *t1, double exponent){
    if(!t1)return NULL;
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = powf(t1->data.float32[i], (float )exponent);
            }
            if(!t1->requires_grad ){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;

        case FLOAT64:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.float64[i] = pow(t1->data.float64[i], exponent);
            }
            if (!t1->requires_grad)
            {
                t->grad.float64 = (double*)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            }else{
                t->grad.float64 = NULL;
            }
            break;

        case INT32:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.int32[i] =(int32_t)pow((double)t1->data.int32[i], exponent);
            }
            if (!t1->requires_grad)
            {
                fprintf(stderr, "Gradient calculation not supported for integer types.\n");
                free(t->data.int32);
                free(t->dims);
                free(t);
                return NULL;
            }else{
                t->grad.float32 = NULL;
            }
            break;
        
        case INT64:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.int64[i] = (int64_t)pow((double)t1->data.int64[i], exponent);
            }
            if (!t1->requires_grad)
            {
                fprintf(stderr, "Gradient calculation not supported for integer types.\n");
                free(t->data.int64);
                free(t->dims);
                free(t);
                return NULL;
            }else{
                t->grad.float64 = NULL;
            }
            break;

        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
            break;           
    }
    t->requires_grad = (t1->requires_grad == true) ? true : false;
    t->op=POW;
    t->num_prevs=1;
    t->prevs[0]= t1;
    t->extra = exponent;

    return t;
}

void Pow_backward(Tensor * out){
    if(!out)return;
    switch (out->dtype){
        case FLOAT32:
            if(out->prevs[0]->requires_grad==true){
                for (int i = 0; i < out->size; i++)
                {
                    out->prevs[0]->grad.float32[i] += out->grad.float32[i] * (float)out->extra * powf(out->prevs[0]->data.float32[i], ((float)out->extra-1));
                }
            }
            break;

        case FLOAT64:
            if(out->prevs[0]->requires_grad==true){
                for (int i = 0; i < out->size; i++)
                {
                    out->prevs[0]->grad.float64[i] += out->grad.float64[i] * out->extra * pow(out->prevs[0]->data.float64[i], (out->extra-1));
                }
            }
            break;
        default:
        free(out);
        fprintf(stderr, "Unsupported data type \n");
        return ;
        break;
    }
}

Tensor * Exp(Tensor *t1){
    if(!t1) return NULL;
    Tensor *t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;

    switch (t1->dtype)
    {
        case FLOAT32:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.float32[i] = expf(t->data.float32[i]);
            }
            if (!t1->requires_grad)
            {
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if (!t1->grad.float32)
                {
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->grad.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
                
            }else{
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.float64[i] = exp(t->data.float64[i]);
            }
            if (!t1->requires_grad)
            {
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if (!t1->grad.float64)
                {
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->grad.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
                
            }else{
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.float32[i] = exp((float)t->data.int32[i]);
            }
            if (!t1->requires_grad)
            {
                fprintf(stderr, "Gradient calculation not supported for integer types.\n");
                free(t->data.float32);
                free(t->dims);
                free(t);
                return NULL;
                
            }else{
                t->grad.float32 = NULL;
            }
            break;
        case INT64:
            for (int i = 0; i < t1->size; i++)
            {
                t->data.float64[i] = exp((double)t->data.int64[i]);
            }
            if (!t1->requires_grad)
            {
                fprintf(stderr, "Gradient calculation not supported for integer types.\n");
                free(t->data.float64);
                free(t->dims);
                free(t);
                return NULL;
                
            }else{
                t->grad.float64 = NULL;
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
            break;           
    }
    t->requires_grad = (t1->requires_grad == true) ? true : false;
    t->op=EXP;
    t->num_prevs=1;
    t->prevs[0]= t1;

    return t;
}

void Exp_backward(Tensor * out){
    if(!out)return;
    switch (out->dtype){
        case FLOAT32:
            if(out->prevs[0]->requires_grad==true){
                for (int i = 0; i < out->size; i++)
                {
                    out->prevs[0]->grad.float32[i] += out->grad.float32[i] * expf(out->prevs[0]->data.float32[i]);
                }
            }
            break;

        case FLOAT64:
            if(out->prevs[0]->requires_grad==true){
                for (int i = 0; i < out->size; i++)
                {
                    out->prevs[0]->grad.float64[i] += out->grad.float64[i] * exp(out->prevs[0]->data.float64[i]);
                }
            }
            break;
        default:
        free(out);
        fprintf(stderr, "Unsupported data type \n");
        return ;
        break;
    }
}

Tensor * relu(Tensor *t1){
    if(!t1) return NULL;
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = (t1->data.float32[i]<0) ? 0 : (t1->data.float32[i]);
            }
            if(!t1->requires_grad ){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = (t1->data.float64[i]<0) ? 0 : (t1->data.float64[i]);
            }
            if(!t1->requires_grad ){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.int32[i] = (t1->data.int32[i]<0) ? 0 : (t1->data.int32[i]);
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.int64[i] = (t1->data.int64[i]<0) ? 0 : (t1->data.int64[i]);
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad = (t1->requires_grad == true) ? true : false;
    t->op=RELU;
    t->prevs[0]= t1;
    t->num_prevs = 1;
    return t;
}

void relu_backward(Tensor * out){
    if(!out) return;
    if(out->prevs[0]->requires_grad == true){
        switch(out->dtype){
        case FLOAT32:
            for(int i=0; i<out->size; i++){
                out->prevs[0]->grad.float32[i] += (out->data.float32[i] > 0) ? 0 : (out->grad.float32[i]);
            }
            break;
        case FLOAT64:
            for(int i=0; i<out->size; i++){
                out->prevs[0]->grad.float64[i] += (out->data.float64[i] > 0) ? 0 : (out->grad.float64[i]);
            }
            break;
        default:
            free(out);
            fprintf(stderr, "Unsupported data type \n");
            return;
        }
    }
}

Tensor * leaky_relu(double negative_slope, Tensor *t1){
    if(!t1) return NULL;
    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;
    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = (t1->data.float32[i]<0) ? ((float)negative_slope * t1->data.float32[i]) : (t1->data.float32[i]);
            }
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = (t1->data.float64[i]<0) ? (negative_slope * t1->data.float64[i]) : (t1->data.float64[i]);
            }
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            free(t);
            fprintf(stderr, " \"leaky_relu\" not implemented for 'int32' \n");
            return NULL;
            break;
        case INT64:
            free(t);
            fprintf(stderr, " \"leaky_relu\" not implemented for 'int64' \n");
            return NULL;
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->requires_grad =(t1->requires_grad == true) ? true : false;
    t->op=LEAKY_RELU;
    t->prevs[0] = t1;
    t->extra = negative_slope;
    t->num_prevs = 1;
    return t;
}

void leaky_relu_backward(Tensor * out){
    if(!out) return;

    if(out->prevs[0]->requires_grad == true){
        switch(out->dtype){
        case FLOAT32:
            for(int i=0; i<out->size; i++){
                out->prevs[0]->grad.float32[i] += (out->data.float32[i]>0) ? ((float)out->extra * out->grad.float32[i]) : (out->grad.float32[i]);
            }
            break;

        case FLOAT64:
            for(int i=0; i<out->size; i++){
                out->prevs[0]->grad.float64[i] += (out->data.float64[i]>0) ? (out->extra * out->grad.float64[i]) : (out->grad.float64[i]);
            }
            break;

        default:
            free(out);
            fprintf(stderr, "Unsupported data type \n");
            return;
        }
    }

}

Tensor * Tanh(Tensor * t1){
    if(!t1) return NULL;

    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;

    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = (exp(2*t1->data.float32[i]) - 1) / (exp(2*t1->data.float32[i]) + 1);
            }
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = (exp(2*t1->data.float64[i]) - 1) / (exp(2*t1->data.float64[i]) + 1);
            }
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = (exp(2*t1->data.int32[i]) - 1) / (1 + exp(2*t1->data.int32[i]) + 1);
            }
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = (exp(2*t1->data.int64[i]) - 1) / (exp(2*t1->data.int64[i]) + 1);
            }
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;

        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }

    t->requires_grad= (t1->requires_grad == true) ? true : false;
    t->prevs[0]=t1;
    t->op=TANH;
    t->num_prevs = 1;

    return t;
}

void Tanh_backward(Tensor * out){
    if(!out) return;

    if(out->prevs[0]->requires_grad==true){
        switch(out->dtype){
        case FLOAT32:
            for(int i=0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float32[i] += (1 - pow(out->data.float32[i], 2)) * out->grad.float32[i];
            }
            break;
        case FLOAT64:
            for(int i=0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float64[i] += (1 - pow(out->data.float64[i], 2)) * out->grad.float64[i];
            }
            break;
        case INT32:
            for(int i=0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float32[i] += (1 - pow(out->data.int32[i], 2)) * out->grad.float32[i];
            }
            break;
        case INT64:
            for(int i=0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float64[i] += (1 - pow(out->data.int64[i], 2)) * out->grad.float64[i];
            }
            break;

        default:
            free(out);
            fprintf(stderr, "Unsupported data type \n");
            return;
        }
    }
}

Tensor * Sigmoid(Tensor * t1){
    if(!t1) return NULL;

    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;

    switch(t1->dtype){
        case FLOAT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = 1 / (1 + exp(-t1->data.float32[i]));
            }
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = 1 / (1 + exp(-t1->data.float64[i]));
            }
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i=0; i<t1->size; i++){
                t->data.float32[i] = 1 / (1 + exp(-t1->data.int32[i]));
            }
            break;
        case INT64:
            for(int i=0; i<t1->size; i++){
                t->data.float64[i] = 1 / (1 + exp(-t1->data.int64[i]));
            }
            break;
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }

    t->requires_grad = (t1->requires_grad == true) ? true : false;
    t->op = SIGMOID;
    t->prevs[0] = t1;
    t->num_prevs = 1;

    return t;
}

void Sigmoid_backward(Tensor * out){
    if(!out) return;

    if(out->prevs[0]->requires_grad == true){
        switch(out->prevs[0]->dtype){
            case FLOAT32:
            case INT32:
                for(int i=0; i<out->prevs[0]->size; i++){
                    out->prevs[0]->grad.float32[i] += out->data.float32[i] * (1 - out->data.float32[i]) * out->grad.float32[i];
                }
                break;
            case FLOAT64:
            case INT64:
                for(int i=0; i<out->prevs[0]->size; i++){
                    out->prevs[0]->grad.float64[i] += out->data.float64[i] * (1 - out->data.float64[i]) * out->grad.float64[i];
                }
                break;
            default:
                free(out);
                fprintf(stderr, "Unsupported data type \n");
                return;
            }
    }
}

Tensor * softmax(Tensor *t1){
    if(!t1) return NULL;

    Tensor * t = tensor(NULL, t1->dtype, t1->dims, t1->ndim, false);
    if(!t) return NULL;

    float s_float = 0.0f;
    float max_val_float;
    double s_double = 0.0;
    double max_val_double;
    float *ex_float = (float *)malloc(t1->size * sizeof(float));
    double *ex_double = (double *)malloc(t1->size * sizeof(double));

    switch(t1->dtype){
        case FLOAT32:
            if(!ex_float){
                free(t);
                fprintf(stderr, "Memory allocation failed \n");
                return NULL;
            }

            max_val_float = t1->data.float32[0];
            for(int i = 1; i < t1->size; i++) {
                if(t1->data.float32[i] > max_val_float) {
                    max_val_float = t1->data.float32[i];
                }
            }

            for( int i = 0; i<t1->size; i++){
                ex_float[i] = expf(t1->data.float32[i] - max_val_float);
                s_float += ex_float[i];
            }

            for(int i = 0; i<t1->size; i++){
                t->data.float32[i] = ex_float[i] / s_float;
            }
            free(ex_float);
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;

        case FLOAT64:
            if(!ex_double){
                free(t);
                fprintf(stderr, "Memory allocation failed \n");
                return NULL;
            }

            max_val_double = t1->data.float64[0];
            for(int i = 1; i < t1->size; i++) {
                if(t1->data.float64[i] > max_val_double) {
                    max_val_double = t1->data.float64[i];
                }
            }

            for( int i = 0; i<t1->size;i++){
                ex_double[i] = exp(t1->data.float64[i] - max_val_double);
                s_double += ex_double[i];
            }

            for(int i = 0; i<t1->size; i++){
                t->data.float64[i] = ex_double[i] / s_double;
            }
            free(ex_double);
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;

        case INT32:
            free(t);
            fprintf(stderr, " \"softmax\" not implemented for 'int32' \n");
            return NULL;
            break;

        case INT64:
            free(t);
            fprintf(stderr, " \"softmax\" not implemented for 'int64' \n");
            return NULL;
            break;

        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }

    t->op=SOFTMAX;
    t->prevs[0] = t1;
    t->num_prevs = 1;
    t->requires_grad = (t1->requires_grad == true) ? true : false;

    return t;
}

void softmax_backward(Tensor *out){
    if(!out) return;

    float gradsum_float = 0.0f;
    double gradsum_double = 0.0;

    switch(out->prevs[0]->dtype){
        case FLOAT32:
            for(int i = 1; i < out->prevs[0]->size; i++) {
                gradsum_float += out->grad.float32[i];
            }

            for( int i = 0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float32[i] += out->grad.float32[i] - expf(out->data.float32[i]) * gradsum_float;
            }
            break;

        case FLOAT64:
            for(int i = 1; i < out->prevs[0]->size; i++) {
                gradsum_double += out->grad.float64[i];
            }

            for( int i = 0; i<out->prevs[0]->size; i++){
                out->prevs[0]->grad.float64[i] += out->grad.float64[i] - expf(out->data.float64[i]) * gradsum_double;
            }
            break;

        default:
            free(out);
            fprintf(stderr, "Unsupported data type \n");
            return;
    }
}

Tensor * sum(Tensor * t1){
    if(!t1) return NULL;
    Tensor *t=tensor(NULL, t1->dtype, (int[]){1}, 1, false);

    switch(t1->dtype){
        case FLOAT32:
            for(int i = 0; i<t1->size; i++){
                t->data.float32[0] += t1->data.float32[i];
            }
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i = 0; i<t1->size; i++){
                t->data.float64[0] += t1->data.float64[i];
            }
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            for(int i = 0; i<t1->size; i++){
                t->data.int32[0] += t1->data.int32[i];
            }
            break;
        case INT64:
            for(int i = 0; i<t1->size; i++){
                t->data.int64[0] += t1->data.int64[i];
            }
        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }

    t->op=SUM;
    t->prevs[0] = t1;
    t->requires_grad = (t1->requires_grad == true) ? true : false;

    return t;
}

Tensor * mean(Tensor * t1){
    if(!t1) return NULL;

    Tensor *t = tensor(NULL, t1->dtype, (int[]){1}, 1, false);

    switch(t1->dtype){
        case FLOAT32:
            for(int i = 0; i<t1->size; i++){
                t->data.float32[0] += t1->data.float32[i];
            }
            t->data.float32[0] = t->data.float32[0]/t1->size;
            if(!t1->requires_grad){
                t->grad.float32 = (float *)calloc(t->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float32 = NULL;
            }
            break;
        case FLOAT64:
            for(int i = 0; i<t1->size; i++){
                t->data.float64[0] += t1->data.float64[i];
            }
            t->data.float64[0] = t->data.float64[0]/t1->size;
            if(!t1->requires_grad){
                t->grad.float64 = (double *)calloc(t->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            } else {
                t->grad.float64 = NULL;
            }
            break;
        case INT32:
            free(t);
            fprintf(stderr, " mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: int32 \n");
            return NULL;
            break;

        case INT64:
            free(t);
            fprintf(stderr, " mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: int64 \n");
            return NULL;
            break;

        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }

    t->op=MEAN;
    t->prevs[0] = t1;
    t->num_prevs =1;
    t->requires_grad = (t1->requires_grad == true) ? true : false;

    return t;
}

void mean_backward(Tensor * out){
    if(!out) return;
    
    if(out->prevs[0]->requires_grad == true){
        switch(out->dtype){
            case FLOAT32:
                // Ensure `out->grad` and `out->prevs[0]->grad` are allocated
                if (!out->grad.float32 || !out->prevs[0]->grad.float32) {
                    fprintf(stderr, "Gradient memory not allocated\n");
                    return;
                }
                for(int i=0; i < out->prevs[0]->size; i++){
                    out->prevs[0]->grad.float32[i] += out->grad.float32[0] / out->prevs[0]->size;
                }
                break;
            case FLOAT64:
                // Ensure `out->grad` and `out->prevs[0]->grad` are allocated
                if (!out->grad.float64 || !out->prevs[0]->grad.float64) {
                    fprintf(stderr, "Gradient memory not allocated\n");
                    return;
                }
                for(int i=0; i < out->prevs[0]->size; i++){
                    out->prevs[0]->grad.float64[i] += out->grad.float64[0] / out->prevs[0]->size;
                }
                break;
            default:
                fprintf(stderr, "Unsupported data type \n");
                break;
        }
    }
}

Tensor *MSELoss(Tensor * yTrue, Tensor * yPred){
    if(!yTrue ||!yPred) return NULL;
    if(yTrue->ndim != yPred->ndim || yPred->dtype != yTrue->dtype) return NULL;
    for (int i = 0; i < yPred->ndim; i++)
    {
        if(yTrue->dims[i] != yPred->dims[i]) return NULL;
    }

    Tensor *t=tensor(NULL, yPred->dtype, (int[]){1}, 1, false);
    if(!t){
        fprintf(stderr, "Memory allocation for MSE tensor failed\n");
        return NULL;
    }
    
    switch (yPred->dtype)
    {
        case FLOAT32:
            for (int i = 0; i < yPred->size; i++)
            {
                t->data.float32[0] += powf((yTrue->data.float32[i] - yPred->data.float32[i]), 2.0f);
            }
            t->data.float32[0] /= yPred->size;
            
            if(!yPred->requires_grad){
                t->grad.float32 = (float *) calloc(yPred->size, sizeof(float));
                if(!t->grad.float32){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float32);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            }else{
                    t->grad.float32 =NULL;
                }
            break;
        case FLOAT64:
            for (int i = 0; i < yPred->size; i++)
            {
                t->data.float64[0] += pow((yTrue->data.float64[i] - yPred->data.float64[i]), (double)2.0);
            }
            t->data.float64[0] /= yPred->size;
            
            if(!yPred->requires_grad){
                t->grad.float64 = (double*)calloc(yPred->size, sizeof(double));
                if(!t->grad.float64){
                    fprintf(stderr, "Memory allocation for grad failed\n");
                    free(t->data.float64);
                    free(t->dims);
                    free(t);
                    return NULL;
                }
            }else{
                    t->grad.float64 =NULL;
                }
            break;

        case INT32:
            free(t);
            fprintf(stderr, " RuntimeError: \"mse_cpu\" not implemented for 'Int' \n");
            return NULL;
            break;

        case INT64:
            free(t);
            fprintf(stderr, " RuntimeError: \"mse_cpu\" not implemented for 'Int' \n");
            return NULL;
            break;

        default:
            free(t);
            fprintf(stderr, "Unsupported data type \n");
            return NULL;
    }
    t->op=MSE;
    t->prevs[0] = yPred;
    t->prevs[1] = yTrue;
    t->num_prevs =2;
    t->requires_grad = (!yPred->requires_grad) ? true : false;
    return t;
}

void MSELoss_backward(Tensor * out){
    if(!out) return;

    if(out->prevs[0]->requires_grad == true){
        switch (out->dtype)
        {
        case FLOAT32:
            for (int i = 0; i < out->prevs[0]->size; i++)
            {
                out->prevs[0]->grad.float32[i] += (float)(-2/out->prevs[0]->size) * (out->prevs[1]->data.float32[i] - out->prevs[0]->data.float32[i]) * out->grad.float32[i];
            }
            break;
        case FLOAT64:
            for (int i = 0; i < out->prevs[0]->size; i++)
            {
                out->prevs[0]->grad.float64[i] += (double)(-2/out->prevs[0]->size) * (out->prevs[1]->data.float64[i] - out->prevs[0]->data.float64[i]) * out->grad.float64[i];
            }
            break;
        
        default:
            fprintf(stderr, "Unsupported data type \n");
            // free(out);
            break;
        }
    }
}

void backward(Tensor * t){
    //check if loss is NULL
    if(!t) return;

    if(t->op == MUL){
        mul_backward(t);
    }else if(t->op == ADD){
        add_backward(t);
    }else if(t->op == SUB){
        sub_backward(t);
    }else if(t->op == MATMUL){
        matmul_backward(t);
    }else if(t->op == MEAN){
        mean_backward(t);
    }else if(t->op == RELU){
        relu_backward(t);
    }else if(t->op == LEAKY_RELU){
        leaky_relu_backward(t);
    }else if(t->op == TANH){
        Tanh_backward(t);
    }else if(t->op == SIGMOID){
        Sigmoid_backward(t);
    }else if(t->op == SOFTMAX){
        softmax_backward(t);
    }else if(t->op ==POW){
        Pow_backward(t);
    }else if(t->op == EXP){
        Exp_backward(t);
    }else if(t->op == DIV){
        Div_backward(t);
    }else if(t->op == MSE){
        MSELoss_backward(t);
    }

    for(int i=0; i<t->num_prevs; i++){
        backward(t->prevs[i]);
    }
}

// print data
void print(Tensor* t){
    if(!t) return;

    printf("Tensor {\n");
    printf("  dtype: ");
    switch(t->dtype){
        case FLOAT32: printf("float32\n"); break;
        case FLOAT64: printf("float64\n"); break;
        case INT32: printf("int32\n"); break;
        case INT64: printf("int64\n"); break;
        default: printf("unknown\n"); break;
    }

    printf("  dims:  [");
    for(int i=0; i<t->ndim; i++){
        printf("%d%s", t->dims[i], i<t->ndim-1 ? ", " : "");
    }
    printf("]\n");
    printf("  data:  [");
    for(int i=0; i<t->size && i<10; i++){
        switch (t->dtype){
            case FLOAT32: printf("%.4f", t->data.float32[i]); break;
            case FLOAT64: printf("%.4lf", t->data.float64[i]); break;
            case INT32: printf("%d", t->data.int32[i]); break;
            case INT64: printf("%lld", t->data.int64[i]); break;
            default: printf("Unsupported type"); break;
        }
        printf("%s", i < t->size-1 ? ", " : "");
        if(i == 9 && t->size > 10){
            printf("...");
            break;
        }
    }
    printf("]\n");
    if(!t->requires_grad){
        printf("  grads: [");
        if (t->grad.float32){
            for(int i=0; i<t->size && i<10; i++){
                printf("%.4f", t->grad.float32[i]);
                printf("%s", i < t->size-1? ", " : "");
                if(i == 9 && t->size > 10){
                    printf("...");
                    break;
                }
            }
        }else if (t->grad.float64){
            // int size = sizeof(t->grad.float64)/sizeof(t->grad.float64[0]);
            size_t size=sizeof(t->grad.float64);
            for(int i=0; i<size && i<10; i++){
                printf("%.4lf", t->grad.float64[i]);
                printf("%s", i < size-1? ", " : "");
                if(i == 9 && size > 10){
                    printf("...");
                    break;
                }
            }
        }
    printf("]\n");
    }else {
            printf("  grads:  ");
            printf("None\n");
        }
    printf("}\n");
}