<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/imgs/ellipse.png">
  <img alt="ellipse logo" src="/imgs/ellipse.png" width="50%" height="50%">
</picture>

**Ellipse**: Something between [PyTorch](https://github.com/pytorch/pytorch) , [karpathy/micrograd](https://github.com/karpathy/micrograd) and [XLA](https://openxla.org/xla). Maintained by [nileAGI](https://www.nileagi.com/).

<h3>

[Home Page](../README.md) | [Documentation](../Documentation/documentation.md)

</h3>

[**Announcement Blog Post**](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---


# Quick Start Guide

This guide assumes you have no prior knowledge of C, C++, PyTorch or any other language or deelearning framework.

First you have to download nan library from GitHub.

```bash
git clone https://github.com/oderoi/ellipse.git
```

Then create a working directory in your machine/Laptop folder, open **ellipse** and copy `ellipse.h` and pest it in your folder you created. Then create a `new_file` that you will use to code your project.

Then open the `new_file` in your fevorite code editor. and start to code.

Start by importing the library.

```c
#inlude "ellipse.h"
```

Now you are set.

## Tensor

Tensors are the base data structure in **ellipse**. They are like multi-dimensional array of a specific data type. All higher-level operations in **ellipse** operate on these tensors.

Tensor can be created from existing data.

```c
Tensor * t1 = tensor((float[]){1, 2, 3, 4, 5, 6}, FLOAT32, (int[]){2, 3}, true);
```

So you might being asking yourself what the heck is that, don't sweet , let's see what the hell is that, and will start by looking one part after the other.

* `Tensor * t1`: this is the variable definition where.
    * `Tensor * `: telling our program that this is the `Tensor` type multidimensional array like.
    * `t1`: is the name of our data, you can choose to use any name you like.
* `tensor()`: is the function that helping us to hold together all the important information about out array, like data, data type,data dimension, data rank and requires grad.
    * `(float[]){1, 2, 3, 4, 5, 6}`:this is the data itself that `Tensor * t1` carries and `(float[])` is the data type of the array, while `{1, 2, 3, 4, 5, 6}` is the arry itself.
    * `FLOAT32`: this is the Tensor data type, which is suppossed to be the same as `(float[])`.
    * `(int[]){2, 3}`: is the array dimension. As well here `(int[])` is the data type of the dimension array ( and Yes, I just say array again, because array data and dimension are both representade using array) and `{2, 3}` is the dimension itself.
    <!-- * `2`: this number represent rank of our Tensor, simply put, if dimesion is of two dimensions like `{2, 3}` then rank will be 2 aswell and if dimension if just one dimension like `{3}` then the rank will be `1`. -->
    * `true`: this boolean tell us that this `Tensor` will carry gradients of it's variable, during back propagation. So it might be `true` or `false`, depending on weither you want to calculate gradiant correspond of the `Tensor`.

### Tensor for different data types

**int**

```c
Tensor *x = tensor((int[]){1,2,3,4}, INT, (int[]){2, 2}, false);

print(x);
```

*Run*
```bash
gcc nameOfFile.c -lm

./a.out
```

*output*
```bash
Tensor {
  dtype: int
  dims:  [2, 2]
  data:  [[1, 2],
	  [3, 4]]
}
```

**float32**

```c
Tensor *y = tensor((float[]){1,2,3,4,5,6}, FLOAT32, (int[]){2, 2}, true);

print(y);
```

*Run*
```bash
gcc nameOfFile.c -lm

./a.out
```

*output*
```basha
Tensor {
  dtype: float32
  dims:  [3, 2]
  data:  [[1.0000, 2.0000],
	  [3.0000, 4.0000],
	  [5.0000, 6.0000]]

  grads: [[0.0000e+00, 0.0000e+00],
	  [0.0000e+00, 0.0000e+00],
	  [0.0000e+00, 0.0000e+00]]
}
```

**float64 / double**

```c
Tensor *x = tensor((double[]){1,2,3,4,5,6,7,8}, FLOAT64, (int[]){2, 4}, true);

print(x);
```

*Run*
```bash
gcc nameOfFile.c -lm

./a.out
```

*output*
```bash
Tensor {
  dtype: float64
  dims:  [2, 4]
  data:  [[1.0000, 2.0000, 3.0000, 4.0000],
	  [5.0000, 6.0000, 7.0000, 8.0000]]

  grads: [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
	  [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]
}
```
