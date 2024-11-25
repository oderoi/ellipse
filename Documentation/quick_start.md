# Quick Start Guide

This guide assumes you have no prior knowledge of C, C++, PyTorch or any other language or deelearning framework.

First you have to download nan library from GitHub.

```C
git clone https://github.com/oderoi/nanoTorch.git
```

Then create a working directory in your machine/Laptop folder, open **nanoTorch** and copy `torch.h` and pest it in your folder you created. Then create a `new_file` that you will use to code your project.

Then open the `new_file` in your fevorite code editor. and start to code.

Start by importing the library.

```c
#inlude "torch.h"
```

Now you are set.

## Tensor

Tensors are the base data structure in **nan**. They are like multi-dimensional array of a specific data type. All higher-level operations in **nan** operate on these tensors.

Tensor can be created from existing data.

```c
Tensor * t1 = tensor((float[]){1,2,3,4,5,6}, FLOAT64, (int[]){2,3}, 2, true);
```