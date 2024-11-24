<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/imgs/nan/16.svg">
  <img alt="tiny corp logo" src="/imgs/nan/nan.svg" width="50%" height="50%">
</picture>

nano: Something between [PyTorch](https://github.com/pytorch/pytorch) , [karpathy/micrograd](https://github.com/karpathy/micrograd) and [XLA](https://openxla.org/xla). Maintained by [nano corp](https://github.com/oderoi/nanoTorch/tree/main).

<h3>

[Home Page](README.md) | [Progress and Roadmap](Progress_and_Roadmap.md)

</h3>

</div>

---

<h1 align='center'><b>nan Documentation</b></h1>

Welcome to the nan documentation. This page is for those who are really want to make a change in AI, if it is you, you are welcome.

To get this library in your local machine, you can download it from GitHub. See...

```bash
git clone https://github.com/oderoi/nanoTorch.git 
```

This library is created in C and it has no frontend yet, so you will use C to use it.

# nan Usage

The one thing you will need to import is **torch.h** header.

```C
#include  "torch.h"
```
In C we don't use `import` like in Python, we use `#include`.

Amaizing enough `torch.h` header is the library in itself and it is just a single file. It contain functions to help you perform math operations for machine leaning and automatic differentiation capabilities.

For now **nan** library operations are not lazy but Backpropagation is lazy, meaning it won't do backward pass operations until you realize.

* **nan** has **AOT** support, so it run very close to hardware to achieve high performance, high speed and it give's you more cotrol.
* **nan** support **CPU** only for now. But it will support **GPUs** and **TPUs**. 

## **nan** Stack

|Library |Core Language|Kernel Layer|	Assembly/Hardware Layer|
|--------|-------------|------------|-----------------------|
|PyTorch|Python + C++|	ATen|	SIMD/AVX/CUDA/TPU instructions|
|TensorFlow|	Python + C++|	XLA Kernels|	LLVM-generated assembly, GPU, TPU|
TinyGrad|	Python|	Numpy/Custom Ops|	CPU SIMD, CUDA for GPU|
|Nan   | C         |  nan   | nan  |

* **nan** stack combines Kernel Layer and Assembly/Hardware Layer to make it more simple to improve, read and improve for anyone interested.
* **nan** Assembly/Hardware Layer only supports **CPU** for now.