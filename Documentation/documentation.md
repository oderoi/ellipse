<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/imgs/ellipse.png">
  <img alt="ellipse logo" src="/imgs/ellipse.png" width="50%" height="50%">
</picture>

**Ellipse**: Something between [PyTorch](https://github.com/pytorch/pytorch) , [karpathy/micrograd](https://github.com/karpathy/micrograd) and [XLA](https://openxla.org/xla). Maintained by [nileAGI](https://www.nileagi.com/).

<h3>

[Home Page](../README.md) | [Progress and Roadmap](../Progress_and_Roadmap.md)

</h3>

[**Announcement Blog Post**](https://www.nileagi.com/blog/ellipse-introduction)

</div>

---

<h1 align='center'><b>Ellipse Documentation</b></h1>

Welcome to the **ellipse** documentation. This page is for those who are really want to make a change in AI, if it is you, you are welcome.

To get this library in your local machine, you can download it from GitHub. See...

```bash
git clone  https://github.com/oderoi/ellipse.git
```

This library is created in C and it has no frontend yet, so you will use C to use it.

# ellipse Usage

The one thing you will need to import is **ellipse.h** header.

```C
#include  "ellipse.h"
```
In C we don't use `import` like in Python, we use `#include`.

Amaizing enough `ellipse.h` header is the library in itself and it is just a single file. It contain functions to help you perform math operations for machine leaning and automatic differentiation capabilities.

For now **ellipse** library operations are not lazy but Backpropagation is lazy, meaning it won't do backward pass operations until you realize.

* **ellipse** has **AOT** support, so it run very close to hardware to achieve high performance, high speed and it give's you more cotrol.
* **ellipse** support **CPU** only for now. But it will support **GPUs** and **TPUs**. 

## **nan** Stack

|Library |Core Language|Kernel Layer|	Assembly/Hardware Layer|
|--------|-------------|------------|-----------------------|
|PyTorch|Python + C++|	ATen|	SIMD/AVX/CUDA/TPU instructions|
|TensorFlow|	Python + C++|	XLA Kernels|	LLVM-generated assembly, GPU, TPU|
TinyGrad|	Python|	Numpy/Custom Ops|	CPU SIMD, CUDA for GPU|
|ellipse   | C         |  ellipse   | ellipse  |

* **ellipse** stack combines Kernel Layer and Assembly/Hardware Layer to make it more simple to improve, read and improve for anyone interested.
* **ellipse** Assembly/Hardware Layer only supports **CPU** for now.

<h3 align="center">

[Quick Start](./quick_start.md)

</h3>
