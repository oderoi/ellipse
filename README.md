<h1 align='center'><b>nanoTorch</b></h1>
<p align='center'>
    A PyTorch like AI Engine from scratch in C programming language
</p>

<p align="center">
  <img src="imgs/cerebrix.png" alt="Dainemo Logo" width="150"/>
</p>

<p>
NanoTorch is a lightweight, minimalistic deep learning library written in pure       C, designed to bring essential neural network functionalities to low-resource environments. Inspired by projects like tinygrad, NanoTorch aims to provide a foundational toolkit for machine learning enthusiasts, embedded developers, and researchers who want to experiment with deep learning concepts in an efficient, resource-conscious manner.
</p>

#### Key Features

	•Lightweight Design: Focused on simplicity, NanoTorch provides core deep learning operations without heavy dependencies.
	•Pure C Implementation: Built entirely in C, NanoTorch is designed to be portable and optimized for low-level manipulation.
	•Gradient Calculation: Includes basic automatic differentiation to support backpropagation for training models.
	•Flexible Tensor Operations: Supports fundamental tensor operations required for deep learning.
	•Modular Architecture: Easy to extend or modify, allowing you to explore and experiment with new layers, optimizers, and more.

#### Who is This For?

NanoTorch is perfect for those looking to:
	• Understand the inner workings of a deep learning library from the ground up.
	• Run simple neural networks in resource-limited environments.
	• Prototype and test custom ML operations in C.

#### Getting Started

This repository includes setup instructions, usage examples, and documentation to help you dive into developing with NanoTorch. Explore the source code to understand how core deep learning concepts like tensor operations and automatic differentiation are implemented.

#### Contributing

NanoTorch is open for contributions! Whether you’re fixing bugs, adding new features, or experimenting with optimizations, we welcome your input.


# Progress

❌: Not implemented  
✅: Done

### Operators

| Task       | Status |
|------------|--------|
| ADD        |   ✅   |
| SUB        |   ✅   |
| MUL        |   ✅   |
| DIV        |   ✅   |
| MATMUL     |   ✅   |
| EXP        |   ❌   |
| LOG        |   ❌   |
| POW        |   ❌   |
| SUM        |   ✅   |
| TRANSPOSE  |   ❌   |
| FLATTEN    |   ❌   |
| RESHAPE    |   ❌   |
| CONV2D     |   ❌   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ❌   |
| MAXPOOL3D  |   ❌   |

### Operators Derivative

| Task       | Status |
|------------|--------|
| ADD_BACKWARD        |   ✅   |
| SUB_BACKWARD        |   ✅   |
| MUL_BACKWARD        |   ✅   |
| DIV_BACKWARD        |   ❌   |
| MATMUL_BACKWARD     |   ✅   |
| EXP_BACKWARD        |   ❌   |
| LOG_BACKWARD        |   ❌   |
| POW_BACKWARD        |   ❌   |
| SUM_BACKWARD        |   ✅   |

### Activations

| Task      | Status |
|-----------|--------|
| RELU      |   ✅   |
| SIGMOID   |   ✅   |
| TANH      |   ✅   |
| SOFTMAX   |   ✅   |
| LEAKY_RELU|   ✅   |
| MEAN      |   ✅   |

### Activations Derivative

| Task      | Status |
|-----------|--------|
| RELU_BACKWARD      |   ✅   |
| SIGMOID_BACKWARD   |   ✅   |
| TANH_BACKWARD      |   ✅   |
| SOFTMAX_BACKWARD   |   ✅   |
| LEAKY_RELU_BACKWARD|   ✅   |
| MEAN_BACKWARD      |   ✅   |


### BackPropagation

| Task       | Status |
|------------|--------|
| Backward   |   ✅   |

### Loss Functions

| Task      | Status |
|-----------|--------|
| MSE       |   ❌   |
| CE        |   ❌   |
| BCE       |   ❌   |
| SoftmaxCE |   ❌   |

### Optimizers

| Task  | Status |
|-------|--------|
| ADAM  |   ❌   |

### Layers

| Task       | Status |
|------------|--------|
| SEQUENTIAL |   ❌   |
| LINEAR     |   ❌   |
| DROPOUT    |   ❌   |
| CONV2D     |   ❌   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ❌   |
| MAXPOOL3D  |   ❌   |
