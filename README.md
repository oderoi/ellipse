<!-- <h1 align='center'><b>nano</b></h1> -->
<h1 align="center">
  <img src="imgs/nan/nan.svg" alt="Dainemo Logo" width="300" height="200"/>
</h1>
<p align='center'>
    A PyTorch like AI Engine from scratch in C programming language
</p>
<!-- <p align="center">
  <img src="imgs/cerebrix.png" alt="Dainemo Logo" width="150"/>
</p> -->

<p>
NanoTorch is a lightweight, minimalistic deep learning library written in pure       C, designed to bring essential neural network functionalities to low-resource environments. Inspired by projects like tinygrad, NanoTorch aims to provide a foundational toolkit for machine learning enthusiasts, embedded developers, and researchers who want to experiment with deep learning concepts in an efficient, resource-conscious manner.
</p>

#### Key Features

-	Lightweight Design: Focused on simplicity, NanoTorch provides core deep learning operations without heavy dependencies.
-	Pure C Implementation: Built entirely in C, NanoTorch is designed to be portable and optimized for low-level manipulation.
-	 Gradient Calculation: Includes basic automatic differentiation to support backpropagation for training models.
-	 Flexible Tensor Operations: Supports fundamental tensor operations required for deep learning.
-	 Modular Architecture: Easy to extend or modify, allowing you to explore and experiment with new layers, optimizers, and more.

#### Who is This For?

NanoTorch is perfect for those looking to:
-	 Understand the inner workings of a deep learning library from the ground up.
-	 Run simple neural networks in resource-limited environments.
-	 Prototype and test custom ML operations in C.

#### Getting Started

This repository includes setup instructions, usage examples, and documentation to help you dive into developing with NanoTorch. Explore the source code to understand how core deep learning concepts like tensor operations and automatic differentiation are implemented.

#### Contributing

NanoTorch is open for contributions! Whether you’re fixing bugs, adding new features, or experimenting with optimizations, we welcome your input.


# Progress and Roadmap

NanoTorch is designed to provide an accessible, low-level deep learning framework with a focus on simplicity and modularity. Here’s a roadmap showcasing its primary components and future progress milestones:

## Core Components

❌: Not implemented  
✅: Done


1.	Tensor Operations

-    Tensor Creation and Manipulation: Support for tensor creation with various data types (float, double, int) and shapes.

| Task       | Status |
|------------|--------|
| Tensor     |   ✅   |


-    Basic Tensor Math:

<!-- | Task       | Status |
|------------|--------|
| ADD        |   ✅   |
| SUB        |   ✅   |
| MUL        |   ✅   |
| DIV        |   ✅   |
| MATMUL     |   ✅   |
| EXP        |   ✅   |
| LOG        |   ❌   |
| POW        |   ✅   |
| SUM        |   ✅   |
| TRANSPOSE  |   ❌   |
| FLATTEN    |   ❌   |
| RESHAPE    |   ❌   |
| CONV2D     |   ❌   |
| CONV3D     |   ❌   |
| MAXPOOL2D  |   ❌   |
| MAXPOOL3D  |   ❌   | -->

| Operation          | Formula                                                                   | Status |
|--------------------|---------------------------------------------------------------------------|--------|
| Addition           | $C_{i,j} = A_{i,j} + B_{i,j}$                                             |   ✅   |
| Subtraction        | $C_{i,j} = A_{i,j} - B_{i,j}$                                             |   ✅   |
| Maltiplication     | $C_{i,j} = A_{i,j} * B_{i,j}$                                             |   ✅   |
| Division           | $C_{i,j} = A_{i,j} / B_{i,j}$                                             |   ✅   |
| Dot_Product        | $C_{i,j} = \sum_{k=0}^{k-1}\(A_{i,k} \cdot B_{k,j}\)$                     |   ✅   |
| Exponent           | $C_{i,j} = e^{x_{i,j}}$                                                   |   ✅   |
| Logarithm          | $C_{i,j} = \log_{10}(X_{i,j})$                                            |   ❌   |
| Power              | $C_{i,j} = (\mathbf{A}^p){i,j} = (\mathbf{A}{i,j})^n$                     |   ✅   |
| Sum                | $\mathbf{C}   = \sum_{i=0}^{i-1}\(X_{i}\)$                                |   ✅   |
| Transpose          | $(\mathbf{A}^\top){i,j} = (\mathbf{A}){j,i}$                              |   ✅   |
| Flatten            | $\text{Flatten}(A_{m,n}) = \[A_{0,0},\  A_{0,1}, \dots\,\  A_{m-1, n-1}]$ |   ✅   |
| Reshape            | $\text{Reshape}(A_{m,n}) = A_{n,m}$                                       |   ✅   |
| Identity matrix (eye)            | $\text{The Identity matrix }{ I_n }\text{ of size }{ n} { x } { n }\text{ is defined as:} \ \ {I_{ij}} = \bigg( \frac{1 \text{ if } i = j}{0 \text{ if } i\ne j}$                                       |   ✅   |


Where:
- $\(A_{ij}\)$, $\(B_{ij}\)$, and $\(C_{ij}\)$ represent elements at the $\(i\)-th$ row and $\(j\)-th$ column of matrices $\(A\)$, $\(B\)$, and $\(C\)$, respectively.
- The matrices $\(A\)$ and $\(B\)$ must have the same dimensions for addition to be valid.


-	Operation Derivative

| Operation Derivative    | Formula                                                                        | Status |
|-------------------------|--------------------------------------------------------------------------------|--------|
| Addition_backward       | $\frac{\partial C}{\partial A} = I, \quad \frac{\partial C}{\partial B} = I$   |   ✅   |
| Subtraction_backward    | $\frac{\partial C}{\partial A} = I, \quad \frac{\partial C}{\partial B} = -I$  |   ✅   |
| Maltiplication_backward | $\frac{\partial C}{\partial A} = B, \quad \frac{\partial C}{\partial B} = A$   |   ✅   |
| Division_backward       | $\frac{\partial C}{\partial A} = B, \quad \frac{\partial C}{\partial B} = A$   |   ✅   |
| Dot_Product_backward    | $\frac{\partial C}{\partial A} = I, \quad \frac{\partial C}{\partial B} = I$   |   ✅   |
| Exponent_backward       | $\frac{\partial C}{\partial X} = e^{x_{i,j}}$                                  |   ✅   |
| Logarithm_backward      | $\frac{\partial C}{\partial X} = \frac{1}{X}$                                  |   ❌   |
| Power_backward          | $\frac{\partial C}{\partial A} = B \cdot A^{n-1}$                              |   ✅   |
| Sum_backward            | $\frac{\partial C}{\partial X_i} = 1\ \  \text{for each}\ \  {i}$              |   ✅   |
| Transpose_backward      | Not applicable for individual elements but preserves structure.                |   -   |
| Flatten_backward        | No derivative directly, but a 1-to-1 mapping between elements is maintained.   |   -   |
| Reshape_backward        | No direct derivative as it doesn’t involve computation. Used for data structure organization.|   -   |


-    Memory Management: Efficient use of malloc, calloc, memcpy, and memset for optimized memory handling.

| Task       | Status |
|------------|--------|
| Free Tensor|   ✅   |


2.	Automatic Differentiation

-	Gradient Storage: Each tensor can store its gradient, initialized with calloc for zeroing the memory.

| Task          | Status |
|---------------|--------|
| grad	        |   ✅   |
| requires_grad |   ✅   |


-	Backward Propagation: Simple backpropagation framework to calculate gradients for model parameters.

| Task       | Status |
|------------|--------|
| Backward   |   ✅   |


-	Operators for Gradient Tracking: Support for chaining operations to compute gradients through layers of the network.

| Task       | Status |
|------------|--------|
| prev       |   ✅   |
| op	     |	 ✅   |
| num_prev   |	 ✅   |


3.	Basic Neural Network Layers
   
-	Linear (Dense) Layer: Implement a fully connected layer, allowing the network to learn transformations.

| Task       | Status |
|------------|--------|
| Linear     |   ❌   |


-	Activation Functions: Include foundational activation functions (e.g., ReLU, Sigmoid, Tanh) with support for gradient calculations.

i.	Activations

| Task      |            Formular               | Status |
|-----------|-----------------------------------|--------|
| ReLU      | $\text{ReLU({x})} = \bigg( \frac{{x }\ \text{ if } {x }  \geq\ 0} {0  \text{ if } {x } < 0}$ |   ✅   |
| sigmoid   |  $\sigma(x) = \frac{1}{1 + e^{-x}}$   |   ✅   |
| tanh      |  $\text{tanh}(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$  |   ✅   |
| softmax   |  $\text{Sofmax}{(x_i)} = \frac{e^{x_i - \text{max(x)}}}{\sum_{j}{e^{x_j - \text{max(x)}}}}$  |   ✅   |
| LeakyReLU| $\text{LeakyReLU({x})} = \bigg( \frac{{x }\ \text{ if } {x }  \geq\ 0} {\alpha {x}  \text{ if } {x } < 0}$                                   |   ✅   |
| mean      |  $\mu = \frac{1}{n} \sum_{i=1}^n x_i$ |   ✅   |

- Note: Softmax Numerical Stability
    - When  x  has large values,  $e^{x_i}$  may overflow. For numerical stability, PyTorch internally subtracts the maximum value from  $x$  before applying the softmax:


ii.	Activations Derivative

| Task      |            Formular               | Status |
|-----------|-----------------------------------|--------|
| ReLU_backward      | $\frac{\partial}{\partial{x}} = \bigg( \frac{{1 }\ \text{ if } {x }  \geq\ 0} {0  \text{ if } {x } \le 0}$ |   ✅   |
| sigmoid_backward   |  $\sigma{\prime}(x) = \sigma(x)(1 - \sigma(x))$   |   ✅   |
| tanh_backward      |  $\text{tanh}{\prime}(x) = 1 - \text{tanh}^2(x)$   |   ✅   |
| softmax_backward   |  $\frac{\partial}{\partial{x_k}} = \text{Softmax}{(x_k)}(1 - \text{Softmax}{(x_k)})_{(diagonal: )} cross-element\ requires\ Jacobian$  |   ❌   |
| LeakyReLU_backward| $\frac{\partial}{\partial{x}} = \bigg( \frac{{1 }\ \text{ if } {x }  \geq\ 0} {\alpha  \text{ if } {x } \le 0}$                                   |   ✅   |
| mean_backward      |  $\frac{\partial{\mu}}{\partial{x_i}} = \frac{1}{n}$ |   ✅   |

- Note: Derivative of Softmat.
    - The derivative depends on whether you’re computing it for the same index $( i = j )$ or different indices $( i \neq j )$.
    - For a vector  $\mathbf{s} = \text{Softmax}(\mathbf{x})$ , the derivative is a matrix (Jacobian) given by:
    - When  $i = j$ : The derivative is  $s_i (1 - s_i)$ , representing the change in  $s_i$  with respect to  $x_i$ .
    - When  $i \neq j$ : The derivative is  $-s_i s_j$ , showing the interaction between different outputs of the softmax.
      
$$\frac{\partial s_i}{\partial x_j} =
\begin{cases}
s_i (1 - s_i), & \text{if } i = j \\
-s_i s_j, & \text{if } i \neq j
\end{cases}$$

- Matrix Form of the Derivative (Jacobian).
    - The derivative can be represented as a Jacobian matrix for the softmax vector:


$$\mathbf{J}(\mathbf{s}) = \text{diag}(\mathbf{s}) - \mathbf{s} \mathbf{s}^T$$

 - Where:
    - 	$\text{diag}(\mathbf{s})$ : Diagonal matrix with  $s_i$  on the diagonal.
	-	$\mathbf{s} \mathbf{s}^T$ : Outer product of  $\mathbf{s}$  with itself.

    -   Explicitly:

$$\mathbf{J}(\mathbf{s}) =
\begin{bmatrix}
s_1 (1 - s_1) & -s_1 s_2 & \cdots & -s_1 s_n \\
-s_2 s_1 & s_2 (1 - s_2) & \cdots & -s_2 s_n \\
\vdots & \vdots & \ddots & \vdots \\
-s_n s_1 & -s_n s_2 & \cdots & s_n (1 - s_n)
\end{bmatrix}$$




-	Loss Functions: Basic loss functions like Mean Squared Error and Cross-Entropy to train simple models.


### Loss Functions


| Loss Function                   | Type                   | When to Use                             | Advantages                           | Disadvantages                          |Status  |
|---------------------------------|------------------------|-----------------------------------------|--------------------------------------|----------------------------------------|--------|
| **Mean Squared Error (MSE)**    | Regression             | Regression with continuous targets      | Simple, differentiable               | Sensitive to outliers                  |   ✅   |
| **Mean Absolute Error (MAE)**   | Regression             | Regression with noisy data              | Less sensitive to outliers           | Less smooth gradient                   |   ❌   |
| **Cross-Entropy**               | Classification         | Binary or multi-class classification    | Works well with probabilistic models | Sensitive to class imbalance           |   ❌   |
| **Hinge Loss (SVM)**            | Classification         | Support Vector Machines (SVM)           | Efficient for margin classifiers     | Not suitable for probabilistic tasks   |   ❌   |
| **Huber Loss**                  | Regression             | Regression with outliers                | Robust to outliers, smooth           | Requires tuning of threshold $\delta$  |   ❌   |    
| **KL Divergence**               | Probabilistic Models   |Variational inference, generative models | Compares probability distributions   | Asymmetric, computationally expensive  |   ❌   |

**i.    Mean Squared Error (MSE) Loss**

Formula:


$\text{MSE} = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

The derivative of MSE with respect to each prediction  $\hat{y}_i$  is:


$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = -\frac{1}{n}(y_i - \hat{y}_i)$


Where:
-	 $y_i  = True value$
-	 $\hat{y}_i  = Predicted value$
-	$n  = Number of data points$

How it Works:

-    MSE calculates the average of the squared differences between predicted and true values. It penalizes large errors more significantly due to the squaring of the difference.


**ii.    Mean Absolute Error (MAE) Loss**

Formula:


$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

How it Works:

-    MAE computes the average of the absolute differences between predicted and true values. Unlike MSE, it does not square the differences, which makes it less sensitive to large errors.
  

**iii.    Cross-Entropy Loss**

Formula (for Binary Classification):


$\text{Binary Cross-Entropy} = - \frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$


For Multi-Class Classification:

$\text{Categorical Cross-Entropy} = - \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$

Where:
-	 $y_i  = True probability distribution$
-	 $\hat{y}_i  = Predicted probability distribution$
-    $C  = Number of classes$
-    $y_{ic}  = 1 if the instance belongs to class  c , else 0$
-    $\hat{y}_{ic}  = Predicted probability for class  c $

How it Works:

-    Cross-entropy loss measures the difference between two probability distributions: the true label distribution and the predicted probability distribution. It is widely used in classification tasks.


**iv.    Hinge Loss (SVM Loss)**

**Formula:**


$\text{Hinge Loss} = \sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)$

**Where:**
-	 $y_i  = True label (+1 or -1)$
-	 $\hat{y}_i  = Predicted score (not probability)$


**How it Works:**

-    Hinge loss is used in Support Vector Machines (SVM) and other classification tasks. It penalizes predictions that are on the wrong side of the decision boundary and doesn’t penalize correctly classified points as long as they are on the correct side of the margin.



**v.    Huber Loss**

**Formula:**


$$\text{Huber}(\delta) =
\begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2, & \text{for } |y_i - \hat{y}_i| \leq \delta \\
\delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2, & \text{otherwise}
\end{cases}$$

**Where:**
-	 $\delta$  is a threshold that determines the transition from quadratic to linear loss.

**How it Works:**

-    Huber loss combines both MSE and MAE. It behaves like MSE for small errors (i.e., when the absolute error is less than  \delta ) and like MAE for large errors. This makes it less sensitive to outliers compared to MSE while still being differentiable.

**vi.    Kullback-Leibler Divergence (KL Divergence)**

**Formula:**


$\text{KL Divergence} = \sum_{i=1}^{n} p_i \log\left(\frac{p_i}{q_i}\right)$

**Where:**
-	 $p_i  = True distribution (e.g., true labels)$
-	 $q_i  = Predicted distribution$

**How it Works:**

-    $KL$ divergence measures the difference between two probability distributions. It is asymmetric, meaning  $\text{KL}(p \parallel q) \neq \text{KL}(q \parallel p)$ .


4. Optimization Algorithms
   
-	Gradient Descent: Implement vanilla gradient descent for updating weights.

| Task            | Status |
|-----------------|--------|
| step optimizer  |   ❌   |

 
-	Extensions: Planned support for optimizations like Stochastic Gradient Descent (SGD) and other optimizers (e.g., Adam) as the library progresses.
### Optimizers

| Task  | Status |
|-------|--------|
| ADAM  |   ❌   |
| SGD   |   ❌   |


5.	Training and Evaluation Loop
   
-	Forward and Backward Passes: Execution of forward pass and automatic differentiation for backpropagation.
  
-	Metrics Tracking: Calculate and log accuracy or loss during training.
  
-	Progress Display: Basic progress bar for training epochs and mini-batches.
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


# Future Milestones

1.	Additional Neural Network Layers
   
-	Convolutional Layers: Add convolution layers for basic image-processing tasks.
  
-	Pooling Layers: Max and average pooling layers for reducing spatial dimensions.

  
2.	Expanded Tensor Operations
   
-	Broadcasting: Support for basic broadcasting to handle mismatched tensor shapes.
  
-	Advanced Math Operations: Include more operations (e.g., exponentiation, logarithms) for increased model complexity.

  
3.	GPU/Hardware Support
   
-	OpenCL/CUDA Integration: Explore integration with OpenCL or CUDA to leverage GPUs for faster computation.
  
-	SIMD Optimizations: Use SIMD instructions for faster CPU-based tensor operations.

  
4.	Serialization and Model Exporting
   
-	Model Saving and Loading: Save model weights and parameters for reproducibility and deployment.
  
-	ONNX Export: Basic support for ONNX format export, allowing compatibility with other deep learning frameworks.

  
5.	Python Bindings
   
-	Python API: Create a minimal Python API for easy usage and debugging, making it accessible for Python-based experimentation.
