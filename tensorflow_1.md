# TensorFlow Beginner Guide

This guide explains the basics of using **TensorFlow** for creating and manipulating tensors, indexing, reshaping, GPU memory management, and setting up TensorFlow with GPU support. It is aimed at beginners, so everything is explained in simple terms.

---

## 1. Suppressing TensorFlow Logs

TensorFlow can produce a lot of log messages. To reduce unnecessary logs, you can use:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses warnings and info logs
```

---

## 2. Installing CUDA and cuDNN for GPU Support

TensorFlow leverages GPUs for faster computation. To enable GPU support, you need to install **CUDA** and **cuDNN**:

### 2.1 Installing CUDA
1. Download the CUDA Toolkit from the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads). Choose the version compatible with your TensorFlow version (e.g., CUDA 11.8 for TensorFlow 2.11+).
2. Install the toolkit and follow the instructions for your operating system.
3. Verify the installation:
   ```bash
   nvcc --version
   ```
   This command should display the installed CUDA version.

### 2.2 Installing cuDNN
1. Download cuDNN from the [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn). Ensure you download the version compatible with your CUDA version.
2. Extract the cuDNN files and copy them to the CUDA installation directory:
   - Copy the contents of `bin/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`
   - Copy the contents of `include/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\include`
   - Copy the contents of `lib/` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\lib\x64`

### 2.3 Adding CUDA and cuDNN to Environment Variables
Add the following paths to your system’s environment variables:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\libnvvp`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\lib\x64`

---

## 3. Managing GPU Memory Usage

If you are using a GPU, TensorFlow can sometimes allocate all available memory, even if your program doesn't need it. To prevent this, you can set memory growth for GPUs:

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

This ensures TensorFlow only uses the memory it needs, instead of pre-allocating all available GPU memory.

---

## 4. Tensor Basics

### 4.1 Tensor Initialization
Refer to the earlier section in this README for how to initialize tensors using constants, zeros, ones, and random values.

---

## 5. Tensor Indexing and Slicing

You can access elements of a tensor using indexing or slicing, similar to Python lists.

### 5.1 Simple Indexing
```python
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])

print(x[:])    # Print all elements
print(x[1:])   # From index 1 to the end
print(x[1:3])  # From index 1 to index 3 (excluding 3)
print(x[::2])  # Every second element
print(x[::-1]) # Reverse the tensor
print(x[:-1])  # All elements except the last one
```

### 5.2 Advanced Indexing
You can also use TensorFlow's `gather()` function to extract specific indices:

```python
indices = tf.constant([0, 3])  # Indices to gather
x_ind = tf.gather(x, indices)  # Gather values at these indices
```

---

## 6. Tensor Reshaping and Transposing

### 6.1 Reshaping Tensors
Change the shape of a tensor without altering its data:

```python
x = tf.range(9)  # Tensor with values 0 to 8
print(x)

x = tf.reshape(x, (3, 3))  # Reshape into a 3x3 matrix
print(x)
```

### 6.2 Transposing Tensors
Swap the rows and columns (or dimensions) of a tensor:

```python
x = tf.transpose(x, perm=[1, 0])  # Transpose matrix
print(x)
```

---

## 7. Summary

- **CUDA and cuDNN Installation**: Proper setup is critical for enabling GPU support.
- **GPU Memory Management**: Configure TensorFlow to use memory dynamically.
- **Indexing and Slicing**: Access elements and slices of a tensor easily.
- **Reshaping and Transposing**: Transform tensor shapes and dimensions as needed.

This guide is designed to help you get started with TensorFlow. Experiment with the examples and modify them to learn more!
