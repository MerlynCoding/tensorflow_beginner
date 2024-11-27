# TensorFlow Beginner Guide

This guide explains the basics of using **TensorFlow** for creating and manipulating tensors, indexing, reshaping, and GPU memory management. It is aimed at beginners, so everything is explained in simple terms.

---

## 1. Suppressing TensorFlow Logs

TensorFlow can produce a lot of log messages. To reduce unnecessary logs, you can use:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses warnings and info logs
```

---

## 2. Managing GPU Memory Usage

If you are using a GPU, TensorFlow can sometimes allocate all available memory, even if your program doesn't need it. To prevent this, you can set memory growth for GPUs:

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

This ensures TensorFlow only uses the memory it needs, instead of pre-allocating all available GPU memory.

---

## 3. Tensor Basics

### 3.1 Tensor Initialization
Refer to the earlier section in this README for how to initialize tensors using constants, zeros, ones, and random values.

---

## 4. Tensor Indexing and Slicing

You can access elements of a tensor using indexing or slicing, similar to Python lists.

### 4.1 Simple Indexing
```python
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])

print(x[:])    # Print all elements
print(x[1:])   # From index 1 to the end
print(x[1:3])  # From index 1 to index 3 (excluding 3)
print(x[::2])  # Every second element
print(x[::-1]) # Reverse the tensor
print(x[:-1])  # All elements except the last one
```

### 4.2 Advanced Indexing
You can also use TensorFlow's `gather()` function to extract specific indices:

```python
indices = tf.constant([0, 3])  # Indices to gather
x_ind = tf.gather(x, indices)  # Gather values at these indices
```

---

## 5. Tensor Reshaping and Transposing

### 5.1 Reshaping Tensors
Change the shape of a tensor without altering its data:

```python
x = tf.range(9)  # Tensor with values 0 to 8
print(x)

x = tf.reshape(x, (3, 3))  # Reshape into a 3x3 matrix
print(x)
```

### 5.2 Transposing Tensors
Swap the rows and columns (or dimensions) of a tensor:

```python
x = tf.transpose(x, perm=[1, 0])  # Transpose matrix
print(x)
```

---

## 6. Summary

- **GPU Memory Management**: Configure TensorFlow to use memory dynamically.
- **Indexing and Slicing**: Access elements and slices of a tensor easily.
- **Reshaping and Transposing**: Transform tensor shapes and dimensions as needed.

This guide is designed to help you get started with TensorFlow. Experiment with the examples and modify them to learn more!
