# TensorFlow Neural Network Tutorial

This guide explains how to build, train, and evaluate neural networks in TensorFlow using the **Sequential API** and **Functional API**. It also includes feature extraction techniques.

---

## 1. Setup

### Import Required Libraries
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
```

### GPU Memory Management
```python
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

---

## 2. Loading and Preprocessing the MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
```

---

## 3. Building Neural Networks

### Sequential API

Define, compile, train, and evaluate the model as shown in the previous sections.

---

## 4. Feature Extraction from Intermediate Layers

### Creating a Feature Extraction Model
You can extract features from intermediate layers using TensorFlow's Functional API. For example:

```python
model = keras.Sequential(
    [
        keras.Input(shape=(784)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

# Create a new model for feature extraction
feature_model = keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers[:-1]],  # Exclude the last layer
)

# Generate predictions
features = feature_model.predict(x_train)

# Loop through features from intermediate layers
for feature in features:
    print(feature.shape)  # Print the shape of features from each layer
```

---

## 5. Why Feature Extraction?

Feature extraction is useful when you want to:
- Visualize intermediate outputs of a neural network.
- Use features from pre-trained models for transfer learning.
- Analyze the network's behavior.
