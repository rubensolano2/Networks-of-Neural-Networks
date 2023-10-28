# 🧠 Neural Architecture with Nested Neurons

A novel neural network architecture that incorporates three hidden layers, with the unique feature of independent neural networks within each neuron of the second layer. These "neuron-nets" 🌀 receive values from adjacent layers, allowing them to adapt individually.

## 🚀 Introduction

Traditional neural networks consist of interconnected neurons within layers. This project aims to explore the potential of adding a level of complexity by turning each neuron in the second hidden layer into its own independent neural network. By doing so, we hypothesize that our architecture can adapt more effectively to complex datasets.

## ✨ Features

- **🌟 Three Hidden Layers:** Ensures depth and complexity in the model.
- **🔗 Neuron-Nets:** Each neuron in the second layer is a standalone neural network.
- **⚙️ Dynamic Adaptation:** The neuron-nets adapt based on values from their adjacent layers, allowing for more granular learning.

## 🛠 How It Works

1. **📥 Input Layer:** Receives the initial data.
2. **🔍 First Hidden Layer:** Processes the input and passes it to the second layer.
3. **🌀 Second Hidden Layer (Neuron-Nets):** Each neuron in this layer is an independent neural network. They receive values from both the first and third layers, adjusting their weights and biases accordingly.
4. **🔍 Third Hidden Layer:** Further processes the data before passing it to the output layer.
5. **📤 Output Layer:** Produces the final results.

## 📖 Detailed Explanation

### Imports:

Here, we're importing the necessary libraries from PyTorch.

### NeuronNet:

`NeuronNet` is a sub-neural network that acts as an "independent neuron" in the second hidden layer of our main architecture.

- `fc1` and `fc2` are linear (fully connected) layers representing the input and output of this sub-network.
- The `forward` function defines how data flows through this sub-network.

### NestedNeuralNetwork:

`NestedNeuralNetwork` is the main architecture we're building.

- `fc1` is the first hidden layer.
- `neuron_nets` is a list of sub-neural networks (`NeuronNet`) that act as neurons in the second hidden layer.
- `fc3` is the third hidden layer.

### `forward` Function of NestedNeuralNetwork:

This function defines how the input is processed through the network:

1. **First Hidden Layer**: The input `x` passes through the first hidden layer (`fc1`) and is then activated using the ReLU function.
2. **Second Hidden Layer (NeuronNets)**: For each "neuron" in the second hidden layer, we take the corresponding output from the first hidden layer and feed it through its sub-neural network (`NeuronNet`). This gives us a series of outputs that we then concatenate together.
3. **Third Hidden Layer**: We take the concatenated output from all the sub-neural networks and pass it through the third hidden layer (`fc3`).

### Initialization and Configuration:

Here, we initialize our `NestedNeuralNetwork` model and set the device on which it will run (in this case, the CPU). If you wanted to run it on a GPU with CUDA, you'd simply change "cpu" to "cuda:0".

In summary, we've created a neural network architecture where each neuron in the second hidden layer is itself an independent neural network. This allows the architecture to have additional adaptability and complexity compared to a standard feed-forward network.
