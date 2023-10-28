# 🧠 Neural Architecture with Nested Neurons

A novel neural network architecture that incorporates three hidden layers, with the unique feature of independent neural networks within each neuron of the second layer. These "neuron-nets" 🌀 receive values from adjacent layers, allowing them to adapt individually.

## 🚀 Introduction

Traditional neural networks consist of interconnected neurons within layers. This project aims to explore the potential of adding a level of complexity by turning each neuron in the second hidden layer into its own independent neural network. By doing so, we hypothesize that our architecture can adapt more effectively to complex datasets.

## ✨ Features

- **🌟 Three Hidden Layers:** Ensures depth and complexity in the model.
- **🔗 Neuron-Nets:** Each neuron in the second layer is a standalone neural network.
- **⚙️ Dynamic Adaptation:** The neuron-nets can adapt based on values from their adjacent layers, allowing for more granular learning.

## 🛠 How It Works

1. **📥 Input Layer:** Receives the initial data.
2. **🔍 First Hidden Layer:** Processes the input and passes it to the second layer.
3. **🌀 Second Hidden Layer (Neuron-Nets):** Each neuron in this layer is an independent neural network. They receive values from both the first and third layers, adjusting their weights and biases accordingly.
4. **🔍 Third Hidden Layer:** Further processes the data before passing it to the output layer.
5. **📤 Output Layer:** Produces the final results.

