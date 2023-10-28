# Import required libraries
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the NeuronNet and FinalInputFirstHiddenSkipConnectionNestedNeuralNetwork classes
class NeuronNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuronNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FinalInputFirstHiddenSkipConnectionNestedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, neuron_net_hidden_size, hidden_size3, output_size, num_neurons):
        super(FinalInputFirstHiddenSkipConnectionNestedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.neuron_nets = nn.ModuleList([NeuronNet(input_size + hidden_size1, neuron_net_hidden_size, 1) for _ in range(num_neurons)])
        self.fc3 = nn.Linear(num_neurons, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.fc1(x)))
        skip_connection_input = torch.cat([x, x1], dim=1)
        x2 = torch.cat([neuron_net(skip_connection_input) for neuron_net in self.neuron_nets], dim=1)
        x3 = F.relu(self.bn3(self.fc3(x2)))
        x4 = self.fc4(x3)
        return x4
