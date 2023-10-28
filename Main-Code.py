import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuronNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuronNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NestedNeuralNetwork(nn.Module):
    def __init__(self, input_size, neuron_net_hidden_size, output_size, num_neurons):
        super(NestedNeuralNetwork, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, num_neurons)
        
        # Second hidden layer with neuron-nets
        self.neuron_nets = nn.ModuleList([NeuronNet(1, neuron_net_hidden_size, 1) for _ in range(num_neurons)])
        
        # Third hidden layer
        self.fc3 = nn.Linear(num_neurons, output_size)

    def forward(self, x):
        # First layer
        x = F.relu(self.fc1(x))
        
        # Second layer with neuron-nets
        x = torch.cat([neuron_net(x[:, i].unsqueeze(1)) for i, neuron_net in enumerate(self.neuron_nets)], dim=1)
        
        # Third layer
        x = self.fc3(x)
        
        return x

# Initialize the model
input_size = 10
neuron_net_hidden_size = 5
output_size = 1
num_neurons = 3
model = NestedNeuralNetwork(input_size, neuron_net_hidden_size, output_size, num_neurons)

# Configure for CPU (can be easily moved to CUDA with .cuda())
device = torch.device("cpu")
model.to(device)
