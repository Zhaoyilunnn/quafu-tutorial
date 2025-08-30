import numpy as np
import torch
from quafu.algorithms.ansatz import QuantumNeuralNetwork
from quafu.algorithms.interface.torch import ModuleWrapper
from quafu.algorithms.templates.angle import AngleEmbedding
from quafu.algorithms.templates.basic_entangle import BasicEntangleLayers
from quafu.elements import Parameter
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Create a synthetic dataset
def generate_random_dataset(num_inputs, num_samples):
    """
    Generate random dataset

    Args:
        num_inputs: dimension of input data
        num_samples: number of samples in the dataset
    """
    # Generate random input coordinates using PyTorch's rand function
    x = 2 * torch.rand([num_samples, num_inputs], dtype=torch.double) - 1

    # Calculate labels based on the sum of input coordinates
    y01 = (torch.sum(x, dim=1) >= 0).to(torch.long)

    # Convert to one-hot vector
    y = torch.zeros(num_samples, 2)  # Two classes (0 and 1)
    y[torch.arange(num_samples), y01] = 1

    # Create a PyTorch dataset
    dataset = TensorDataset(x, y)

    return dataset


dataset = generate_random_dataset(2, 100)

x = dataset.tensors[0]


# Virtualize the data
import matplotlib.pyplot as plt

# Extract x and y from dataset
x = dataset.tensors[0]
y = dataset.tensors[1]

# Plotting the data points with different colors based on labels
plt.figure(figsize=(8, 6))

# Extract coordinates for each class
x_class0 = x[y[:, 0] == 1]
x_class1 = x[y[:, 1] == 1]

# Plot points for each class with different colors
plt.scatter(x_class0[:, 0], x_class0[:, 1], color="blue", label="Class 0")
plt.scatter(x_class1[:, 0], x_class1[:, 1], color="red", label="Class 1")

plt.title("Random Dataset")


# Create a quantum classifier using pyquafu
num_qubits = 2
weights = np.random.randn(num_qubits, num_qubits)
encoder_layer = AngleEmbedding(np.random.random((num_qubits,)), num_qubits=2)
entangle_layer = BasicEntangleLayers(weights, num_qubits=num_qubits)
qnn = QuantumNeuralNetwork(num_qubits, encoder_layer + entangle_layer)

# Convert to torch module
model = ModuleWrapper(qnn)


# Alternatively, you could also customize encoder layer and ansatz layer


# Create a quantum classifier using pyquafu
import quafu.elements.element_gates as qeg

num_qubits = 2
weights = np.random.randn(num_qubits, num_qubits)
data = []
inputs0 = np.random.random((num_qubits * 2,))

# customize encoder layer
ry0 = qeg.RYGate(0, Parameter("phi_0", inputs0[0], tunable=False))
ry1 = qeg.RYGate(0, Parameter("phi_1", inputs0[1], tunable=False))
x = qeg.XGate(1)
cnot = qeg.CXGate(0, 1)
rx0 = qeg.RYGate(0, Parameter("phi_3", inputs0[2], tunable=False))
rx1 = qeg.RYGate(0, Parameter("phi_4", inputs0[3], tunable=False))
# encoder_layer = [ry0, ry1, x, cnot, rx0, rx1]
encoder_layer = [ry0, ry1, x]

entangle_layer = BasicEntangleLayers(weights, num_qubits=num_qubits)

# To execute on real quantum machine, set the `backend` parameter
# qnn1 = QuantumNeuralNetwork(num_qubits, encoder_layer + entangle_layer, backend="<device-name>")
qnn1 = QuantumNeuralNetwork(num_qubits, encoder_layer + entangle_layer)

# Convert to torch module
model1 = ModuleWrapper(qnn1)


# Virtualize the circuit
qnn.draw_circuit()


learning_rate = 0.1
batch_size = 8
num_epochs = 5

# Train the classifier
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # Forward pass
        # outputs = model(inputs)
        outputs = model1(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

        print(f" ----- Loss: {loss.item()}")
        print(model1.weights)

    # Print the loss
    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item()}")

# Evaluate the model on the dataset
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in data_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
