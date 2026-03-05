import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Parameters
epochs = 1_000
input_size = 10        # number of features
sample_size = 100      # number of samples
hidden1 = 10
output_size = 1
lr = 1e-3

# Generate dataset
X = np.random.rand(sample_size, input_size)   # shape: (100, 10)
y = np.random.rand(sample_size, output_size)  # shape: (100, 1)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define network
class NN(nn.Module):
    def __init__(self, input_size, hidden1, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Initialize model, loss, optimizer
model = NN(input_size, hidden1, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Testing
with torch.no_grad():
    test_X = np.random.rand(sample_size, input_size)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    prediction = model(test_X)
    print("Prediction shape:", prediction.shape)
    print("Sample predictions:", prediction[:5])
