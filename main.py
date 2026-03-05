
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Parameters

epochs = 1_000
inputs_size = 10
input_sample_size = 100
hidden1 = 10
output_size = 1
lr = 1e-3


# Data set

input = np.random.rand(inputs_size, input_sample_size)
output = np.random.rand(inputs_size, output_size)


# Network

class NN(nn.Module):
    def __init__(self, input_shape, hidden1, output):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden1, output)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)



# Training

model = NN(input_sample_size, hidden1, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(input)
    loss = criterion(pred, output)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Testing

with torch.no_grad():
    test_input = np.random.rand(inputs_size, input_sample_size)
    prediction = model(test_input)
    print("Prediction: ", prediction.item())

