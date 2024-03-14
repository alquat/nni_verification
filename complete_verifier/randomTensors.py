import random
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import json

# Set a fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_samples(num_samples, num_features, noise_level=0.1):
    # Generate random noise
    noise = np.random.normal(0, noise_level, (num_samples, num_features))
    
    # Create array of 2s and 5s
    samples = np.zeros((num_samples, num_features))
    samples[:, 0] = 2
    samples[:, 1] = 5
    
    # Add noise to samples
    samples_with_noise = samples + noise
    
    return samples_with_noise

def normalize_data(data):
    # Normalize data to fit into neural network's input dimensions
    mean = np.mean(data, axis=0)  # Compute mean along each feature
    std = np.std(data, axis=0)    # Compute standard deviation along each feature
    normalized_data = (data - mean) / std
    return normalized_data

def MimicNet(in_dim=2,out_dim=1):
    model = nn.Sequential(
        nn.Linear(in_dim, 10),
        nn.ReLU(),
        nn.Linear(10, out_dim)
        )
    return model

# Creating the samples
num_samples = 10
num_features = 2
noise_level = 0.1
samples = generate_samples(num_samples,num_features, noise_level)

# Normalize
normalized_samples = normalize_data(samples)
targets = np.sum(normalized_samples, axis=1)

# Convert to PyTorch tensor
input_data = torch.tensor(normalized_samples, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) 




# Instantiate the neural network
model = MimicNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the neural network to sum the samples
for epoch in range(1000):
    total_loss = 0
    for i in range(samples.shape[0]):
        sample = input_data[i].unsqueeze(0)
        target = targets[i]
        optimizer.zero_grad()  # Zero the gradients
        output = model(sample)  # Forward pass
        loss = criterion(output, target)  # Compute the loss
        total_loss += loss.item()
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {total_loss/len(samples)}')

# Test the trained model
output_samples = {}
for i, sample in enumerate(input_data):
    predicted_output = model(sample)
    print(f'Sample {i+1}: \nActual: {targets[i]} \nPredicted: {predicted_output}\n')
    output_samples[f"Sample_{i+1}"] = predicted_output.tolist()

# Export the trained model to ONNX format
dummy_input = torch.randn(1, num_features)  # Create a dummy input tensor
onnx_filename = "mimic_model_true.onnx"
torch.onnx.export(model, dummy_input, onnx_filename)
print(f"Model exported to {onnx_filename}")

# Assuming 'model' is your trained model
# Save the trained model
torch.save(model.state_dict(), 'mimic_model.pth')



