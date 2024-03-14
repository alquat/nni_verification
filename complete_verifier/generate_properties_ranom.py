import random
import onnxruntime  as ort
import json
import os
import csv
import sys

def read_vnnlib_params(property_id, network_name, file_path='vnnlib_params.csv'):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) >= 5 and row[0] == property_id and row[1] == network_name:
                min_perc = float(row[2])
                max_perc = float(row[3])
                random_perc = float(row[4])
                return min_perc, max_perc, random_perc

    return 0.95, 1.05, 0.001  # Default values if data not found


## HERE IS THE CODE FROM RANDOMTENSOR

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

# Generate 10 fixed 2x2 tensors
samples = []
for _ in range(10):
    sample = torch.randn(2, 2)  # Create a random 2x2 tensor
    samples.append(sample)

def MimicNet(in_dim=4,out_dim=4):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 10),
        nn.ReLU(),
        nn.Linear(10, out_dim)
        )
    return model

# Instantiate the neural network
model = MimicNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the neural network to mimic the samples
for epoch in range(1000):
    total_loss = 0
    for sample in samples:
        optimizer.zero_grad()  # Zero the gradients
        output = model(sample.unsqueeze(0))  # Forward pass
        loss = criterion(output, sample.view(-1))  # Compute the loss
        total_loss += loss.item()
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {total_loss/len(samples)}')

# Test the trained model
output_samples = {}
for i, sample in enumerate(samples):
    predicted_output = model(sample.unsqueeze(0))
    print(f'Sample {i+1}: \nActual: {sample} \nPredicted: {predicted_output.view(2, 2)}\n')
    output_samples[f"Sample_{i+1}"] = predicted_output.view(2, 2).tolist()

# Export the trained model to ONNX format
dummy_input = torch.randn(1, 2, 2)  # Create a dummy input tensor
onnx_filename = "mimic_model.onnx"
torch.onnx.export(model, dummy_input, onnx_filename)
print(f"Model exported to {onnx_filename}")

# Assuming 'model' is your trained model
# Save the trained model
torch.save(model.state_dict(), 'mimic_model.pth')


seed=42
# if the seed value is provided
if len(sys.argv) == 2:
    seed = int(sys.argv[1])

network_name = "mimic_model"

min_perc, max_perc, random_perc = 0.95, 1.05, 0.001

L = len(samples)
print("the length is ", L)
first_numers = [sample[0, 0] for sample in samples]
second_numbers = [sample[0, 1] for sample in samples]
third_numbers = [sample[1, 0] for sample in samples]
fourth_numbers = [sample[1, 1] for sample in samples]

with open(f"vnnlib/mimic_model_prop1.vnnlib", 'w') as f:
    for x in range(4):
        f.write(f"(declare-const X_{x} Real)\n")
        f.write("\n")
        for x in range(4):
            f.write(f"(declare-const Y_{x} Real)\n")
            f.write("\n")
            f.write("; Input constraints:\n")
            # input perturbation
        perturbation = [random.uniform(-random_perc, random_perc) for i in range(4)]

        for i in range(L):
            lb = samples


        