import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

# Generate 10 random 2x2 tensors
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
for i, sample in enumerate(samples):
    predicted_output = model(sample.unsqueeze(0))
    print(f'Sample {i+1}: \nActual: {sample} \nPredicted: {predicted_output.view(2, 2)}\n')

# Export the trained model to ONNX format
dummy_input = torch.randn(1, 2, 2)  # Create a dummy input tensor
onnx_filename = "mimic_model.onnx"
torch.onnx.export(model, dummy_input, onnx_filename)
print(f"Model exported to {onnx_filename}")

# Assuming 'model' is your trained model
# Save the trained model
torch.save(model.state_dict(), 'mimic_model.pth')

