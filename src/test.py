import torch

# Define dimensions
input_size = 3
output_size = 2
batch_size = 4

# Generate some input data
input_data = torch.randn(batch_size, input_size)

# Initialize weights and biases
weights = torch.randn(input_size, output_size, requires_grad=True)

# Forward pass through the custom linear layer
print (input_data.shape)
print (input_data)
print (weights.shape)
print (weights)
output = input_data @ weights

print("Output shape:", output.shape)
print("Output data:", output)
