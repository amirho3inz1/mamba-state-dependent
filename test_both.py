import torch

from model import Mamba as OriginalMamba
from model_modified import Mamba as ModifiedMamba
from model import ModelArgs


# Model configuration
args = ModelArgs(
    d_model=16,
    n_layer=2,
    vocab_size=100
)

# Create both models
original_model = OriginalMamba(args)
modified_model = ModifiedMamba(args)

# Same input for both models
input_ids = torch.randint(0, 100, (2, 10))

# Test original model
original_output = original_model(input_ids)

# Test modified model
modified_output = modified_model(input_ids)

# Compare results
print("Original output shape:", original_output.shape)
print("Modified output shape:", modified_output.shape)

print("\nOriginal model output:")
print(original_output)

print("\nModified model output:")
print(modified_output)

print("\nBoth models executed successfully!")