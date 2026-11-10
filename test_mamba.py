import torch
from model import Mamba, ModelArgs

# 1. Create a tiny Mamba model
args = ModelArgs(
    d_model=16,
    n_layer=1,
    vocab_size=32
)

model = Mamba(args)

# 2. Create two input sequences, each containing 5 tokens
input_ids = torch.randint(0, 32, (2, 5))

# 3. Run the model
output = model(input_ids)

# 4. Display results
print("Input:")
print(input_ids)

print("\nInput shape:", input_ids.shape)
print("Output shape:", output.shape)

print("\nOutput for first sequence, first position:")
print(output[0, 0])