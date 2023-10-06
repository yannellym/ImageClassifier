import torch

# Specify the path to your checkpoint file
checkpoint_path = 'checkpoints/checkpoint.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Print the contents of the checkpoint
print(checkpoint)

