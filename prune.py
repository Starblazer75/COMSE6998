import argparse
import cv2
import glob
import matplotlib
import matplotlib.cm
import numpy as np
import os
import torch
import time
from depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.utils.prune as prune

# Load the model
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to('cuda')
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))

# Identify layers to prune
# Focus on the most computationally expensive components: linear layers, attention, and convolution layers
pruning_targets = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        pruning_targets.append((name, module))
    elif isinstance(module, torch.nn.Conv2d):
        pruning_targets.append((name, module))

print(f"Pruning {len(pruning_targets)} layers...")

# Apply pruning (structured pruning for channels in Conv2d and neurons in Linear)
def prune_module(module, amount):
    if isinstance(module, torch.nn.Linear):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)  # Prune neurons
    elif isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)  # Prune filters

# Prune each identified module
for name, module in pruning_targets:
    print(f"Pruning {name} ({type(module).__name__})...")
    prune_module(module, amount=0.01)  # Adjust the amount based on experimentation

# Remove pruning reparameterization to finalize the model
for name, module in pruning_targets:
    prune.remove(module, 'weight')

# Save the pruned model
pruned_model_path = 'checkpoints/depth_anything_v2_vits_pruned.pth'
torch.save(model.state_dict(), pruned_model_path)
print(f"Pruned model saved to {pruned_model_path}")

# Test the pruned model to ensure it works
x = torch.rand(1, 3, 518, 518).to('cuda')  # Example input size, adjust as per your use case
with torch.no_grad():
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    print(f"Inference time (pruned model): {end_time - start_time:.4f} seconds")