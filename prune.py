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

# CHANGE PATHS HERE
original_model_path = 'checkpoints/depth_anything_v2_vits.pth'
pruned_model_path = 'checkpoints/depth_anything_v2_vits_pruned.pth'
# CHANGE PATHS HERE

model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to('cuda')
model.load_state_dict(torch.load(original_model_path, map_location='cpu'))

pruning_targets = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        pruning_targets.append((name, module))
    elif isinstance(module, torch.nn.Conv2d):
        pruning_targets.append((name, module))

print(f"Pruning {len(pruning_targets)} layers...")

def prune_module(module, amount):
    if isinstance(module, torch.nn.Linear):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    elif isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)  

for name, module in pruning_targets:
    print(f"Pruning {name} ({type(module).__name__})...")

    # CHANGE PERCENTAGE HERE
    prune_module(module, amount=0.01)  
    # CHANGE PERCENTAGE HERE

for name, module in pruning_targets:
    prune.remove(module, 'weight')

pruned_model_path = 'checkpoints/depth_anything_v2_vits_pruned.pth'
torch.save(model.state_dict(), pruned_model_path)
print(f"Pruned model saved to {pruned_model_path}")

x = torch.rand(1, 3, 518, 518).to('cuda')  
with torch.no_grad():
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    print(f"Inference time (pruned model): {end_time - start_time:.4f} seconds")