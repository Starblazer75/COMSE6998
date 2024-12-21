import torch
import torch.onnx
from depth_anything_v2.dpt import DepthAnythingV2

# CHANGE PATHS HERE
original_path = 'checkpoints/depth_anything_v2_vits.pth'
onnx_path = 'checkpoints/depth_anything_v2_vits.onnx'
# CHANGE PATHS HERE

device = 'cuda'
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(device)

# CHANGE PATH HERE
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location=device))
# CHANGE PATH HERE

model.eval()

input = torch.rand(1, 3, 518, 518).to(device)  

torch.onnx.export(
    model,
    f=onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
        'output': {0: 'batch_size', 2: 'height', 3: 'width'} 
    }
)

print(f"Model successfully exported to {onnx_path}")
