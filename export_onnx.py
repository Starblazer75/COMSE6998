import torch
import torch.onnx
import argparse
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DepthAnythingV2 model to ONNX format")
    parser.add_argument('--input', type=str, required=True, help="Path to the input PyTorch model (.pth)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the exported ONNX model (.onnx)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(device)

    model.load_state_dict(torch.load(args.input, map_location=device))
    model.eval()

    dummy_input = torch.rand(1, 3, 518, 518).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        f=args.output,
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

    print(f"Model successfully exported to {args.output}")
