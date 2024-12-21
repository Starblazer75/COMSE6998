import argparse
import torch
import time
import torch.nn.utils.prune as prune
from depth_anything_v2.dpt import DepthAnythingV2

def prune_module(module, amount):
    if isinstance(module, torch.nn.Linear):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    elif isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)

def main(args):
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to('cuda')
    model.load_state_dict(torch.load(args.input_model, map_location='cpu'))

    pruning_targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            pruning_targets.append((name, module))
        elif isinstance(module, torch.nn.Conv2d):
            pruning_targets.append((name, module))

    print(f"Pruning {len(pruning_targets)} layers...")

    for name, module in pruning_targets:
        print(f"Pruning {name} ({type(module).__name__})...")
        prune_module(module, amount=args.pruning_percentage)

    for name, module in pruning_targets:
        prune.remove(module, 'weight')

    torch.save(model.state_dict(), args.output_model)
    print(f"Pruned model saved to {args.output_model}")

    x = torch.rand(1, 3, 518, 518).to('cuda')  
    with torch.no_grad():
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        print(f"Inference time (pruned model): {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prune Depth Anything V2 model')
    
    parser.add_argument('--input-model', type=str, required=True, help="Path to the original model checkpoint")
    parser.add_argument('--output-model', type=str, required=True, help="Path to save the pruned model checkpoint")
    parser.add_argument('--pruning-percentage', type=float, required=True, help="Percentage of weights to prune (e.g., 0.01 for 1%)")

    args = parser.parse_args()
    main(args)
