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
from torch.profiler import profile, ProfilerActivity, record_function


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, required=True, help="Path to input image directory")
    parser.add_argument('--input-size', type=int, default=518, help="Input size for the model")
    parser.add_argument('--outdir', type=str, default='./vis_depth', help="Base output directory")
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help="Output depth images in grayscale")
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    filenames = glob.glob(os.path.join(args.img_path, '*/*'))
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.cm.get_cmap('Spectral_r')
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for k, filename in enumerate(filenames[:10]):  # Limit to first 10 files for profiling
            print(f'Processing {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            with record_function("model_inference"):
                depth = depth_anything.infer_image(raw_image, args.input_size)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            relative_path = os.path.relpath(filename, args.img_path)
            output_path = os.path.join(args.outdir, relative_path)
            output_dir = os.path.dirname(output_path)
            
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
            cv2.imwrite(os.path.join(output_dir, output_filename), depth)
    
    print(prof.key_averages().table(sort_by="cuda_time_total" if DEVICE == 'cuda' else "cpu_time_total", row_limit=20))
