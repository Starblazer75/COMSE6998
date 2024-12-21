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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, required=True, help="Path to input image directory")
    parser.add_argument('--input-size', type=int, default=518, help="Input size for the model")
    parser.add_argument('--outdir', type=str, default='./vis_depth', help="Base output directory")
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help="Output depth images in grayscale")
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize DepthAnythingV2 model with the correct encoder
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])

    # Load the saved weights (ensure the correct path to your saved model weights)
    checkpoint_path = 'depth_anything_v2_16bit_finetuned.pth'  # Path to your saved model weights
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    # Move model to the correct device (GPU or CPU)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        raise ValueError("The input path must be a directory containing subdirectories with images.")
    
    filenames = glob.glob(os.path.join(args.img_path, '*/*'))
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.cm.get_cmap('Spectral_r')
    
    load_times = []
    inference_times = []
    total_times = []
    
    warmup_threshold = 100

    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')
        
        total_start_time = time.time()
        
        load_start_time = time.time()
        raw_image = cv2.imread(filename)
        load_end_time = time.time()

        inference_start_time = time.time()
        depth = depth_anything.infer_image(raw_image, args.input_size)
        inference_end_time = time.time()
        
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
        
        total_end_time = time.time()
        if k >= warmup_threshold:
            load_times.append(load_end_time - load_start_time)
            inference_times.append(inference_end_time - inference_start_time)
            total_times.append(total_end_time - total_start_time)
    
    if len(load_times) > 0:
        avg_load_time = sum(load_times) / len(load_times)
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_total_time = sum(total_times) / len(total_times)
        
        print("\n==== Timing Results ====")
        print(f"Average Load Time (after warm-up): {avg_load_time:.4f} seconds")
        print(f"Average Inference Time (after warm-up): {avg_inference_time:.4f} seconds")
        print(f"Average Total Time (after warm-up): {avg_total_time:.4f} seconds")
    else:
        print("\nNo timing results recorded; less than warm-up threshold images processed.")
