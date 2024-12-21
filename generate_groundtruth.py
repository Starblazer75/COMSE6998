import argparse
import cv2
import glob
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from depth_anything_v2.dpt import DepthAnythingV2

class DepthDataset(Dataset):
    """PyTorch Dataset for hierarchical images and depth maps."""
    def __init__(self, metadata_path, img_dir, depth_dir, transform=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        
        img_path = os.path.join(self.img_dir, entry['image'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth_path = os.path.join(self.depth_dir, entry['depth'])
        depth = np.load(depth_path)
        
        if self.transform:
            augmented = self.transform(image=image, depth=depth)
            image = augmented['image']
            depth = augmented['depth']
        
        return image, depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, required=True, help="Path to input image directory")
    parser.add_argument('--input-size', type=int, default=518, help="Input size for the model")
    parser.add_argument('--outdir', type=str, default='./output', help="Base output directory")
    parser.add_argument('--extension', type=str, default='', help="Extension to append to the checkpoint file path")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])

    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}_{args.extension}.pth', map_location='cpu'))

    depth_anything = depth_anything.to(DEVICE).eval()
    
    if not os.path.isdir(args.img_path):
        raise ValueError("The input path must be a directory containing 'train' and 'val' subdirectories.")
    
    subsets = ['train', 'val']
    for subset in subsets:
        subset_dir = os.path.join(args.img_path, subset)
        if not os.path.isdir(subset_dir):
            raise ValueError(f"Missing required subset directory: {subset}")
        
        filenames = glob.glob(os.path.join(subset_dir, '*/*')) 
        subset_outdir = os.path.join(args.outdir, subset)
        os.makedirs(os.path.join(subset_outdir, "depth_maps"), exist_ok=True)
        
        output_metadata = []
        
        for k, filename in enumerate(filenames):
            print(f'Processing {subset} {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = depth_anything.infer_image(raw_image, args.input_size)
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            
            relative_path = os.path.relpath(filename, subset_dir)
            depth_output_path = os.path.join(subset_outdir, "depth_maps", relative_path)
            depth_output_dir = os.path.dirname(depth_output_path)
            os.makedirs(depth_output_dir, exist_ok=True)
            
            depth_filename = os.path.splitext(os.path.basename(filename))[0] + '.npy'
            depth_file_path = os.path.join(depth_output_dir, depth_filename)
            np.save(depth_file_path, depth_normalized)
            
            metadata_entry = {
                "image": relative_path,
                "depth": os.path.relpath(depth_file_path, subset_outdir)
            }
            output_metadata.append(metadata_entry)
        
        metadata_path = os.path.join(subset_outdir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(output_metadata, f, indent=4)

        print(f"\n{subset.capitalize()} processing complete. Metadata saved to: {metadata_path}")
