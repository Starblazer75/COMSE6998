import argparse
import cv2
import glob
import matplotlib
import matplotlib.cm
import numpy as np
import os
import onnxruntime as ort
import time

def resize_image(image, target_size=518):
    """Resizes the image to the target size."""
    return cv2.resize(image, (target_size, target_size))

def image2tensor(image, target_size=518):
    """Converts the image to the tensor format expected by the model."""
    original_shape = image.shape[:2]
    image_resized = resize_image(image, target_size)
    image_resized = image_resized.astype(np.float32) / 255.0 
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized = np.expand_dims(image_resized, axis=0)  
    
    image_resized = image_resized.astype(np.float16)
    
    return image_resized, original_shape

def resize_depth_map(depth_map, original_shape):
    """Resizes the depth map back to the original image size."""
    return cv2.resize(depth_map, (original_shape[1], original_shape[0]))  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 with ONNX')
    
    parser.add_argument('--img-path', type=str, required=True, help="Path to input image directory")
    parser.add_argument('--input-size', type=int, default=518, help="Input size for the model")
    parser.add_argument('--outdir', type=str, default='./vis_depth', help="Base output directory")
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help="Output depth images in grayscale")
    parser.add_argument('--onnx-model-path', type=str, required=True, help="Path to the ONNX model file")
    
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
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
        if raw_image is None:
            print(f"Error loading image: {filename}")
            continue
        
        raw_image_resized, original_shape = image2tensor(raw_image, target_size=args.input_size)
        load_end_time = time.time()
        
        inference_start_time = time.time()
        depth = session.run([output_name], {input_name: raw_image_resized})[0]
        inference_end_time = time.time()
        
        depth = depth.squeeze()  
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        depth_resized = resize_depth_map(depth, original_shape)
        
        if args.grayscale:
            depth_resized = np.repeat(depth_resized[..., np.newaxis], 3, axis=-1)
        else:
            depth_resized = (cmap(depth_resized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        relative_path = os.path.relpath(filename, args.img_path)
        output_path = os.path.join(args.outdir, relative_path)
        output_dir = os.path.dirname(output_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), depth_resized)
        
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
