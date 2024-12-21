import json
import numpy as np
from PIL import Image
from tqdm import tqdm 

def get_depth_at_points(depth_image_path, point):
    depth_image = np.array(Image.open(depth_image_path), dtype=np.int32) 
    try:
        return depth_image[point[0], point[1]][0]
    except IndexError:
        return False

def evaluate_accuracy(annotations_file, image_directory, image_extension='.png'):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    total = 0
    correct = 0

    for image_path, annotations_list in tqdm(annotations.items(), desc="Processing images", unit="image"):
        image_path = image_path.replace("images", image_directory)
        image_path = image_path.replace('.jpg', image_extension)
        image_path = image_path.replace('.jpeg', image_extension)
        image_path = image_path.replace('.webp', image_extension)

        for annotation in annotations_list:
            total += 1

            point1 = annotation["point1"]
            point2 = annotation["point2"]
            closer_point = annotation["closer_point"]

            try:
                depth1 = get_depth_at_points(image_path, point1)
                depth2 = get_depth_at_points(image_path, point2)
            except FileNotFoundError:
                continue

            if isinstance(depth1, bool) or isinstance(depth2, bool):
                continue

            if depth1 > depth2:  
                expected_closer_point = 'point1'
            else:
                expected_closer_point = 'point2'

            #print(f'Closer: {closer_point}, Expected: {expected_closer_point}, Depth1: {depth1}, Depth2: {depth2}, Image: {image_path}')

            if closer_point == expected_closer_point:
                correct += 1

    accuracy = (correct / (total // 2)) * 100 if total > 0 else 0
    return accuracy

annotations_file = "DA-2K/annotations.json"
image_directory = "fine_weights"  
image_extension = ".png"  
accuracy = evaluate_accuracy(annotations_file, image_directory, image_extension)
print(f"Accuracy: {accuracy:.2f}%")