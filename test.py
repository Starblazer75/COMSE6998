import cv2

def analyze_image_size(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return
    
    # Get the size of the image
    height, width, channels = image.shape
    
    print(f"Image Size: {width}x{height}")
    print(f"Number of Channels: {channels}")
    
    # Additional Information
    image_size_bytes = image.nbytes
    print(f"Image Size in Bytes: {image_size_bytes} bytes")

# Example usage
image_path = '/home/lsaikali/classes/Depth-Anything-V2/16bit_weights/indoor/26253101282_c9ac343362_k.png'
analyze_image_size(image_path)
