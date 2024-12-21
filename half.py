import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# Define paths
data_dir = "DA-2K/images/train"

# Define dataset transformations
transform = transforms.Compose([
    transforms.Resize((504, 504), antialias=True),  # Resize images for consistency
    transforms.ToTensor(),  # Convert to tensor
])

# Load dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Define and load model
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).eval()
model = model.half()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Calibration function (not training the model)
def calibrate(model, data_loader, device):
    with torch.no_grad():  # Disable gradient calculations (no training)
        for images, _ in tqdm(data_loader, desc='Calibrating', unit='batch'):  # Labels are ignored for calibration
            images = images.half()
            images = images.to(device)
            _ = model(images)  # Forward pass through the model

# Run calibration on the training dataset
calibrate(model, train_loader, device)

# Example input for testing the model
input_image = torch.rand(1, 3, 504, 504).half()
input_image = input_image.to(device)

# Forward pass through the model
output = model(input_image)

# Print the output shape
print(output.shape)

# Save the model
save_path = 'depth_anything_v2_16bit.pth'
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')
