import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Define paths
data_dir = "DA-2K/images"
depth_dir = "DA-2K/depths"
pruned_model_path = "checkpoints/depth_anything_v2_vits_pruned.pth"
fine_tuned_model_path = "checkpoints/depth_anything_v2_vits_finetuned.pth"

# Hyperparameters
batch_size = 8
learning_rate = 1e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset for Depth Estimation
class DepthEstimationDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None, depth_transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_paths = []
        self.depth_paths = []

        for category in os.listdir(image_dir):
            category_path = os.path.join(image_dir, category)
            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):  # Update this based on file types
                        image_file = os.path.join(category_path, file_name)
                        depth_file = os.path.join(depth_dir, category, file_name)
                        if os.path.exists(depth_file[0:len(depth_file) - 4] + '.npy'):
                            self.image_paths.append(image_file)
                            self.depth_paths.append(depth_file)

        print(f"Found {len(self.image_paths)} images and {len(self.depth_paths)} depth maps")

        self.transform = transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Modify the depth file path to use .npy instead of .jpg or .png
        depth_file = self.depth_paths[idx][0:len(self.depth_paths[idx]) - 4] + '.npy'
        
        # Load the .npy depth map
        depth = np.load(depth_file)  # Load the .npy file

        # Apply transform for images
        if self.transform:
            image = self.transform(image)

        # Convert depth map to PIL Image for resizing (assuming depth map is a 2D array)
        depth = Image.fromarray(depth.astype(np.float32))  # Convert depth map to PIL Image

        # Apply depth transform (resize depth map to match image size)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        # Convert depth map to a tensor
        depth = torch.tensor(np.array(depth), dtype=torch.float32)

        return image, depth

# Data transform for depth maps (Resize depth map to the same size as the images)
depth_transform = transforms.Compose([
    transforms.Resize((518, 518))  # Same size as the images
])

# Load dataset
train_dataset = DepthEstimationDataset(os.path.join(data_dir, "train"), os.path.join(depth_dir, "train"), transform=data_transforms, depth_transform=depth_transform)
val_dataset = DepthEstimationDataset(os.path.join(data_dir, "val"), os.path.join(depth_dir, "val"), transform=data_transforms, depth_transform=depth_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the pruned model
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(device)
model.load_state_dict(torch.load(pruned_model_path, map_location=device))

# Define optimizer and loss function
criterion = nn.MSELoss()  # Using Mean Squared Error for depth estimation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, depths in train_loader:
            inputs, depths = inputs.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, depths in val_loader:
                inputs, depths = inputs.to(device), depths.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, depths)
                val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

        # Save the best model based on validation loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), fine_tuned_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")

# Start fine-tuning
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)