import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DepthAnythingV2 model")
    parser.add_argument('--input', type=str, required=True, help="Path to the pruned model (.pth)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the fine-tuned model (.pth)")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--depth-dir', type=str, required=True, help="Path to the directory containing depth maps")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of epochs for training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

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
                        if file_name.endswith('.png') or file_name.endswith('.jpg'):
                            image_file = os.path.join(category_path, file_name)
                            depth_file = os.path.join(depth_dir, category, file_name)
                            if os.path.exists(depth_file[:-4] + '.npy'):
                                self.image_paths.append(image_file)
                                self.depth_paths.append(depth_file)

            print(f"Found {len(self.image_paths)} images and {len(self.depth_paths)} depth maps")

            self.transform = transform
            self.depth_transform = depth_transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            depth_file = self.depth_paths[idx][:-4] + '.npy'
            depth = np.load(depth_file)

            if self.transform:
                image = self.transform(image)

            depth = Image.fromarray(depth.astype(np.float32))

            if self.depth_transform:
                depth = self.depth_transform(depth)

            depth = torch.tensor(np.array(depth), dtype=torch.float32)

            return image, depth

    depth_transform = transforms.Compose([
        transforms.Resize((518, 518))
    ])

    train_dataset = DepthEstimationDataset(os.path.join(args.image_dir, "train"), os.path.join(args.depth_dir, "train"),
                                           transform=data_transforms, depth_transform=depth_transform)
    val_dataset = DepthEstimationDataset(os.path.join(args.image_dir, "val"), os.path.join(args.depth_dir, "val"),
                                         transform=data_transforms, depth_transform=depth_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]).to(device)
    model.load_state_dict(torch.load(args.input, map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
        best_loss = float('inf')
        for epoch in range(num_epochs):
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

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                torch.save(model.state_dict(), args.output)
                print(f"Best model saved with loss: {best_loss:.4f}")

        print(f"Training complete. Best validation loss: {best_loss:.4f}")

    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device)
