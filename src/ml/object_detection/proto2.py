import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path):
                continue  # Skip non-directory files like .DS_Store
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
                    images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            nn.Flatten(),
            nn.Linear(64*4*4, 64)  # Adjust the input size based on your image dimensions
        )
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder[:-2](x)  # Pass through conv blocks before flattening
        print(f"Shape before flattening: {x.shape}")
        x = self.encoder[-2](x)  # Flatten
        x = self.encoder[-1](x)  # Linear layer
        return x


def euclidean_dist(x, y):
    return torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)

def create_class_prototypes(model, data_loader, device):
    model.eval()
    prototypes = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)  # Move images to the correct device
            embeddings = model(images)
            
            for embedding, label in zip(embeddings, labels):
                if label.item() not in labels_list:
                    labels_list.append(label.item())
                    prototypes.append(embedding)
                else:
                    idx = labels_list.index(label.item())
                    prototypes[idx] = (prototypes[idx] + embedding) / 2  # Update prototype
    
    # Stack prototypes into a single tensor
    prototypes = torch.stack(prototypes)
    
    return prototypes, labels_list


def classify_query(model, query_image, prototypes, device):
    model.eval()
    
    query_image = query_image.unsqueeze(0).to(device)  # Move query image to the device
    query_embedding = model(query_image)
    
    # Calculate distances between query embedding and prototypes
    distances = torch.cdist(query_embedding, prototypes.unsqueeze(0)).squeeze(0)
    
    # Find the class with the smallest distance
    predicted_label = distances.argmin().item()
    
    return predicted_label, distances

def main():
    # Set up dataset and data loader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset('car_data/support_set/', transform=transform)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Initialize model and move to device
    model = PrototypicalNetwork().to(device)
    
    # Create class prototypes
    prototypes, labels_list = create_class_prototypes(model, data_loader, device)
    
    # Example: Classify a query image
    query_image, true_label = dataset[0]  # Get the first image as an example
    query_image = query_image.to(device)  # Move image to device
    predicted_label, distances = classify_query(model, query_image, prototypes, device)
    
    print(f"True label: {dataset.classes[true_label]}")
    print(f"Predicted label: {dataset.classes[labels_list[predicted_label]]}")
    print("Distances to prototypes:")
    
    # Loop through distances and print them
    for i in range(len(distances)):
        # Assuming distances is a 1D tensor with distances to each class prototype
        print(f"  {dataset.classes[labels_list[i]]}: {distances[i].item():.4f}")

if __name__ == "__main__":
    main()

