import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import random
import numpy as np

class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.7, val_ratio=0.15):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_split_images()

    def _load_split_images(self):
        all_images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            class_images = [(os.path.join(class_dir, img), self.class_to_idx[cls]) 
                            for img in os.listdir(class_dir)]
            random.shuffle(class_images)
            all_images.extend(class_images)
        
        random.shuffle(all_images)
        n_total = len(all_images)
        n_train = int(self.train_ratio * n_total)
        n_val = int(self.val_ratio * n_total)
        
        if self.split == 'train':
            return all_images[:n_train]
        elif self.split == 'val':
            return all_images[n_train:n_train+n_val]
        else:  # test
            return all_images[n_train+n_val:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img1_path, label1 = self.images[idx]
        
        # Randomly choose if we want a similar or dissimilar pair
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            same_class_images = [img for img, label in self.images if label == label1]
            img2_path, _ = random.choice(same_class_images)
        else:
            different_class_images = [img for img, label in self.images if label != label1]
            img2_path, _ = random.choice(different_class_images)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([int(should_get_same_class)], dtype=torch.float32)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    patience = 10  # for early stopping

    # Data transforms with augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = CarDataset(root_dir='path/to/your/dataset', transform=transform, split='train')
    val_dataset = CarDataset(root_dir='path/to/your/dataset', transform=transform, split='val')
    test_dataset = CarDataset(root_dir='path/to/your/dataset', transform=transform, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, and optimizer
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break

    # Test the best model
    model.load_state_dict(torch.load('best_siamese_model.pth'))
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()