import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
from PIL import Image
import multiprocessing

# Early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=4, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Contrastive Loss for Siamese Network
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        loss = label * torch.pow(output, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2)
        return torch.mean(loss)


# Siamese Network with Transfer Learning
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')  # Load pretrained ResNet18
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output similarity score
        )

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten for FC layers
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = torch.abs(output1 - output2)  # Compute L1 distance
        output = self.fc(distance)
        return output


# Custom dataset for image pairs
class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.classes = list(image_folder.class_to_idx.keys())
        self.image_data = []

        for label in self.classes:
            image_paths = [img_path for img_path, _ in image_folder.imgs if image_folder.class_to_idx[label] in image_folder.class_to_idx.values()]
            self.image_data.append((image_paths, label))

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image_paths, label = self.image_data[index]
        img1_path, img2_path = random.sample(image_paths, 2)  # Positive pair

        # 50% chance of selecting a negative pair
        if random.random() > 0.5:
            label2 = random.choice([l for l in self.classes if l != label])
            img2_path = random.choice([img_path for img_path, _ in self.image_folder.imgs if self.image_folder.class_to_idx[label2] in self.image_folder.class_to_idx.values()])
            label = 0  # Different class
        else:
            label = 1  # Same class

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)


# Training function
def train_siamese(model, dataloaders, criterion, optimizer, num_epochs=25, save_path="siamese_best.pt"):
    since = time.time()
    best_loss = float("inf")
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for img1, img2, labels in dataloaders[phase]:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img1, img2)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * img1.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Save best model weights
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_path)

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                model.load_state_dict(torch.load(save_path))
                return model

        print()

    print(f'Training complete in {time.time() - since:.0f}s')
    return model


# Main function to load data and train model
def main():
    cudnn.benchmark = True
    plt.ion()

    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset directory setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), '../dataset')

    train_folder = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    val_folder = datasets.ImageFolder(os.path.join(data_dir, 'val'))

    train_dataset = SiameseDataset(train_folder, transform=data_transforms)
    val_dataset = SiameseDataset(val_folder, transform=data_transforms)

    dataloaders = {
    'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    }
    
    # Set device (MPS for Mac, CUDA for GPU, else CPU)
    global device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and train model
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_siamese(model, dataloaders, criterion, optimizer, num_epochs=25)

    print("Training complete!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
