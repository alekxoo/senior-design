import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class CarDataset(torch.utils.data.Dataset):
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
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} is not a directory. Skipping.")
                continue
            class_images = [(os.path.join(class_dir, img), self.class_to_idx[cls]) 
                            for img in os.listdir(class_dir) 
                            if os.path.isfile(os.path.join(class_dir, img)) and 
                            img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(class_images)
        
        if not all_images:
            raise ValueError("No valid images found in the dataset.")
        
        # Shuffle and split the dataset
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
        
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            same_class_images = [img for img, label in self.images if label == label1]
            img2_path = random.choice(same_class_images)[0]  # Select just the path
        else:
            different_class_images = [img for img, label in self.images if label != label1]
            img2_path = random.choice(different_class_images)[0]  # Select just the path
                    
        # Add error checking
        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            print(f"Invalid file path: {img1_path} or {img2_path}")
            return None  # You might want to handle this in your DataLoader
        
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return None
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor([int(should_get_same_class)], dtype=torch.float32)
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

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
    for img1, img2, label in tqdm(train_loader, desc="Training"):
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
        for img1, img2, label in tqdm(val_loader, desc="Validating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def plot_loss_history(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig('loss_history.png')
    plt.close()

def infer(model, img1_path, img2_path, transform, device):
    model.eval()
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2)
    
    return distance.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0005
    num_epochs = 100
    patience = 10  # for early stopping

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    data_dir = '/home/machvision/Documents/senior_design/senior-design/src/ml/dataset/vehicle_images_vault'  # Update this path
    train_dataset = CarDataset(root_dir=data_dir, transform=transform, split='train')
    val_dataset = CarDataset(root_dir=data_dir, transform=transform, split='val')
    test_dataset = CarDataset(root_dir=data_dir, transform=transform, split='test')
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, loss function, and optimizer
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
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

    # Plot loss history
    plot_loss_history(train_losses, val_losses)

    # Test the best model
    model.load_state_dict(torch.load('best_siamese_model.pth'))
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Example inference
    img1_path = '/path/to/image1.jpg'  # Update this path
    img2_path = '/path/to/image2.jpg'  # Update this path
    distance = infer(model, img1_path, img2_path, transform, device)
    print(f"Distance between images: {distance}")

if __name__ == "__main__":
    main()