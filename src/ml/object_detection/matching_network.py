import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# 1. Feature Extractor using ResNet
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove last fully connected layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten the output

# 2. Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(MatchingNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_set, query_set):
        support_embeddings = self.feature_extractor(support_set)
        query_embeddings = self.feature_extractor(query_set)

        # Compute pairwise cosine similarity
        similarities = torch.mm(query_embeddings, support_embeddings.t())
        similarities = torch.softmax(similarities, dim=1)
        
        return similarities

# 3. Custom Dataset for Few-Shot Learning
class FewShotDataset(Dataset):
    def __init__(self, support_set, query_set, transform=None):
        self.support_set = support_set  # Tuple of (images, labels)
        self.query_set = query_set  # Tuple of (images, labels)
        self.transform = transform

    def __len__(self):
        return len(self.query_set[0])

    def __getitem__(self, idx):
        support_imgs, support_labels = self.support_set
        query_img, query_label = self.query_set[0][idx], self.query_set[1][idx]

        if self.transform:
            support_imgs = [self.transform(img) for img in support_imgs]
            query_img = self.transform(query_img)

        return torch.stack(support_imgs), torch.tensor(support_labels), query_img, torch.tensor(query_label)

# 4. Loss Function for Matching Network
def loss_fn(similarities, query_labels, support_labels):
    support_labels = torch.tensor(support_labels).unsqueeze(0).repeat(similarities.size(0), 1)
    loss = nn.CrossEntropyLoss()(similarities, query_labels)
    return loss

# 5. Training Function for the Matching Network
def train_matching_network(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (support_imgs, support_labels, query_img, query_label) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass
            similarities = model(support_imgs, query_img)

            # Compute loss
            loss = loss_fn(similarities, query_label, support_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}")

# 6. Image Preprocessing and Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 7. Example Data (Support and Query Sets) -- Replace with your actual image paths
support_imgs = [Image.open(f'car_data/support/car_model_{i}.jpg') for i in range(5)]
support_labels = [0, 1, 2, 3, 4]  # Example classes: 5 car models

query_imgs = [Image.open(f'car_data/query/car_{i}.jpg') for i in range(2)]
query_labels = [0, 1]  # Example labels for query images

# Prepare the Dataset
support_set = (support_imgs, support_labels)
query_set = (query_imgs, query_labels)

dataset = FewShotDataset(support_set, query_set, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 8. Initialize Feature Extractor and Matching Network
feature_extractor = FeatureExtractor()
model = MatchingNetwork(feature_extractor)

# 9. Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 10. Train the Matching Network
train_matching_network(model, data_loader, optimizer, epochs=10)

# 11. Inference (Prediction) Example
def classify_query(model, support_imgs, query_img):
    model.eval()
    with torch.no_grad():
        support_embeddings = model.feature_extractor(support_imgs)
        query_embedding = model.feature_extractor(query_img.unsqueeze(0))

        similarities = torch.mm(query_embedding, support_embeddings.t())
        predicted_class = torch.argmax(similarities, dim=1)
        return predicted_class

# Example usage: Classify a new query image
query_image = transform(Image.open('car_data/query/new_car.jpg')).unsqueeze(0)
predicted_class = classify_query(model, torch.stack([transform(img) for img in support_imgs]), query_image)
print(f"Predicted Class: {predicted_class.item()}")
