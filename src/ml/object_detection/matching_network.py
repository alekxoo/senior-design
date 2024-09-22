import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# Feature Extractor using ResNet
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove last fully connected layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten the output

# Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(MatchingNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support_set, query_set):
        # Process each support image individually and store its embedding
        support_embeddings = []
        for img in support_set:
            # Remove the batch dimension if it exists
            if img.dim() == 5:  # [1, 5, 3, 224, 224]
                img = img.squeeze(0)  # Now it's [5, 3, 224, 224]
            if img.dim() == 4:  # [5, 3, 224, 224]
                embeddings = torch.cat([self.feature_extractor(i.unsqueeze(0)) for i in img])
            else:  # [3, 224, 224]
                embeddings = self.feature_extractor(img.unsqueeze(0))
            support_embeddings.append(embeddings)
        support_embeddings = torch.cat(support_embeddings, dim=0)

        # Process the query image
        if query_set.dim() == 4:  # [1, 3, 224, 224]
            query_embeddings = self.feature_extractor(query_set)
        else:  # [3, 224, 224]
            query_embeddings = self.feature_extractor(query_set.unsqueeze(0))

        # Ensure support_embeddings is 2D
        support_embeddings = support_embeddings.view(support_embeddings.size(0), -1)
        query_embeddings = query_embeddings.view(query_embeddings.size(0), -1)

        # Compute pairwise cosine similarity
        similarities = torch.mm(query_embeddings, support_embeddings.t())
        similarities = torch.softmax(similarities, dim=1)
        
        return similarities

# Custom Dataset for Few-Shot Learning
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
            support_imgs = torch.stack([self.transform(img) for img in support_imgs])
            query_img = self.transform(query_img)
        else:
            support_imgs = torch.stack([transforms.ToTensor()(img) for img in support_imgs])
            query_img = transforms.ToTensor()(query_img)

        return support_imgs, torch.tensor(support_labels), query_img, torch.tensor(query_label)

# Loss Function for Matching Network
def loss_fn(similarities, query_labels, support_labels):
    support_labels = support_labels.clone().detach() if isinstance(support_labels, torch.Tensor) else torch.tensor(support_labels)
    query_labels = query_labels.view(-1)
    loss = nn.CrossEntropyLoss()(similarities, query_labels)
    return loss

# Training Function for the Matching Network
def train_matching_network(model, data_loader, optimizer, epochs=10):
    model.train()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Image Preprocessing and Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example Data (Support and Query Sets) -- Replace with your actual image paths
support_imgs = [Image.open(f'car_data/support_set/{i}.jpg') for i in range(5)]
support_labels = [0, 1, 2, 3, 4]  # Example classes: 5 car models

query_imgs = [Image.open(f'car_data/query_set/{i}.jpg') for i in range(2)]
query_labels = [0, 1]  # Example labels for query images

# Prepare the Dataset
support_set = (support_imgs, support_labels)
query_set = (query_imgs, query_labels)

dataset = FewShotDataset(support_set, query_set, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize Feature Extractor and Matching Network
feature_extractor = FeatureExtractor()
model = MatchingNetwork(feature_extractor)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the Matching Network
train_matching_network(model, data_loader, optimizer, epochs=10)

# Inference (Prediction) Function
def classify_query(model, support_imgs, query_img):
    model.eval()
    with torch.no_grad():
        # Process support images
        support_embeddings = []
        for img in support_imgs:
            if img.dim() == 4:  # [1, 3, 224, 224]
                embedding = model.feature_extractor(img)
            else:  # [3, 224, 224]
                embedding = model.feature_extractor(img.unsqueeze(0))
            support_embeddings.append(embedding)
        support_embeddings = torch.cat(support_embeddings, dim=0)

        # Process query image
        if query_img.dim() == 4:  # [1, 3, 224, 224]
            query_embedding = model.feature_extractor(query_img)
        else:  # [3, 224, 224]
            query_embedding = model.feature_extractor(query_img.unsqueeze(0))

        # Ensure embeddings are 2D
        support_embeddings = support_embeddings.view(support_embeddings.size(0), -1)
        query_embedding = query_embedding.view(query_embedding.size(0), -1)

        # Compute similarities
        similarities = torch.mm(query_embedding, support_embeddings.t())
        predicted_class = torch.argmax(similarities, dim=1)
        return predicted_class

# Example usage: Classify a new query image
query_image = transform(Image.open('car_data/query_set/cybertruck.jpeg')).unsqueeze(0)
predicted_class = classify_query(model, torch.stack([transform(img) for img in support_imgs]), query_image)
print(f"Predicted Class: {predicted_class.item()}")