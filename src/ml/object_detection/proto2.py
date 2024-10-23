import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from rembg import remove



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Function to remove background and save the processed image
# def remove_background(input_path, output_path):
#     with open(input_path, 'rb') as i:
#         with open(output_path, 'wb') as o:
#             input = i.read()
#             output = remove(input)
#             o.write(output)

# # Process all images in the dataset
# def process_dataset(input_dir, output_dir):
#     for item in os.listdir(input_dir):
#         item_path = os.path.join(input_dir, item)
#         if os.path.isdir(item_path):
#             class_input_dir = item_path
#             class_output_dir = os.path.join(output_dir, item)
#             os.makedirs(class_output_dir, exist_ok=True)
            
#             for image_name in os.listdir(class_input_dir):
#                 if not image_name.startswith('.'):  # Skip hidden files
#                     input_path = os.path.join(class_input_dir, image_name)
#                     output_path = os.path.join(class_output_dir, image_name)
#                     remove_background(input_path, output_path)

# # Process the dataset
# input_dir = "../dataset/vehicle_images_vault"
# output_dir = "../dataset/vehicle_images_altered"
# process_dataset(input_dir, output_dir)


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
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
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
    
class EnhancedPrototypicalNetwork(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(EnhancedPrototypicalNetwork, self).__init__()
        
        # Load pretrained EfficientNetV2 as base model
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        
        # Get the output features of the model
        with torch.no_grad():
            # Create a dummy input to get the output size
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.base_model.forward_features(dummy_input)
            self.feature_size = features.shape[1] 
        
        #reset classifier to impplement our limited classes
        self.base_model.reset_classifier(0)
        
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        
        self.freeze_base_layers()
        
    def freeze_base_layers(self):
        
        layers_to_freeze = 0
        for i, (name, param) in enumerate(self.base_model.named_parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
                
    def forward(self, x):
        # pass through base model
        features = self.base_model.forward_features(x)
        
        embeddings = self.embedding_head(features)
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_class_prototypes(model, data_loader, device, temperature=0.1):
    model.eval()
    prototypes = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            embeddings = model(images)
            
            # averaging of embeddings
            embeddings = embeddings / temperature
            
            for embedding, label in zip(embeddings, labels):
                if label.item() not in labels_list:
                    labels_list.append(label.item())
                    prototypes.append(embedding)
                else:
                    idx = labels_list.index(label.item())
                    # Exponential moving average update
                    alpha = 0.9
                    prototypes[idx] = alpha * prototypes[idx] + (1 - alpha) * embedding
    
    # Stack and normalize prototypes
    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    return prototypes, labels_list

# Updated classification function with cosine similarity
def classify_query(model, query_image, prototypes, device, temperature=0.1):
    model.eval()
    
    query_image = query_image.unsqueeze(0).to(device)
    query_embedding = model(query_image)
    
    query_embedding = query_embedding / temperature
    
    # calc cosine similarity
    similarities = F.cosine_similarity(
        query_embedding.unsqueeze(1),
        prototypes.unsqueeze(0),
        dim=2
    )
    
    # convert these similarities to probabilities
    probabilities = F.softmax(similarities, dim=1)
    
    predicted_label = probabilities.argmax(dim=1).item()
    
    return predicted_label, probabilities.squeeze()


def load_and_preprocess_image(image_path, transform):
    """
    Load an image from the specified path and apply the provided transform.
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def classify_random_image(model, image_path, prototypes, labels_list, transform, device):
    """
    Load a random image, preprocess it, and classify it using the prototypical network.
    """
    image = load_and_preprocess_image(image_path, transform)
    image = image.to(device)  # Move image to device (GPU/CPU)
    
    predicted_label, probabilities = classify_query(model, image, prototypes, device)
    
    return predicted_label, probabilities

# Updated main function
def main():
    # Set up dataset and data loader with new transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    dataset = CustomImageDataset('../dataset/vehicle_images_vault/', 
                               transform=train_transform)
    test_dataset = CustomImageDataset('../dataset/vehicle_images_vault/', 
                                    transform=test_transform)
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    # Initialize enhanced model
    model = EnhancedPrototypicalNetwork(embedding_size=512).to(device)
    
    # Create prototypes
    prototypes, labels_list = create_class_prototypes(model, data_loader, 
                                                    device, temperature=0.1)
    
    # Example classification
    query_image, true_label = test_dataset[0]
    query_image = query_image.to(device)
    predicted_label, probabilities = classify_query(model, query_image, 
                                                 prototypes, device)
    
    print(f"True label: {test_dataset.classes[true_label]}")
    print(f"Predicted label: {test_dataset.classes[labels_list[predicted_label]]}")
    print("\nClass probabilities:")
    
    # Print sorted probabilities
    probs_dict = {test_dataset.classes[labels_list[i]]: prob.item() 
                  for i, prob in enumerate(probabilities)}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        print(f"  {class_name}: {prob:.4f}")


    #upload test image
    random_image_path = './testBMW.jpg'

    predicted_label, probabilities = classify_random_image(model, random_image_path, prototypes, 
                                                        labels_list, test_transform, device)

    print(f"\nPredicted label for random image: {dataset.classes[labels_list[predicted_label]]}")
    print("\nClass probabilities:")

    probs_dict = {dataset.classes[labels_list[i]]: prob.item() 
                for i, prob in enumerate(probabilities)}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_probs:
        print(f"  {class_name}: {prob:.4f}")

if __name__ == "__main__":
    main()

    