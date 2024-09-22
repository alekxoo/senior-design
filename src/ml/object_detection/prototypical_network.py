# #create 5 classes, 5 images each within support_set folder


# import os
# import random
# from typing import List, Tuple

# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import resnet18
# from torchvision.datasets import ImageFolder
# from PIL import Image

# class PrototypicalNetworks(nn.Module):
#     def __init__(self, backbone: nn.Module):
#         super(PrototypicalNetworks, self).__init__()
#         self.backbone = backbone

#     def forward(
#         self,
#         support_images: torch.Tensor,
#         support_labels: torch.Tensor,
#         query_images: torch.Tensor,
#     ) -> torch.Tensor:
#         z_support = self.backbone.forward(support_images)
#         z_query = self.backbone.forward(query_images)

#         n_way = len(torch.unique(support_labels))
#         z_proto = torch.cat(
#             [
#                 z_support[torch.nonzero(support_labels == label)].mean(0)
#                 for label in range(n_way)
#             ]
#         )

#         dists = torch.cdist(z_query, z_proto)
#         scores = -dists
#         return scores

# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#         self.images = self._load_images()

#     def _load_images(self):
#         images = []
#         valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}  # Add any other valid image extensions

#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 # Only load files with valid image extensions
#                 if os.path.splitext(img_name)[1].lower() in valid_extensions:
#                     images.append((img_path, self.class_to_idx[class_name]))
#         return images


#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label

# def create_episode(dataset: CustomImageDataset, n_way: int, n_shot: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
#     # Filter classes with enough images
#     valid_classes = [cls for cls in dataset.classes 
#                      if len([img for img, img_label in dataset.images if dataset.classes[img_label] == cls]) >= n_shot + n_query]
    
#     if len(valid_classes) < n_way:
#         raise ValueError(f"Not enough classes with {n_shot + n_query} images. Only {len(valid_classes)} valid classes found.")
    
#     classes = random.sample(valid_classes, n_way)
#     support_images = []
#     support_labels = []
#     query_images = []
#     query_labels = []

#     for label, class_name in enumerate(classes):
#         class_images = [img for img, img_label in dataset.images if dataset.classes[img_label] == class_name]
#         selected_images = random.sample(class_images, n_shot + n_query)
        
#         for i, img_path in enumerate(selected_images):
#             img = Image.open(img_path).convert('RGB')
#             img = dataset.transform(img)
            
#             if i < n_shot:
#                 support_images.append(img)
#                 support_labels.append(label)
#             else:
#                 query_images.append(img)
#                 query_labels.append(label)

#     support_images = torch.stack(support_images)
#     support_labels = torch.tensor(support_labels)
#     query_images = torch.stack(query_images)
#     query_labels = torch.tensor(query_labels)

#     return support_images, support_labels, query_images, query_labels, classes

# def main(image_folder: str, n_way: int = 5, n_shot: int = 5, n_query: int = 5, n_episodes: int = 100):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     dataset = CustomImageDataset(image_folder, transform=transform)
    
#     # Print dataset statistics
#     print(f"Total number of classes: {len(dataset.classes)}")
#     for cls in dataset.classes:
#         class_images = [img for img, img_label in dataset.images if dataset.classes[img_label] == cls]
#         print(f"Class '{cls}' has {len(class_images)} images")
    
#     convolutional_network = resnet18(pretrained=True)
#     convolutional_network.fc = nn.Flatten()
#     model = PrototypicalNetworks(convolutional_network).to(device)
#     model.eval()

#     total_accuracy = 0

#     for episode in range(n_episodes):
#         try:
#             support_images, support_labels, query_images, query_labels, episode_classes = create_episode(dataset, n_way, n_shot, n_query)
#         except ValueError as e:
#             print(f"Error creating episode: {e}")
#             print("Try reducing n_way, n_shot, or n_query.")
#             return

#         support_images = support_images.to(device)
#         support_labels = support_labels.to(device)
#         query_images = query_images.to(device)
#         query_labels = query_labels.to(device)

#         with torch.no_grad():
#             scores = model(support_images, support_labels, query_images)
#             _, predicted_labels = torch.max(scores.data, 1)
#             accuracy = (predicted_labels == query_labels).float().mean().item()
#             total_accuracy += accuracy

#         print(f"Episode {episode + 1}/{n_episodes}: Accuracy = {accuracy:.4f}")

#     average_accuracy = total_accuracy / n_episodes
#     print(f"\nAverage accuracy over {n_episodes} episodes: {average_accuracy:.4f}")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Few-shot classification with Prototypical Networks")
#     parser.add_argument("image_folder", type=str, help="Path to the image folder")
#     parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode")
#     parser.add_argument("--n_shot", type=int, default=3, help="Number of support examples per class")
#     parser.add_argument("--n_query", type=int, default=2, help="Number of query examples per class")
#     parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to evaluate")
    
#     args = parser.parse_args()
    
#     main(args.image_folder, args.n_way, args.n_shot, args.n_query, args.n_episodes)




import os
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}  # Only load valid image files
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.splitext(img_name)[1].lower() in valid_extensions:
                    images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def create_episode(dataset: CustomImageDataset, n_way: int, n_shot: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    # Filter classes with enough images
    valid_classes = [cls for cls in dataset.classes
                     if len([img for img, img_label in dataset.images if dataset.classes[img_label] == cls]) >= n_shot + n_query]

    if len(valid_classes) < n_way:
        raise ValueError(f"Not enough classes with {n_shot + n_query} images. Only {len(valid_classes)} valid classes found.")

    classes = random.sample(valid_classes, n_way)
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for label, class_name in enumerate(classes):
        class_images = [img for img, img_label in dataset.images if dataset.classes[img_label] == class_name]
        selected_images = random.sample(class_images, n_shot + n_query)

        for i, img_path in enumerate(selected_images):
            img = Image.open(img_path).convert('RGB')
            img = dataset.transform(img)

            if i < n_shot:
                support_images.append(img)
                support_labels.append(label)
            else:
                query_images.append(img)
                query_labels.append(label)

    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels, classes


def predict_new_image(model, new_image_path: str, dataset: CustomImageDataset, support_images: torch.Tensor, support_labels: torch.Tensor, device: torch.device):
    # Load and preprocess the new image
    new_image = Image.open(new_image_path).convert('RGB')
    new_image = dataset.transform(new_image)
    new_image = new_image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Pass support set and the new image (query image) through the model
    with torch.no_grad():
        scores = model(support_images, support_labels, new_image)
        _, predicted_label = torch.max(scores.data, 1)

    predicted_class = dataset.classes[predicted_label.item()]
    return predicted_class


def main(image_folder: str, n_way: int = 5, n_shot: int = 5, n_query: int = 5, n_episodes: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomImageDataset(image_folder, transform=transform)

    # Print dataset statistics
    print(f"Total number of classes: {len(dataset.classes)}")
    for cls in dataset.classes:
        class_images = [img for img, img_label in dataset.images if dataset.classes[img_label] == cls]
        print(f"Class '{cls}' has {len(class_images)} images")

    convolutional_network = resnet50(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(convolutional_network).to(device)
    model.eval()

    total_accuracy = 0

    for episode in range(n_episodes):
        try:
            support_images, support_labels, query_images, query_labels, episode_classes = create_episode(dataset, n_way, n_shot, n_query)
        except ValueError as e:
            print(f"Error creating episode: {e}")
            print("Try reducing n_way, n_shot, or n_query.")
            return

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        with torch.no_grad():
            scores = model(support_images, support_labels, query_images)
            _, predicted_labels = torch.max(scores.data, 1)
            accuracy = (predicted_labels == query_labels).float().mean().item()
            total_accuracy += accuracy

        print(f"Episode {episode + 1}/{n_episodes}: Accuracy = {accuracy:.4f}")

    average_accuracy = total_accuracy / n_episodes
    print(f"\nAverage accuracy over {n_episodes} episodes: {average_accuracy:.4f}")

    # Test with a new image
    new_image_path = "car_data/query_set/testChevy.jpg"  # Replace with your actual image path
    support_images, support_labels, _, _, _ = create_episode(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query)
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)

    predicted_class = predict_new_image(model, new_image_path, dataset, support_images, support_labels, device)
    print(f"Predicted class for the new image: {predicted_class}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot classification with Prototypical Networks")
    parser.add_argument("image_folder", type=str, help="Path to the image folder")
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n_shot", type=int, default=3, help="Number of support examples per class")
    parser.add_argument("--n_query", type=int, default=2, help="Number of query examples per class")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to evaluate")

    args = parser.parse_args()

    main(args.image_folder, args.n_way, args.n_shot, args.n_query, args.n_episodes)

