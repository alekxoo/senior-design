import cv2
import torch
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for loading images
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
                continue
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

# Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64)  # Adjust the input size based on your image dimensions
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
            images = images.to(device)
            embeddings = model(images)
            
            for embedding, label in zip(embeddings, labels):
                if label.item() not in labels_list:
                    labels_list.append(label.item())
                    prototypes.append(embedding)
                else:
                    idx = labels_list.index(label.item())
                    prototypes[idx] = (prototypes[idx] + embedding) / 2  # Update prototype
    
    prototypes = torch.stack(prototypes)
    
    return prototypes, labels_list

def classify_query(model, query_image, prototypes, device):
    model.eval()
    
    query_image = query_image.unsqueeze(0).to(device)
    query_embedding = model(query_image)
    
    distances = torch.cdist(query_embedding, prototypes.unsqueeze(0)).squeeze(0)
    predicted_label = distances.argmin().item()
    
    return predicted_label, distances

def classify_car(model, query_image, prototypes, labels_list, device):
    query_image = query_image.unsqueeze(0).to(device)
    predicted_label, distances = classify_query(model, query_image, prototypes, device)
    return labels_list[predicted_label], distances

def process_frame(frame, model, prototypes, labels_list, device, yolo_model):
    # Resize the frame to 640x640 for YOLOv5
    img = cv2.resize(frame, (640, 640))
    
    # Convert the image to a tensor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = torch.from_numpy(img).float()  # Convert to tensor
    img /= 255.0  # Normalize to [0, 1]
    img = img.permute(2, 0, 1).unsqueeze(0)  # Change shape to (1, 3, 640, 640)

    # Detect cars using YOLOv5
    detections = detect_cars(img, yolo_model)

    # Process the detections as needed
    # Here you can draw boxes or process the detections further
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return frame

def detect_cars(img, yolo_model):
    # Perform inference
    pred = yolo_model(img)[0]  # Get predictions

    # Apply Non-Maximum Suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]  # Adjust thresholds as necessary

    # Convert predictions to a more manageable format
    detections = []
    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            detections.append((*xyxy, conf.item(), cls.item()))

    return detections


def main():
    # Set up dataset and data loader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset('car_data/support_set/', transform=transform)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # Initialize Prototypical Network
    model = PrototypicalNetwork().to(device)
    
    # Create class prototypes
    prototypes, labels_list = create_class_prototypes(model, data_loader, device)
    
    # Load YOLOv5 model
    yolo_model = attempt_load('yolov5s.pt')  # Adjust the path as necessary
    yolo_model.eval()
    
    # Process video feed
    cap = cv2.VideoCapture(1)  # Use 0 for webcam or provide video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, model, prototypes, labels_list, device, yolo_model)
        cv2.imshow('Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
