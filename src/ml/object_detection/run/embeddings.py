import os
import cv2
import torch
import numpy as np
import faiss
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Load ResNet50 for feature extraction
resnet_model = resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove last FC layer
resnet_model.eval()

# Image preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FAISS for fast similarity search
embedding_dim = 2048
index = faiss.IndexFlatL2(embedding_dim)
car_embeddings = {}  # Store client car embeddings

from PIL import Image
import torchvision.transforms as transforms

def extract_embedding(image):
    # Convert NumPy array to PIL image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Use the preloaded resnet_model for feature extraction
    with torch.no_grad():
        embedding = resnet_model(image).squeeze().numpy()  # Get the feature vector

    return embedding / np.linalg.norm(embedding)  # Normalize the embedding



def store_client_car(client_id, folder_path):
    """Extract embeddings from all images in the folder and store them in FAISS."""
    embeddings = []
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Warning: Could not read {img_path}, skipping...")
            continue
        embedding = extract_embedding(image)
        embeddings.append(embedding)

    if len(embeddings) == 0:
        print("🚨 No valid images found in the provided folder.")
        return

    embeddings = np.array(embeddings)
    car_embeddings[client_id] = embeddings
    index.add(embeddings)  # Add to FAISS
    print(f"✅ Stored {len(embeddings)} images for Client {client_id}.")
