import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Function to load the trained CNN model from the .pt file
def load_classification_model(model_path, device):
    # Load the ResNet18 model architecture
    model = models.resnet18(weights=None)  # Start with a fresh ResNet18, no weights added
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # Modify the last layer for 5 classes (example)

    # Load the weights from the .pt file
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Function to process the input image and classify it
def classify_image(image_path, model, device):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')

    # Define the same preprocessing transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Apply the transformations and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Classify the image
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

# List of class labels (replace with your own class labels if different)
class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

def main():
    # Device configuration (use GPU if available, otherwise fallback to CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get the path to the pytorch model weights
    model_path = "./CNNModels/best.pt"  # Modify this with the correct path to your .pt file

    # Load the trained model
    model = load_classification_model(model_path, device)

    # Input image path for classification
    image_path = "testBMW.jpg" 

    # Classify the image
    predicted_class = classify_image(image_path, model, device)

    # Print the result using the index to class label mapping
    print(f"Predicted Class: {class_labels[predicted_class]}")

    # Display the image with the predicted label
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_labels[predicted_class]}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
