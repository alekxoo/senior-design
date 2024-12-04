import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the classification model
def load_classification_model(model_path, device):
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Classification model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

# Class labels
class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# Real-time webcam classification
# Real-time webcam classification with confidence threshold
def classify_webcam(classification_model, device, confidence_threshold=0.8):
    cap = cv2.VideoCapture(0)  # Use webcam (default index is 0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Preprocess the frame
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # Predict the class
            with torch.no_grad():
                output = classification_model(frame_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_prob, pred_class = torch.max(probabilities, 1)
                max_prob = max_prob.item()  # Extract the probability value
                pred_class_name = class_labels[pred_class.item()]

            # Check confidence threshold
            if max_prob >= confidence_threshold:
                label = f"{pred_class_name} ({max_prob:.2f})"
            else:
                label = "Uncertain"

            # Add label to the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Webcam Classification", frame)

        except Exception as e:
            print(f"Error during classification: {e}")

        # Check for exit key (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    cudnn.benchmark = True

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the classification model
    classification_model_path = "./CNNModels/best.pt"  # Path to your custom trained model
    classification_model = load_classification_model(classification_model_path, device)

    if classification_model is None:
        print("Failed to load the classification model.")
        return

    # Run webcam classification
    classify_webcam(classification_model, device)

if __name__ == '__main__':
    main()



