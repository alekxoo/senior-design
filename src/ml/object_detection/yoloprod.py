import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch.backends.cudnn as cudnn
from yolov5 import YOLOv5

# Function to load the trained classification model
def load_classification_model(model_path, device):
    # Load ResNet18 model
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes from your training
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Function to load the YOLOv5 model for object detection
def load_yolov5_model(model_path):
    yolov5_model = YOLOv5(model_path)
    return yolov5_model


class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# Function for object detection and classification
def infer(image_path, yolov5_model, classification_model, device):
    # Load image for YOLOv5
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use YOLOv5 to detect objects
    results = yolov5_model.predict(img_rgb)
    detections = results.xywh[0]  # detections (x, y, w, h, confidence, class)

    # Print detections
    print(f"Detected {len(detections)} objects.")

    # Loop over detections
    for detection in detections:
        x_center, y_center, width, height, confidence, class_idx = detection
        print(f"Class ID: {class_idx}, Confidence: {confidence}")
        
        # If confidence is above a threshold, classify the object
        if confidence > 0.5:
            # Extract the region of interest (ROI) from the image
            x1 = int((x_center - width / 2) * img.shape[1])
            y1 = int((y_center - height / 2) * img.shape[0])
            x2 = int((x_center + width / 2) * img.shape[1])
            y2 = int((y_center + height / 2) * img.shape[0])
            
            roi = img_rgb[y1:y2, x1:x2]
            roi_pil = Image.fromarray(roi)

            # Preprocess the image for classification
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

            # Predict the class of the object using the classification model
            with torch.no_grad():
                output = classification_model(roi_tensor)
                _, pred_class = torch.max(output, 1)
                predicted_class_name = class_labels[pred_class.item()]
                print(f"Predicted Class: {predicted_class_name}")

            # Draw bounding box and label on the image
            label = f"{predicted_class_name} ({confidence:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with detections
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def main():
    cudnn.benchmark = True

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the YOLOv5 and classification models
    yolov5_model_path = "./models/yolov5s.pt"  # Path to YOLOv5 weights
    classification_model_path = "./CNNModels/best.pt"  # Path to your custom trained model

    yolov5_model = load_yolov5_model(yolov5_model_path)
    classification_model = load_classification_model(classification_model_path, device)

    # Image for inference
    image_path = "testImage.jpg"  # Example image path for inference

    # Run inference
    infer(image_path, yolov5_model, classification_model, device)

if __name__ == '__main__':
    main()



# # Function to load the trained classification model
# def load_classification_model(model_path, device):
#     # Load ResNet18 model
#     model = models.resnet18(weights='IMAGENET1K_V1')
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes from your training
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)
#     model.eval()
#     return model

# # Function to load the YOLOv5 model for object detection
# def load_yolov5_model(model_path):
#     yolov5_model = YOLOv5(model_path)
#     return yolov5_model


# class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# # Function for object detection and classification
# def infer_webcam(yolov5_model, classification_model, device):
#     # Open the webcam
#     cap = cv2.VideoCapture(0)  # 0 is the default webcam
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, img = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Use YOLOv5 to detect objects
#         results = yolov5_model.predict(img_rgb)
#         detections = results.xywh[0]  # detections (x, y, w, h, confidence, class)

#         # Print detections
#         print(f"Detected {len(detections)} objects.")

#         # Loop over detections
#         for detection in detections:
#             x_center, y_center, width, height, confidence, class_idx = detection
#             print(f"Class ID: {class_idx}, Confidence: {confidence}")
            
#             # If confidence is above a threshold, classify the object
#             if confidence > 0.5:
#                 # Extract the region of interest (ROI) from the image
#                 x1 = int((x_center - width / 2) * img.shape[1])
#                 y1 = int((y_center - height / 2) * img.shape[0])
#                 x2 = int((x_center + width / 2) * img.shape[1])
#                 y2 = int((y_center + height / 2) * img.shape[0])
                
#                 roi = img_rgb[y1:y2, x1:x2]
#                 roi_pil = Image.fromarray(roi)

#                 # Preprocess the image for classification
#                 transform = transforms.Compose([
#                     transforms.Resize((224, 224)),
#                     transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])
#                 roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

#                 # Predict the class of the object using the classification model
#                 with torch.no_grad():
#                     output = classification_model(roi_tensor)
#                     _, pred_class = torch.max(output, 1)
#                     predicted_class_name = class_labels[pred_class.item()]
#                     print(f"Predicted Class: {predicted_class_name}")

#                 # Draw bounding box and label on the image
#                 label = f"{predicted_class_name} ({confidence:.2f})"
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         # Display the image with detections
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         cv2.imshow("Webcam Inference", img_rgb)

#         # Check for exit key (press 'q' to quit)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     cudnn.benchmark = True

#     # Device configuration
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Load the YOLOv5 and classification models
#     yolov5_model_path = "./models/yolov5s.pt"  # Path to YOLOv5 weights
#     classification_model_path = "./CNNModels/best.pt"  # Path to your custom trained model

#     yolov5_model = load_yolov5_model(yolov5_model_path)
#     classification_model = load_classification_model(classification_model_path, device)

#     # Run webcam inference
#     infer_webcam(yolov5_model, classification_model, device)

# if __name__ == '__main__':
#     main()
