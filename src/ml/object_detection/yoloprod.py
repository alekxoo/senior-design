# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from yolov5 import YOLOv5
# from torch.cuda.amp import autocast
# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# def load_classification_model(model_path, device):
#     try:
#         # Load ResNet18 model
#         model = models.resnet18(weights='IMAGENET1K_V1')
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes from your training
#         model.load_state_dict(torch.load(model_path))
#         model = model.to(device)
#         model.eval()
#         print("Classification model loaded successfully.")
#         return model
#     except Exception as e:
#         print(f"Error loading classification model: {e}")
#         return None

# def load_yolov5_model(model_path):
#     try:
#         yolov5_model = YOLOv5(model_path)
#         print("YOLOv5 model loaded successfully.")
#         return yolov5_model
#     except Exception as e:
#         print(f"Error loading YOLOv5 model: {e}")
#         return None

# class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# def infer_webcam(yolov5_model, classification_model, device):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     while True:
#         ret, img = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         try:
#             # Use YOLOv5 to detect objects
#             with autocast(enabled=torch.cuda.is_available()):
#                 results = yolov5_model.predict(img_rgb)
                
            
#             # Print detection information
#             print(f"Total detections: {len(results.xywh[0])}")
            
#             # Check if results.xywh is not empty
#             if len(results.xywh[0]) > 0:
#                 detections = results.xywh[0]

#                 for i, detection in enumerate(detections):
#                     x_center, y_center, width, height, confidence, class_idx = detection
                    
#                     print(f"Detection {i+1}: Confidence {confidence}, Class {class_labels[int(class_idx)]}, Class Index {class_idx}")
                    
#                     # If confidence is above a threshold, classify the object
#                     if confidence > 0.3: #lowered the confident threshold to test if it can classify anything but BMW
#                         # Extract the region of interest (ROI) from the image
#                         x1 = int((x_center - width / 2) * img.shape[1])
#                         y1 = int((y_center - height / 2) * img.shape[0])
#                         x2 = int((x_center + width / 2) * img.shape[1])
#                         y2 = int((y_center + height / 2) * img.shape[0])
                        
#                         # Ensure ROI is within image bounds
#                         x1 = max(0, x1)
#                         y1 = max(0, y1)
#                         x2 = min(img.shape[1], x2)
#                         y2 = min(img.shape[0], y2)
                        
#                         roi = img_rgb[y1:y2, x1:x2]
                        
#                         # Skip if ROI is too small
#                         if roi.size == 0:
#                             print("ROI is empty, skipping...")
#                             continue
                        
#                         roi_pil = Image.fromarray(roi)

#                         # Preprocess the image for classification
#                         roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

#                         # Predict the class of the object using the classification model
#                         with torch.no_grad():
#                             output = classification_model(roi_tensor)
#                             _, pred_class = torch.max(output, 1)
#                             predicted_class_name = class_labels[pred_class.item()]

#                         # Draw bounding box and label on the image
#                         label = f"{predicted_class_name} ({confidence:.2f})"
#                         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
#             # Display the image with detections
#             cv2.imshow("Webcam Inference", img)

#         except Exception as e:
#             print(f"Error during inference: {e}")

#         # Check for exit key (press 'q' to quit)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     cudnn.benchmark = True

#     # Device configuration
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Load the YOLOv5 and classification models
#     yolov5_model_path = "./models/carbest.pt"  # Path to YOLOv5 weights
#     classification_model_path = "./CNNModels/best.pt"  # Path to your custom trained model

#     yolov5_model = load_yolov5_model(yolov5_model_path)
#     classification_model = load_classification_model(classification_model_path, device)

#     if yolov5_model is None or classification_model is None:
#         print("Failed to load one or more models.")
#         return

#     # Run webcam inference
#     infer_webcam(yolov5_model, classification_model, device)

# if __name__ == '__main__':
#     main()


import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from yolov5 import YOLOv5
from torch.cuda.amp import autocast
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def load_classification_model(model_path, device):
    try:
        # Load ResNet18 model
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes from your training
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        print("Classification model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

def load_yolov5_model(model_path):
    try:
        yolov5_model = YOLOv5(model_path)
        print("YOLOv5 model loaded successfully.")
        return yolov5_model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

#TODO: find dynamic way by means of meta data or so in order to change class labels each time after training
class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

def infer_webcam(yolov5_model, classification_model, device):
    cap = cv2.VideoCapture(0)
    
    # Image transformation for classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # Run YOLOv5 detection
            results = yolov5_model.predict(img_rgb)

            # Process each detection
            for *xyxy, conf, cls in results.xyxy[0]:
                #check if confidence is high enough to classify based on CNN probabilities
                if conf > 0.7:
                    # convert coordinates to integers
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)

                    #check if ROI is within a reasonable size
                    #TODO: may need to edit this to be able to cover a larger region if needed while zooming in/out
                    if x2 - x1 > 20 and y2 - y1 > 20:
                        roi = img_rgb[y1:y2, x1:x2]
                        roi_pil = Image.fromarray(roi)

                        #use the transform to preprocess the image
                        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = classification_model(roi_tensor)
                            _, pred_class = torch.max(output, 1)
                            predicted_class_name = class_labels[pred_class.item()]

                        #draw the bounding box and label
                        label = f"{predicted_class_name} ({conf:.2f})"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the image with detections
            cv2.imshow("YOLO/CNN Test Interface", img)

        except Exception as e:
            print(f"Error during inference: {e}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cudnn.benchmark = True

    # device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Tested loading the default weights trained on COCO but started to classify everything so use carbest for now and then finetune later to just detect vehicle objects
    yolov5_model_path = "./models/carbest.pt"  # grab yolov5 model weights
    classification_model_path = "./CNNModels/best.pt"  # grab cnn model weights file

    #load model weights for both the detection and classification layers
    yolov5_model = load_yolov5_model(yolov5_model_path)
    classification_model = load_classification_model(classification_model_path, device)

    if yolov5_model is None or classification_model is None:
        print("Failed to load one or more models.")
        return

    
    infer_webcam(yolov5_model, classification_model, device)

if __name__ == '__main__':
    main()
