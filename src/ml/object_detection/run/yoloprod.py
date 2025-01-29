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

# #TODO: find dynamic way by means of meta data or so in order to change class labels each time after training
# class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# def infer_webcam(yolov5_model, classification_model, device):
#     cap = cv2.VideoCapture(0)
    
#     # Image transformation for classification
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     while True:
#         ret, img = cap.read()
#         if not ret:
#             break

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         try:
#             # Run YOLOv5 detection
#             results = yolov5_model.predict(img_rgb)

#             # Process each detection
#             for *xyxy, conf, cls in results.xyxy[0]:
#                 #check if confidence is high enough to classify based on CNN probabilities
#                 if conf > 0.7:
#                     # convert coordinates to integers
#                     x1, y1, x2, y2 = map(int, xyxy)
                    
#                     # ensure coordinates are within image bounds
#                     x1 = max(0, x1)
#                     y1 = max(0, y1)
#                     x2 = min(img.shape[1], x2)
#                     y2 = min(img.shape[0], y2)

#                     #check if ROI is within a reasonable size
#                     #TODO: may need to edit this to be able to cover a larger region if needed while zooming in/out
#                     if x2 - x1 > 20 and y2 - y1 > 20:
#                         roi = img_rgb[y1:y2, x1:x2]
#                         roi_pil = Image.fromarray(roi)

#                         #use the transform to preprocess the image
#                         roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

#                         with torch.no_grad():
#                             output = classification_model(roi_tensor)
#                             _, pred_class = torch.max(output, 1)
#                             predicted_class_name = class_labels[pred_class.item()]

#                         #draw the bounding box and label
#                         label = f"{predicted_class_name} ({conf:.2f})"
#                         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                         cv2.putText(img, label, (x1, y1 - 10), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#             # Display the image with detections
#             cv2.imshow("YOLO/CNN Test Interface", img)

#         except Exception as e:
#             print(f"Error during inference: {e}")

#         # Exit on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     cudnn.benchmark = True

#     # device configuration
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     #Tested loading the default weights trained on COCO but started to classify everything so use carbest for now and then finetune later to just detect vehicle objects
#     #TODO: train yolov5 model on generic vehicle database to grab all vehicle objects and pass into cnn model for deeper classification
#     yolov5_model_path = "./models/yolov5s.pt"  # grab yolov5 model weights
#     classification_model_path = "./CNNModels/best.pt"  # grab cnn model weights file

#     #load model weights for both the detection and classification layers
#     yolov5_model = load_yolov5_model(yolov5_model_path)
#     classification_model = load_classification_model(classification_model_path, device)

#     if yolov5_model is None or classification_model is None:
#         print("Failed to load one or more models.")
#         return

    
#     infer_webcam(yolov5_model, classification_model, device)

# if __name__ == '__main__':
#     main()



####using YOLOV9
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ultralytics import YOLO  # Using YOLOv9
import yaml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_class_data(data):
    class_labels = [cls['label'] for cls in data['classes']]
    num_classes = data['num_classes']
    return class_labels, num_classes

def load_classification_model(model_path, device, classes):
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        print("Classification model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

def load_yolov9_model(model_path):
    try:
        yolov9_model = YOLO(model_path)  # Load YOLOv9 model
        print("YOLOv9 model loaded successfully.")
        return yolov9_model
    except Exception as e:
        print(f"Error loading YOLOv9 model: {e}")
        return None


def infer_webcam(yolov9_model, classification_model, device, labels):
    cap = cv2.VideoCapture(0)
    
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
            results = yolov9_model.predict(img_rgb, classes=[2])  # Car class
            annotated_frame = results[0].plot()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()

                    if conf > 0.7:
                        roi = img_rgb[y1:y2, x1:x2]
                        roi_pil = Image.fromarray(roi)
                        roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = classification_model(roi_tensor)
                            _, pred_class = torch.max(output, 1)
                            predicted_class_name = labels[pred_class.item()]

                        label = f"{predicted_class_name} ({conf:.2f})"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow("YOLOv9/CNN Test Interface", annotated_frame)

        except Exception as e:
            print(f"Error during inference: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolov9_model_path = "./config/yolov9c.pt"
    classification_model_path = "./CNNModels/best.pt"

    yaml_data = load_yaml("./config/config_b490dad8.yaml") #dynamically load data from different sources (maybe include information from the user we want to load)
    class_labels, num_classes = parse_class_data(yaml_data)
    print("Class labels:", class_labels)
    print("Number of classes:", num_classes)

    yolov9_model = load_yolov9_model(yolov9_model_path)
    classification_model = load_classification_model(classification_model_path, device, num_classes)

    if yolov9_model is None or classification_model is None:
        print("Failed to load one or more models.")
        return

    infer_webcam(yolov9_model, classification_model, device, class_labels)

if __name__ == '__main__':
    main()


"""
Notes for software dev:
- When getting information from new user-probably follow this workload
    - get user name/ID if needed
    - get number of cars they are uploading(not # of photos but how many unique cars)
    - get user to add labels for each class of cars
    - get user to upload images
    - Use the yamlGen.py file to generate the yaml file and store in the database under their name

ML dev:
 When training:
    - obtain images from the database along with the respective yaml file generated
    - EDIT training script to use labels and num_classes from yaml into our variables
When running:
    -Use the yaml file again to use labels and num_classes
"""