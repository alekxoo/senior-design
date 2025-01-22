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
    #TODO: train yolov5 model on generic vehicle database to grab all vehicle objects and pass into cnn model for deeper classification
    yolov5_model_path = "./models/yolov5s.pt"  # grab yolov5 model weights
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


# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import models, transforms
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import yolov5
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
#         #changes: instead of instantiating YOLOv5 model, use the yolov5.load function and init conf/iou
#         yolov5_model = yolov5.load(model_path)
#         yolov5_model.conf = 0.5
#         yolov5_model.iou = 0.0
#         print("YOLOv5 model loaded successfully.")
#         return yolov5_model
#     except Exception as e:
#         print(f"Error loading YOLOv5 model: {e}")
#         return None

# # Class labels
# class_labels = ['mazda', 'audi', 'bmw', 'lexus', 'toyota']

# # Process an image and perform detection and classification
# def infer_image(yolov5_model, classification_model, device, input_image_path):
#     # Load the image
#     img = cv2.imread(input_image_path)
#     if img is None:
#         print("Error: Unable to load image.")
#         return

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Image transformation for classification
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     try:
#         # Run YOLOv5 detection
#         results = yolov5_model(img_rgb)

#         # Process each detection
#         for *xyxy, conf, cls in results.xyxy[0]:
#             if conf > 0.5:  # If detection confidence is high enough
#                 # Convert coordinates to integers
#                 x1, y1, x2, y2 = map(int, xyxy)

#                 # Ensure coordinates are within image bounds
#                 x1 = max(0, x1)
#                 y1 = max(0, y1)
#                 x2 = min(img.shape[1], x2)
#                 y2 = min(img.shape[0], y2)

#                 # Ensure ROI size is reasonable
#                 if x2 - x1 > 20 and y2 - y1 > 20:
#                     roi = img_rgb[y1:y2, x1:x2]
#                     roi_pil = Image.fromarray(roi)

#                     # Use the transform to preprocess the image
#                     roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

#                     with torch.no_grad():
#                         output = classification_model(roi_tensor)
#                         _, pred_class = torch.max(output, 1)
#                         predicted_class_name = class_labels[pred_class.item()]

#                     # Draw the bounding box and label
#                     label = f"{predicted_class_name} ({conf:.2f})"
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(img, label, (x1, y1 - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         # Display the image with detections
#         cv2.imshow("YOLO/CNN Test Interface", img)

#         # Optionally, save the output image
#         cv2.imwrite('output_image.jpg', img)

#         # Wait for key press to exit
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     except Exception as e:
#         print(f"Error during inference: {e}")

# def main():
#     cudnn.benchmark = True

#     # Device configuration
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Model paths
#     yolov5_model_path = "./models/yolov5s.pt"  # YOLOv5 model path
#     classification_model_path = "./CNNModels/best.pt"  # CNN model path

#     # Load the models
#     yolov5_model = load_yolov5_model(yolov5_model_path)
#     classification_model = load_classification_model(classification_model_path, device)

#     if yolov5_model is None or classification_model is None:
#         print("Failed to load one or more models.")
#         return

#     # Image path for processing
#     input_image_path = "./multi_vehicle.png"  # Specify the path to your image

#     # Perform inference on the image
#     infer_image(yolov5_model, classification_model, device, input_image_path)

# if __name__ == '__main__':
#     main()
