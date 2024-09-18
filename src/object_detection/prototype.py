import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the parent directory of 'senior-design'
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

# Add YOLOv5 to the system path
yolo_path = os.path.join(project_root, 'yolov5')
if yolo_path not in sys.path:
    sys.path.append(yolo_path)

# Now import YOLOv5 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load YOLOv5 model
yolo_model = attempt_load('yolov5s.pt')
yolo_model.eval()

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Calculate the flattened size
        self.fc_input_size = 384 * 12 * 12

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        print(f"Input shape: {x.shape}")
        output = self.cnn1(x)
        print(f"After CNN shape: {output.shape}")
        output = output.view(output.size()[0], -1)
        print(f"After flatten shape: {output.shape}")
        output = self.fc1(output)
        print(f"Final output shape: {output.shape}")
        return output

    def forward(self, input1, input2=None):
        output1 = self.forward_once(input1)
        if input2 is not None:
            output2 = self.forward_once(input2)
            return output1, output2
        return output1

# Instantiate the model
siamese_model = SiameseNetwork()
siamese_model.eval()
# Load and preprocess the reference car image
reference_car = cv2.imread('cybertruck.jpeg')
reference_car = cv2.resize(reference_car, (224, 224))  # Adjust size as needed
reference_car = torch.from_numpy(reference_car).float().permute(2, 0, 1).unsqueeze(0) / 255.0
with torch.no_grad():
    reference_features = siamese_model(reference_car)

def detect_cars(frame):
    img = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = yolo_model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[2])  # 2 is the class index for 'car' in COCO
    return pred[0]

def compare_car(car_patch):
    car_patch = cv2.resize(car_patch, (224, 224))  # Resize to match reference image
    car_patch = torch.from_numpy(car_patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        car_features = siamese_model(car_patch)
    similarity = F.pairwise_distance(reference_features, car_features)
    
    return similarity.item()

def process_frame(frame):
    detections = detect_cars(frame)
    for *xyxy, conf, cls in detections:
        car_patch = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        similarity = compare_car(car_patch)
        color = (0, 255, 0) if similarity < 0.5 else (0, 0, 255)  # Green if similar (lower distance), red otherwise
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        cv2.putText(frame, f'Car {similarity:.2f}', (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow('Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()