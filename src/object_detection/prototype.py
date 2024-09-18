import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load YOLOv5 model
yolo_model = attempt_load('yolov5s.pt')
yolo_model.eval()

# Siamese Network (placeholder)
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Define your Siamese Network architecture here
        # This is just a placeholder. You'll need to implement the actual architecture.
        self.conv = torch.nn.Conv2d(3, 64, 3)
        self.fc = torch.nn.Linear(64, 128)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

siamese_model = SiameseNetwork()
siamese_model.eval()

# Load and preprocess the reference car image
reference_car = cv2.imread('reference_car.jpg')
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
    similarity = torch.nn.functional.cosine_similarity(reference_features, car_features)
    return similarity.item()

def process_frame(frame):
    detections = detect_cars(frame)
    
    for *xyxy, conf, cls in detections:
        car_patch = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        similarity = compare_car(car_patch)
        
        color = (0, 255, 0) if similarity > 0.8 else (0, 0, 255)  # Green if similar, red otherwise
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        cv2.putText(frame, f'Car {similarity:.2f}', (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Main loop
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
