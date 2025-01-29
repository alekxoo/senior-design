from ultralytics import YOLO
import cv2

# Load YOLOv8 model
yolo_model = YOLO("./models/yolov9c.pt")

def detect_cars(frame):
    """Detect cars in the frame using YOLO."""
    results = yolo_model(frame, classes=[2]) #obtain the cars class on the weights
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            detections.append((x1, y1, x2, y2))
    return detections
