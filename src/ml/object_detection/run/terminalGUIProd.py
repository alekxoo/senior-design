import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import yaml
import threading
import time

# Load YAML config
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_class_data(data):
    return [cls['label'] for cls in data['classes']], data['num_classes']

# Classification model + preprocessing
def load_classification_model(path, num_classes, device):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_vehicle(model, roi_tensor, class_labels, logit_threshold=2, entropy_threshold=0.5):
    with torch.no_grad():
        output = model(roi_tensor)
        probabilities = F.softmax(output, dim=1)
        max_logit, pred_class = torch.max(output, 1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item()
        if max_logit.item() < logit_threshold or entropy < entropy_threshold:
            return "Unknown"
        return class_labels[pred_class.item()]

class TerminalVehicleTracker:
    def __init__(self):
        self.tracking_enabled = False
        self.selected_label = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()

        print("Loading YOLO model...")
        self.yolo_model = YOLO("./config/yolov5su.pt").to(self.device)

        print("Loading config...")
        yaml_data = load_yaml("./config/config_b490dad8.yaml")
        self.class_labels, num_classes = parse_class_data(yaml_data)
        for i, label in enumerate(self.class_labels):
            print(f"  {i + 1}. {label}")

        print("Loading classification model...")
        self.classifier = load_classification_model("./CNNModels/best.pt", num_classes, self.device)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam not found")

    def start(self):
        print("\nCommands:\n  list - List all detected vehicles\n  track <class> - Start tracking\n  stop - Stop tracking\n  exit - Quit\n")
        threading.Thread(target=self.process_loop, daemon=True).start()
        while True:
            cmd = input(">>> ").strip().lower()
            if cmd == "exit":
                break
            elif cmd == "list":
                print("Looking for vehicles...")
            elif cmd.startswith("track"):
                try:
                    _, label = cmd.split()
                    if label in self.class_labels:
                        self.tracking_enabled = True
                        self.selected_label = label
                        print(f"Tracking enabled for class: {label}")
                    else:
                        print(f"Invalid class. Available: {self.class_labels}")
                except:
                    print("Usage: track <class>")
            elif cmd == "stop":
                self.tracking_enabled = False
                self.selected_label = None
                print("Tracking stopped.")
            else:
                print("Unknown command.")

        self.cap.release()
        print("Exited.")

    def process_loop(self):
        while True:
            with self.lock:
                ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Camera read failed.")
                time.sleep(1)
                continue

            resized = cv2.resize(frame, (854, 480))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            results = self.yolo_model.predict(img_rgb, classes=[2], verbose=False, imgsz=480)
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    if conf > 0.7:
                        roi = img_rgb[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        roi_pil = Image.fromarray(roi)
                        roi_tensor = transform(roi_pil).unsqueeze(0).to(self.device)
                        label = classify_vehicle(self.classifier, roi_tensor, self.class_labels)

                        cx, cy = (x1 + x2)//2, (y1 + y2)//2
                        info = f"Detected: {label} at ({cx}, {cy})"
                        if self.tracking_enabled and label == self.selected_label:
                            print(f"[TRACKING] {info}")
                        else:
                            print(info)
            else:
                if self.tracking_enabled:
                    print("[TRACKING] Lost...")

            time.sleep(0.5)

if __name__ == '__main__':
    tracker = TerminalVehicleTracker()
    tracker.start()
