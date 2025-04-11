# For Jetson 

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import yaml
import warnings
import tkinter as tk
from tkinter import ttk

warnings.filterwarnings("ignore", category=FutureWarning)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=60,
    flip_method=0,
):
    return (
        "latency=0 ! nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_class_data(data):
    class_labels = [cls['label'] for cls in data['classes']]
    num_classes = data['num_classes']
    return class_labels, num_classes

class VehicleTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Tracker")
        # Load configuration and models
        yaml_data = load_yaml("../config/config_b490dad8.yaml")
        self.class_labels, num_classes = parse_class_data(yaml_data)
        
        # Set up device
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # Load models
        # print("Loading YOLOv9 model...")
        # self.yolov9_model = YOLO("../config/yolov5su.pt").to(self.device)
        #TODO: use later on jetson to test
        print("Loading YOLOv5su model...")
        self.yolov9_model = YOLO("./config/yolov5su.pt").to(self.device)

        # Export to TensorRT engine
        print("Exporting model to TensorRT engine...")
        self.yolov9_model.export(format="engine", device="cuda", half=True)

        self.yolov9_model = YOLO("yolov5su.engine")
        
        print("Loading classification model...")
        self.classification_model = models.resnet18(weights='IMAGENET1K_V1')
        self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
        self.classification_model.load_state_dict(torch.load("../CNNModels/best.pt"))
        self.classification_model.to(self.device)
        self.classification_model.eval()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize webcam
        #nk = "v4l2src device=/dev/video0 ! video/x-raw(memory:NVMM),format=H264,width=1280,height=720,framerate=30/1 ! appsink"
        nk = 'nvarguscamerasrc ! "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" ! nvvidconv ! appsink'
        #pipeline = 'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'

        self.cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=4032, capture_height=3040, framerate=21, display_width=1280, display_height=720, flip_method=0), cv2.CAP_GSTREAMER)
        #self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            self.master.destroy()
            return
        
        print(self.cap.get(cv2.CAP_PROP_FPS))
        # Setup GUI components
        self.tracking_enabled = False
        self.selected_label = tk.StringVar()
        self.track_status = tk.StringVar(value="Not Tracking")
        self.vehicle_position = tk.StringVar(value="N/A")
        
        # Create frame for bettreturner organization
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Dropdown for vehicle selection
        ttk.Label(main_frame, text="Select Vehicle:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.label_dropdown = ttk.Combobox(main_frame, textvariable=self.selected_label, values=self.class_labels)
        self.label_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.label_dropdown.current(0)
        
        # Start/Stop tracking button
        self.track_button = ttk.Button(main_frame, text="Start Tracking", command=self.toggle_tracking)
        self.track_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Display tracking status
        ttk.Label(main_frame, text="Tracking Status:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(main_frame, textvariable=self.track_status).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Display vehicle coordinates
        ttk.Label(main_frame, text="Vehicle Position:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(main_frame, textvariable=self.vehicle_position).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Add quit button
        self.quit_button = ttk.Button(main_frame, text="Quit", command=self.on_closing)
        self.quit_button.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Style the quit button to stand out
        style = ttk.Style()
        style.configure('Quit.TButton', foreground='red')
        self.quit_button.configure(style='Quit.TButton')
        
        # Schedule the first update of the webcam frame
        self.update_frame()
    
    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        self.track_button.config(text="Stop Tracking" if self.tracking_enabled else "Start Tracking")
        self.track_status.set("Tracking" if self.tracking_enabled else "Not Tracking")
        
        # Clear vehicle position when tracking is stopped
        if not self.tracking_enabled:
            self.vehicle_position.set("N/A")
    
    def update_frame(self):
        try:
            # Read frame from webcam
            ret, img = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                self.root.after(100, self.update_frame)
                return
            
            # Convert to RGB for YOLO
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.yolov9_model.predict(img_rgb, classes=[2], verbose=False, imgsz=480)
            
            try:
                # Try to use built-in plot function
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"Warning: Could not use built-in plotting: {e}")
                annotated_frame = img.copy()
            
            vehicle_found = False
            
            # Process detection results
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        
                        if conf > 0.7:
                            # Extract ROI for classification
                            if y2 > y1 and x2 > x1:  # Ensure valid dimensions
                                roi = img_rgb[y1:y2, x1:x2]
                                if roi.size > 0:  # Ensure ROI has content
                                    roi_pil = Image.fromarray(roi)
                                    roi_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)

                                    #TODO: check max logit and entropy filtering to ensure we handle OOD
                                    
                                    with torch.no_grad():
                                        output = self.classification_model(roi_tensor)
                                        pred_class = torch.argmax(output, dim=1).item()
                                        predicted_label = self.class_labels[pred_class]
                                        
                                        # Display the class label
                                        cv2.putText(annotated_frame, f"{predicted_label}", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                        
                                        if self.tracking_enabled and predicted_label == self.selected_label.get():
                                            vehicle_found = True
                                            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                                            self.vehicle_position.set(f"({x_center}, {y_center})")
                                            
                                            # Highlight tracked vehicle with green
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
                    except Exception as e:
                        print(f"Error processing box: {e}")
                        continue
            
            # Update tracking status and clear position if vehicle is lost
            is_tracking_lost = self.tracking_enabled and not vehicle_found
            self.track_status.set("Tracking" if self.tracking_enabled and vehicle_found else 
                                 "Lost" if is_tracking_lost else "Not Tracking")
            
            # Clear vehicle position if tracking is lost
            if is_tracking_lost:
                self.vehicle_position.set("N/A")
                
            # Show the frame in a window
            try:
                cv2.imshow("YOLOv9/CNN Tracking Interface", annotated_frame)
                cv2.waitKey(1)  # Process GUI events
            except Exception as e:
                print(f"Error displaying frame: {e}")
                
        except Exception as e:
            print(f"Error in processing loop: {e}")
        
        # Schedule the next frame update
        self.root.after(30, self.update_frame)  # Update approximately 33 times per second
    
    def on_closing(self):
        print("Shutting down...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VehicleTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()