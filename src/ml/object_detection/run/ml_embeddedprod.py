import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import yaml
import warnings
import tkinter as tk
from tkinter import ttk, Label
import threading
import time
import customtkinter as ctk

warnings.filterwarnings("ignore", category=FutureWarning)

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
        self.current_mode = tk.StringVar(value="autonomous")

        # Create a Label to display the video feed
        self.video_label = Label(self.root)
        self.video_label.pack()

        # Load configuration and models
        yaml_data = load_yaml("./config/config_b490dad8.yaml")
        self.class_labels, num_classes = parse_class_data(yaml_data)
        
        # Set up device
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load models
        print("Loading YOLOv9 model...")
        self.yolov9_model = YOLO("./config/yolov9c.pt").to(self.device)
        
        print("Loading classification model...")
        self.classification_model = models.resnet18(weights='IMAGENET1K_V1')
        self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
        self.classification_model.load_state_dict(torch.load("./CNNModels/best.pt"))
        self.classification_model.to(self.device)
        self.classification_model.eval()
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 1080p resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 1080p resolution
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Tracking variables
        self.tracking_enabled = False
        self.selected_label = tk.StringVar()
        self.track_status = tk.StringVar(value="Not Tracking")
        self.vehicle_position = tk.StringVar(value="N/A")
        
        # Video recording variables
        self.recording = False
        self.out = None  # For writing video

        # Schedule the first update of the webcam frame
        self.create_ui()
        self.update_frame()

    import customtkinter as ctk

    def create_ui(self):
        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel - Video Feed
        left_panel = ctk.CTkFrame(main_container, width=550)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        self.video_label = ctk.CTkLabel(left_panel, text="")  # Placeholder for video feed
        self.video_label.pack(fill="both", expand=True)

        self.recording_controls = ctk.CTkFrame(left_panel)
        self.recording_controls.pack(fill="x", pady=(0, 10))

        self.record_button = ctk.CTkButton(self.recording_controls, text="Start Recording", command=self.start_recording)
        self.record_button.pack(side="left", expand=True, padx=5)

        self.save_button = ctk.CTkButton(self.recording_controls, text="Save Video", command=self.save_video, state="disabled")
        self.save_button.pack(side="right", expand=True, padx=5)

        # Right panel - Controls
        self.right_panel = ctk.CTkFrame(main_container, width=600)
        self.right_panel.pack(side="right", fill="both", padx=(10, 0), expand=True)
        self.right_panel.pack_propagate(False)

        # Mode selection buttons
        mode_frame = ctk.CTkFrame(self.right_panel)
        mode_frame.pack(fill="x", pady=(0, 10))

        autonomous_btn = ctk.CTkButton(mode_frame, text="Autonomous", width=160, height=32,
                                       command=lambda: self.switch_mode("autonomous"))
        autonomous_btn.pack(side="left", expand=True, padx=5)

        ptz_btn = ctk.CTkButton(mode_frame, text="PTZ Control", width=160, height=32,
                                command=lambda: self.switch_mode("ptz"))
        ptz_btn.pack(side="left", expand=True, padx=5)

        quit_btn = ctk.CTkButton(mode_frame, text = "Quit", width=160, height=32, command=lambda: self.switch_mode("quit"))
        quit_btn.pack(side="right", expand=True, padx=5)


        # Dynamic content area for additional features
        self.mode_content_frame = ctk.CTkFrame(self.right_panel)
        self.mode_content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        
        self.update_controls()

    def switch_mode(self, mode):
        self.current_mode.set(mode)
        self.update_controls()

    def update_controls(self):
        # Clear current controls
        for widget in self.mode_content_frame.winfo_children():
            widget.destroy()
        
        if self.current_mode.get() == "autonomous":
            ctk.CTkLabel(self.mode_content_frame, text="Vehicle Tracking", font=("Arial", 14)).pack()
            self.track_button = ctk.CTkButton(self.mode_content_frame, text="Start Tracking", command=self.toggle_tracking)
            self.track_button.pack(pady=5)
            ctk.CTkLabel(self.mode_content_frame, text="Select Vehicle:").pack()
            self.vehicle_menu = ctk.CTkComboBox(self.mode_content_frame, values=self.class_labels, variable=self.selected_label)
            self.vehicle_menu.pack()
            ctk.CTkLabel(self.mode_content_frame, text="Status:").pack()
            ctk.CTkLabel(self.mode_content_frame, textvariable=self.track_status, font=("Arial", 12)).pack()
            ctk.CTkLabel(self.mode_content_frame, text="Position:").pack()
            ctk.CTkLabel(self.mode_content_frame, textvariable=self.vehicle_position, font=("Arial", 12)).pack()
        
        elif self.current_mode.get() == "ptz":
            ctk.CTkLabel(self.mode_content_frame, text="PTZ Control", font=("Arial", 14)).pack()
            ctk.CTkLabel(self.mode_content_frame, text="Pan:").pack()
            ctk.CTkSlider(self.mode_content_frame, from_=0, to=100).pack(fill="x", padx=10)
            ctk.CTkLabel(self.mode_content_frame, text="Tilt:").pack()
            ctk.CTkSlider(self.mode_content_frame, from_=0, to=100).pack(fill="x", padx=10)
            ctk.CTkLabel(self.mode_content_frame, text="Zoom:").pack()
            ctk.CTkSlider(self.mode_content_frame, from_=0, to=100).pack(fill="x", padx=10)
        
        elif self.current_mode.get() == "quit":
            self.on_closing()

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        self.track_button.configure(text="Stop Tracking" if self.tracking_enabled else "Start Tracking")
        self.track_status.set("Tracking" if self.tracking_enabled else "Not Tracking")
        if not self.tracking_enabled:
            self.vehicle_position.set("N/A")

    def start_recording(self):
        """Start video recording."""
        if not self.recording:
            self.recording = True
            self.record_button.configure(text="Stop Recording")
            self.save_button.configure(state=tk.DISABLED)  # Disable save while recording
            self.start_video_writer()
        else:
            self.recording = False
            self.record_button.configure(text="Start Recording")
            self.save_button.configure(state=tk.NORMAL)  # Enable save once stopped

    def start_video_writer(self):
        """Initialize video writer in a separate thread."""
        def record_video():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1920, 1080))  # 1080p video
            while self.recording:
                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)  # Write original 1080p frame to file
                time.sleep(0.03)  # Approx 30 fps

            if self.out:
                self.out.release()

        # Run video writer in a separate thread
        threading.Thread(target=record_video, daemon=True).start()

    def save_video(self):
        """Save video to file."""
        print("Saving video...")

    def update_frame(self):
        try:
            ret, img = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                self.root.after(100, self.update_frame)
                return

            # Scale down for inference to 480p
            img_resized = cv2.resize(img, (854, 480))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = self.yolov9_model.predict(img_rgb, classes=[2], verbose=False, imgsz=480)

            # Ensure results exist
            annotated_frame = img_resized.copy()
            vehicle_found = False

            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()

                    if conf > 0.7:
                        # Default bounding box color (white)
                        bbox_color = (255, 255, 255)

                        # Extract ROI and classify
                        roi = img_rgb[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi_pil = Image.fromarray(roi)
                            roi_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                output = self.classification_model(roi_tensor)
                                pred_class = torch.argmax(output, dim=1).item()
                                predicted_label = self.class_labels[pred_class]

                                # If tracking is enabled and this is the target vehicle, change bounding box to green
                                if self.tracking_enabled and predicted_label == self.selected_label.get():
                                    bbox_color = (0, 255, 0)  # Green for tracked vehicle
                                    vehicle_found = True
                                    x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                                    self.vehicle_position.set(f"({x_center}, {y_center})")

                                # Display class label
                                cv2.putText(annotated_frame, f"{predicted_label} ({conf:.2f})", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Draw bounding box with determined color
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)

            # Handle tracking status
            is_tracking_lost = self.tracking_enabled and not vehicle_found
            self.track_status.set("Tracking" if self.tracking_enabled and vehicle_found else
                                "Lost" if is_tracking_lost else "Not Tracking")

            if is_tracking_lost:
                self.vehicle_position.set("N/A")

            # Convert frame for Tkinter display
            frame_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update label with the new frame
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

        except Exception as e:
            print(f"Error in processing loop: {e}")

        # Schedule next update
        self.root.after(30, self.update_frame)

    def on_closing(self):
        print("Shutting down...")
        if self.cap.isOpened():
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