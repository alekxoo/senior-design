import glob
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
from torchvision import models, transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from ultralytics import YOLO
import yaml
import warnings
import tkinter as tk
from tkinter import ttk, Label, messagebox
import threading
from threading import Thread
import time
import customtkinter as ctk
import customtkinter as ctk
import guiComponents
import os
import boto3
from dotenv import load_dotenv

import sys
sys.path.append("/home/machvision/Documents/senior-design/src/embedded")
from PIDControl import PID, PID_reset

load_dotenv()


warnings.filterwarnings("ignore", category=FutureWarning)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def parse_class_data(data):
    class_labels = [cls['label'] for cls in data['classes']]
    num_classes = data['num_classes']
    return class_labels, num_classes

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
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
        "video/x-raw, format=(string)BGR ! appsink sync=0 drop=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class VehicleTrackerApp:
    def on_model_download_success(self, username, racename, yaml_path=None, model_path=None):
        global USERNAME, RACENAME
        USERNAME = username
        RACENAME = racename
        print(f"Updated USERNAME to '{USERNAME}' and RACENAME to '{RACENAME}'")

        try:
            # Load YAML
            if yaml_path and os.path.isfile(yaml_path):
                yaml_data = load_yaml(yaml_path)
                self.class_labels, num_classes = parse_class_data(yaml_data)
                print(f"Parsed YAML. Classes: {self.class_labels}")
            else:
                print("YAML path not valid.")
                return 

            # Load classification model
            if model_path and os.path.isfile(model_path):
                print(f"Loading CNN model from: {model_path}")
                self.classification_model = models.resnet18(weights='IMAGENET1K_V1')
                self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
                self.classification_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.classification_model.to(self.device)
                self.classification_model.eval()
                print("CNN Model loaded")
            else:
                print(" Model path not valid.")
                return 

            #  After model loads, update the vehicle dropdown if it already exists
            if hasattr(self, "vehicle_menu") and self.vehicle_menu:
                self.vehicle_menu.configure(values=self.class_labels)
                if self.class_labels:
                    self.selected_label.set(self.class_labels[0])  # Auto-select first class
                print("Updated vehicle ComboBox with new class labels.")
            else:
                print("vehicle_menu not initialized yet — will show new labels on next UI refresh.")

        except Exception as e:
            print(f"Failed to load downloaded model: {e}")
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Tracker")
        self.current_mode = tk.StringVar(value="autonomous")

        #create instance of guiComponents for race info and model download, and also pass in s3 client for upload
        self.race_info_section = guiComponents.ModelInfoComponents(self, on_model_download_success=self.on_model_download_success)
        self.s3 = self.race_info_section.s3

        #create a thread locker for recording
        self.lock = threading.Lock()

        # Create a Label to display the video feed
        self.video_label = Label(self.root)
        self.video_label.pack()

        
        # Set up device
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load models
        print("Loading YOLOv9s model...")
        self.yolov9_model = YOLO("./yoloModels/yolov9s.pt").to(self.device)

        
        # Load configuration and models
        # yaml_data = load_yaml("./config/config_b490dad8.yaml")
        # self.class_labels, num_classes = parse_class_data(yaml_data)

        # print("Loading classification model...")
        # self.classification_model = models.resnet18(weights='IMAGENET1K_V1')
        # self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
        # self.classification_model.load_state_dict(torch.load("./config/best.pt"))
        # self.classification_model.to(self.device)
        # self.classification_model.eval()

        self.class_labels = []
        self.classification_model = None
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize webcam
        self.capture_framerate = 59
        self.cap = cv2.VideoCapture(gstreamer_pipeline(display_width=1920, display_height=1080, framerate=self.capture_framerate, capture_width=1920, capture_height=1080), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Tracking variables
        self.tracking_enabled = False
        self.selected_label = tk.StringVar()
        self.track_status = tk.StringVar(value="Not Tracking")
        self.vehicle_position = tk.StringVar(value="N/A")
        
        # Video recording variables
        self.is_recording = False
        self.out = None  # For writing video

        # Schedule the first update of the webcam frame
        self.create_ui()
        self.update_frame()

        #TODO: create instance of PTZ stuff

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

        self.record_button = ctk.CTkButton(self.recording_controls, text="Start Recording", command=self.record_and_save)
        self.record_button.pack(side="left", expand=True, padx=5)

        self.save_button = ctk.CTkButton(self.recording_controls, text="Upload Video", command=self.upload_video_to_s3)
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
            self.race_info_section.create_model_section(self.mode_content_frame)
            self.race_info_section.create_race_info_section(self.mode_content_frame)

        
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

    def record_and_save(self):
        """Starts or stops recording video safely."""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.configure(text="Stop Recording")

            output_folder = "VideoOutputs"
            os.makedirs(output_folder, exist_ok=True)

            self.temp_frames = []  # To store frames before writer is ready
            self.frame_timestamps = []  # To calculate real FPS
            self.actual_fps_computed = False
            self.output_path = os.path.join(output_folder, "output.mov")
            self.video_writer = None  # We'll init it later

            self.recording_thread = threading.Thread(target=self.record_video, daemon=True)
            self.recording_thread.start()
        else:
            self.is_recording = False
            self.record_button.configure(text="Starting Recording")

            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)

            with self.lock:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

            self.record_button.configure(text="Start Recording")


    def record_video(self):
        """Continuously records video frames in memory, estimates actual FPS, and initializes writer."""
        frame_count = 0
        start_time = time.time()

        while self.is_recording:
            try:
                with self.lock:
                    ret, frame = self.cap.read()

                if not ret:
                    print("WARNING: Frame capture failed!")
                    time.sleep(0.01)
                    continue

                frame_copy = cv2.resize(frame.copy(), (1920, 1080))
                current_time = time.time()

                if not self.actual_fps_computed:
                    self.temp_frames.append(frame_copy)
                    self.frame_timestamps.append(current_time)
                    frame_count += 1

                    if current_time - start_time >= 2.0:  # Wait ~2 seconds to stabilize
                        elapsed_time = self.frame_timestamps[-1] - self.frame_timestamps[0]
                        estimated_fps = frame_count / elapsed_time
                        estimated_fps = max(5, min(estimated_fps, 60))  # Clamp between 5 and 60
                        print(f"Estimated FPS: {estimated_fps:.2f}")

                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, estimated_fps, (1920, 1080))

                        for buffered_frame in self.temp_frames:
                            self.video_writer.write(buffered_frame)

                        self.temp_frames = []
                        self.frame_timestamps = []
                        self.actual_fps_computed = True
                else:
                    with self.lock:
                        if self.video_writer and self.is_recording:
                            self.video_writer.write(frame_copy)

                del frame_copy

            except Exception as e:
                print(f"ERROR in recording thread: {e}")
                time.sleep(0.1)





    def stop_recording(self):
        """Stops recording safely."""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None  # Prevent accidental access



    def upload_video_to_s3(self):
        """
        Uploads a video file to the specified S3 bucket under <username>/<racename>/video/.
        Looks for 'VideoOutputs/output.mp4' in a sibling folder relative to the current script.
        """

        global USERNAME, RACENAME
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the video file relative to the script
        video_path = os.path.join(current_dir, "..", "VideoOutputs", "output.mov")
        video_path = os.path.normpath(video_path)  # Clean up the path

        # Validate video file
        if not os.path.isfile(video_path):
            print(f"Error: File '{video_path}' not found.")
            return
        # Extract the filename from the path
        video_filename = os.path.basename(video_path)

        # Check file extension
        valid_extensions = (".mp4", ".mov", ".avi", ".mkv")
        if not video_filename.lower().endswith(valid_extensions):
            print("Error: Invalid file format. Allowed formats: .mp4, .mov, .avi, .mkv")
            return


        # Retrieve bucket name from environment variable
        bucket_name = os.getenv("S3_RACES_BUCKET_NAME")
        if not bucket_name:
            print("Error: S3_RACES_BUCKET_NAME environment variable is not set.")
            return
        
        # Construct the S3 key (path in S3 bucket)
        s3_key = f"{USERNAME}/{RACENAME}/video/{video_filename}"

        try:
            # Upload the video
            self.s3.upload_file(video_path, bucket_name, s3_key, ExtraArgs={'ContentType': 'video/quicktime'})
            messagebox.showinfo("Success", "Video has been uploaded to the cloud!")
            print(f"Upload successful: s3://{bucket_name}/{USERNAME}/{RACENAME}/video/")
        except Exception as e:
            messagebox.showinfo("Fail", f"Upload failed: {e}")

    #function to compute max logit of classification and entropy loss
    def classify_vehicle(self, roi_tensor, logit_threshold=2, entropy_threshold=0.5):
        """Runs CNN classification and applies both logit thresholding and entropy filtering."""
        with torch.no_grad():
            output = self.classification_model(roi_tensor)
            probabilities = F.softmax(output, dim=1)

            max_logit, pred_class = torch.max(output, 1)
            predicted_class_name = self.class_labels[pred_class.item()]
            max_logit = max_logit.item()
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item()

            # if max_logit < logit_threshold or entropy < entropy_threshold:
            #     return f"Unknown ({max_logit:.2f}, entropy: {entropy:.2f})"
            return f"{predicted_class_name} ({max_logit:.2f}, entropy: {entropy:.2f})"

    def update_frame(self):
        try:
            with self.lock:
                ret, img = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                self.root.after(100, self.update_frame)
                return

            old_time = time.time()
            
            # Scale down for inference to 480p
            img_resized = cv2.resize(img, (854, 480))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = self.yolov9_model.predict(img_rgb, classes=[2], verbose=False, imgsz=480)

            annotated_frame = img_resized.copy()
            vehicle_found = False
            vehicle_positions = []
            threads = []

            tracking_vehicle_x = 0.0
            tracking_vehicle_y = 0.0

            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for idx, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()

                    if conf > 0.3:
                        # Compute center coordinates
                        x_center, y_center = (x1 + x2) / (2*854) , (y1 + y2) / (2 * 480)
                        
                        # Extract ROI for classification
                        roi = img_rgb[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi_pil = Image.fromarray(roi)
                            roi_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)


                            # Get classification result synchronously
                            classification_result = self.classify_vehicle(roi_tensor)
                            vehicle_class_name = classification_result.split(" (")[0]  # Extract just the class name


                            # Run classification in a separate thread
                            thread = Thread(target=lambda: vehicle_positions.append(
                                f"Vehicle {idx+1}: {classification_result} ({x_center}, {y_center})"
                            ))
                            threads.append(thread)
                            thread.start()

                            # Draw bounding box
                            bbox_color = (255, 255, 255)  # Default white
                            if self.tracking_enabled and vehicle_class_name == self.selected_label.get(): 
                                bbox_color = (0, 255, 0)  # Green for tracked vehicle
                                vehicle_found = True
                                self.vehicle_position.set(f"({x_center}, {y_center})")
                                tracking_vehicle_x = x_center
                                tracking_vehicle_y = y_center

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)

            # Wait for all classification threads to finish
            for thread in threads:
                thread.join()

            # Display vehicle positions
            y_offset = annotated_frame.shape[0] - 40
            for position in vehicle_positions:
                cv2.putText(annotated_frame, position, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset -= 30

            # Handle tracking status
            is_tracking_lost = self.tracking_enabled and not vehicle_found
            self.track_status.set("Tracking" if self.tracking_enabled and vehicle_found else
                                "Lost" if is_tracking_lost else "Not Tracking")
            if is_tracking_lost:
                self.vehicle_position.set("N/A")
            
            if self.tracking_enabled:
                PID(tracking_vehicle_x, tracking_vehicle_y, (1.0 / 60.0), vehicle_found)
                new_time = time.time()
                print(new_time - old_time)
            else:
                PID(0.0, 0.0, 0.0, False) #make sure reset timer is running always

            # Convert frame for Tkinter display
            frame_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

        except Exception as e:
            print(f"Error in processing loop: {e}")

        # Schedule next update
        self.root.after(30, self.update_frame)

    def on_closing(self):
        print("Shutting down...")

        # Release camera
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

        # --- Delete .yaml and .pt files in ./config/ ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, "..", "config")
            config_dir = os.path.normpath(config_dir)

            if os.path.exists(config_dir) and os.path.isdir(config_dir):
                yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
                pt_files = glob.glob(os.path.join(config_dir, "*.pt"))

                if not yaml_files and not pt_files:
                    print("No .yaml or .pt files found in config directory.")
                else:
                    for file_path in yaml_files + pt_files:
                        try:
                            os.remove(file_path)
                            print(f"Deleted config file: {file_path}")
                        except Exception as file_err:
                            print(f"Failed to delete {file_path}: {file_err}")
            else:
                print("Config directory does not exist.")

        except Exception as e:
            print(f"Error deleting config files: {e}")
            messagebox.showerror("Error", f"Error deleting config files: {e}")

        # --- Delete video if it exists ---
        try:
            video_path = os.path.join(script_dir, "..", "VideoOutputs", "output.mov")
            video_path = os.path.normpath(video_path)

            print(f"Checking for video file at: {video_path}")

            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted video: {video_path}")
            else:
                print("No video file found.")
        except Exception as e:
            print(f"Error deleting video: {e}")
            messagebox.showerror("Error", f"Error deleting video: {e}")

        # --- Close the app window ---
        self.root.destroy()



def main():
    root = tk.Tk()
    app = VehicleTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
