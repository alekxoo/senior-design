# In gui_components.py, update the imports at the top:
import customtkinter as ctk
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional, List
from models import RecordingState, CameraSettings, RaceModel  # Add RaceModel here
import cv2
from PIL import Image, ImageTk
import threading
import socket

class RecordingState(Enum):
    STOPPED = "stopped"
    RECORDING = "recording"

@dataclass
class PTZPosition:
    pan: float
    tilt: float
    zoom: float

@dataclass
class CameraSettings:
    resolution: str = "1920x1080"
    fps: int = 60
    debug_overlay: bool = False

class DefaultPositionPanel(ctk.CTkFrame):
    def __init__(self, parent, ptz_controls, **kwargs):
        super().__init__(parent, **kwargs)
        self.ptz_controls = ptz_controls
        self.default_position: Optional[PTZPosition] = None
        self.setup_ui()
    
    def setup_ui(self):
        # Header with status indicator
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=5, pady=5)
        
        self.status_indicator = ctk.CTkLabel(
            header,
            text="●",
            text_color="red",
            font=("Arial", 16)
        )
        self.status_indicator.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            header,
            text="Return Position",
            font=("Arial", 14, "bold")
        ).pack(side="left", padx=5)
        
        # Position display
        self.position_frame = ctk.CTkFrame(self)
        self.position_frame.pack(fill="x", padx=5, pady=5)
        
        self.position_label = ctk.CTkLabel(
            self.position_frame,
            text="Not Set",
            font=("Arial", 12)
        )
        self.position_label.pack(side="left", padx=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.set_button = ctk.CTkButton(
            button_frame,
            text="Set Current as Return",
            command=self.set_default_position,
            width=150
        )
        self.set_button.pack(side="left", padx=5)
        
        self.goto_button = ctk.CTkButton(
            button_frame,
            text="Go to Return",
            command=self.goto_default_position,
            width=150,
            state="disabled"
        )
        self.goto_button.pack(side="left", padx=5)
    
    def set_default_position(self):
        self.default_position = PTZPosition(
            pan=self.ptz_controls.pan_slider.get(),
            tilt=self.ptz_controls.tilt_slider.get(),
            zoom=self.ptz_controls.zoom_slider.get()
        )
        self.update_display()
        self.status_indicator.configure(text_color="green")
        self.goto_button.configure(state="normal")
    
    def goto_default_position(self):
        if self.default_position:
            self.ptz_controls.pan_slider.set(self.default_position.pan)
            self.ptz_controls.tilt_slider.set(self.default_position.tilt)
            self.ptz_controls.zoom_slider.set(self.default_position.zoom)
            self.ptz_controls.update_value('pan', self.default_position.pan)
            self.ptz_controls.update_value('tilt', self.default_position.tilt)
            self.ptz_controls.update_value('zoom', self.default_position.zoom)
    
    def update_display(self):
        if self.default_position:
            self.position_label.configure(
                text=f"Pan: {self.default_position.pan:.1f} | "
                     f"Tilt: {self.default_position.tilt:.1f} | "
                     f"Zoom: {self.default_position.zoom:.1f}"
            )

class TrackingStatusBar(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_ui()
        self.tracking = False
        self.vehicle_lost = False
        self.current_car = None
    
    def setup_ui(self):
        # Left side frame for status indicator and label
        status_frame = ctk.CTkFrame(self)
        status_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Status indicator (dot)
        self.status_indicator = ctk.CTkLabel(
            status_frame,
            text="●",
            text_color="red",
            font=("Arial", 16)
        )
        self.status_indicator.pack(side="left", padx=5)
        
        # Status label with wrapping enabled
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Not Tracking",
            font=("Arial", 12),
            wraplength=200,  # Adjust this value based on your needs
            justify="left"
        )
        self.status_label.pack(side="left", padx=5, fill="x", expand=True)
        
        # Start/Stop button
        self.start_button = ctk.CTkButton(
            self,
            text="Start Tracking",
            command=self.toggle_tracking,
            width=120
        )
        self.start_button.pack(side="right", padx=5)
    
    def toggle_tracking(self):
        self.tracking = not self.tracking
        self.vehicle_lost = True  # Reset lost state when toggling tracking
        self.update_status()
    
    def update_status(self):
        if not self.tracking:
            # Not tracking state
            self.status_indicator.configure(text_color="red")
            self.status_label.configure(text="Not Tracking")
            self.start_button.configure(text="Start Tracking")
        elif self.vehicle_lost:
            # Vehicle lost state
            self.status_indicator.configure(text_color="yellow")
            self.status_label.configure(
                text=f"Vehicle Lost - ({self.current_car})"
            )
        else:
            # Active tracking state
            self.status_indicator.configure(text_color="green")
            self.status_label.configure(text=f"Tracking {self.current_car}")
            self.start_button.configure(text="Stop Tracking")
    
    def set_car(self, car_name: str):
        self.current_car = car_name
        self.update_status()
    
    def set_vehicle_lost(self, lost: bool):
        """Call this method when the tracking system loses/regains the vehicle"""
        if self.tracking:  # Only update lost state if we're actively tracking
            self.vehicle_lost = lost
            self.update_status()
        

class RecordingControls(ctk.CTkFrame):
    def __init__(self, parent, log_console=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.log_console = log_console
        self.recording_state = RecordingState.STOPPED
        self.record_start_time = None
        self.setup_ui()
        
    def setup_ui(self):
        # Recording controls with enhanced layout
        self.record_row = ctk.CTkFrame(self)
        self.record_row.pack(fill="x", padx=5)
        
        # Recording indicator (red dot)
        self.record_indicator = ctk.CTkLabel(
            self.record_row,
            text="⚫",
            text_color="gray",
            font=("Arial", 20)
        )
        self.record_indicator.pack(side="left", padx=5)
        
        # Record button
        self.record_button = ctk.CTkButton(
            self.record_row,
            text="Start Recording",
            command=self.toggle_recording,
            width=120
        )
        self.record_button.pack(side="left", padx=5)
        
        # Timer
        self.time_label = ctk.CTkLabel(self.record_row, text="00:00:00")
        self.time_label.pack(side="left", padx=5)
        
        # Cloud upload button
        self.upload_button = ctk.CTkButton(
            self.record_row,
            text="Upload to Cloud",
            command=self.show_upload_dialog,
            width=120,
            state="disabled"
        )
        self.upload_button.pack(side="right", padx=5)

    def toggle_recording(self):
        if self.recording_state == RecordingState.STOPPED:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording_state = RecordingState.RECORDING
        self.record_button.configure(text="Stop Recording")
        self.record_indicator.configure(text_color="red")
        self.upload_button.configure(state="disabled")
        self.record_start_time = time.time()
        self.update_timer()
        if self.log_console:
            self.log_console.log("Recording started")
            
    def stop_recording(self):
        self.recording_state = RecordingState.STOPPED
        self.record_button.configure(text="Start Recording")
        self.record_indicator.configure(text_color="gray")
        self.upload_button.configure(state="normal")
        self.record_start_time = None
        self.time_label.configure(text="00:00:00")
        if self.log_console:
            self.log_console.log("Recording Ended")


    def update_timer(self):
        if self.recording_state == RecordingState.RECORDING and self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.time_label.configure(text=time_str)
            self.after(1000, self.update_timer)

    def show_upload_dialog(self):
        UploadDialog(self)

class UploadDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Upload Status")
        self.geometry("300x150")
        
        # Make dialog unclosable while operation is in progress
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Wait for the dialog to be ready
        self.update_idletasks()
        
        # Calculate position x and y coordinates
        x = parent.winfo_toplevel().winfo_x() + (parent.winfo_toplevel().winfo_width()/2 - self.winfo_width()/2)
        y = parent.winfo_toplevel().winfo_y() + (parent.winfo_toplevel().winfo_height()/2 - self.winfo_height()/2)
        
        # Set the dialog position
        self.geometry(f"+{int(x)}+{int(y)}")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.setup_ui()        
    def setup_ui(self):
        self.progress_label = ctk.CTkLabel(self, text="Uploading recording...")
        self.progress_label.pack(pady=20)
        
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        self.simulate_upload()
    
    def simulate_upload(self, value=0):
        if value <= 1:
            self.progress_bar.set(value)
            self.after(100, self.simulate_upload, value + 0.1)
        else:
            self.progress_label.configure(text="Upload complete!")
            self.after(1000, self.destroy)

class PTZControls(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_ui()
    
    def setup_ui(self):
        ctk.CTkLabel(self, text="PTZ Controls").pack()
        
        # Zoom control
        zoom_frame = ctk.CTkFrame(self)
        zoom_frame.pack(fill="x", pady=5)
        
        # Zoom label and value
        zoom_label_frame = ctk.CTkFrame(zoom_frame)
        zoom_label_frame.pack(side="left", padx=5)
        ctk.CTkLabel(zoom_label_frame, text="Zoom").pack(side="left", padx=5)
        self.zoom_value = ctk.CTkLabel(zoom_label_frame, text="50")
        self.zoom_value.pack(side="left", padx=5)
        
        # Zoom buttons and slider frame
        zoom_control_frame = ctk.CTkFrame(zoom_frame)
        zoom_control_frame.pack(side="right", expand=True, padx=5)
        
        self.zoom_left = ctk.CTkButton(
            zoom_control_frame, 
            text="◀",
            width=30,
            command=lambda: self.adjust_value('zoom', -1)
        )
        self.zoom_left.pack(side="left", padx=2)
        
        self.zoom_slider = ctk.CTkSlider(
            zoom_control_frame,
            from_=0,
            to=100,
            command=lambda x: self.update_value('zoom', x)
        )
        self.zoom_slider.pack(side="left", expand=True, padx=5)
        self.zoom_slider.set(50)
        
        self.zoom_right = ctk.CTkButton(
            zoom_control_frame,
            text="▶",
            width=30,
            command=lambda: self.adjust_value('zoom', 1)
        )
        self.zoom_right.pack(side="left", padx=2)
        
        # Pan control
        pan_frame = ctk.CTkFrame(self)
        pan_frame.pack(fill="x", pady=5)
        
        # Pan label and value
        pan_label_frame = ctk.CTkFrame(pan_frame)
        pan_label_frame.pack(side="left", padx=5)
        ctk.CTkLabel(pan_label_frame, text="Pan").pack(side="left", padx=5)
        self.pan_value = ctk.CTkLabel(pan_label_frame, text="50")
        self.pan_value.pack(side="left", padx=5)
        
        # Pan buttons and slider frame
        pan_control_frame = ctk.CTkFrame(pan_frame)
        pan_control_frame.pack(side="right", expand=True, padx=5)
        
        self.pan_left = ctk.CTkButton(
            pan_control_frame,
            text="◀",
            width=30,
            command=lambda: self.adjust_value('pan', -1)
        )
        self.pan_left.pack(side="left", padx=2)
        
        self.pan_slider = ctk.CTkSlider(
            pan_control_frame,
            from_=0,
            to=100,
            command=lambda x: self.update_value('pan', x)
        )
        self.pan_slider.pack(side="left", expand=True, padx=5)
        self.pan_slider.set(50)
        
        self.pan_right = ctk.CTkButton(
            pan_control_frame,
            text="▶",
            width=30,
            command=lambda: self.adjust_value('pan', 1)
        )
        self.pan_right.pack(side="left", padx=2)
        
        # Tilt control
        tilt_frame = ctk.CTkFrame(self)
        tilt_frame.pack(fill="x", pady=5)
        
        # Tilt label and value
        tilt_label_frame = ctk.CTkFrame(tilt_frame)
        tilt_label_frame.pack(side="left", padx=5)
        ctk.CTkLabel(tilt_label_frame, text="Tilt").pack(side="left", padx=5)
        self.tilt_value = ctk.CTkLabel(tilt_label_frame, text="50")
        self.tilt_value.pack(side="left", padx=5)
        
        # Tilt buttons and slider frame
        tilt_control_frame = ctk.CTkFrame(tilt_frame)
        tilt_control_frame.pack(side="right", expand=True, padx=5)
        
        self.tilt_left = ctk.CTkButton(
            tilt_control_frame,
            text="◀",
            width=30,
            command=lambda: self.adjust_value('tilt', -1)
        )
        self.tilt_left.pack(side="left", padx=2)
        
        self.tilt_slider = ctk.CTkSlider(
            tilt_control_frame,
            from_=0,
            to=100,
            command=lambda x: self.update_value('tilt', x)
        )
        self.tilt_slider.pack(side="left", expand=True, padx=5)
        self.tilt_slider.set(50)
        
        self.tilt_right = ctk.CTkButton(
            tilt_control_frame,
            text="▶",
            width=30,
            command=lambda: self.adjust_value('tilt', 1)
        )
        self.tilt_right.pack(side="left", padx=2)
    
    def adjust_value(self, control, direction):
        """Adjust the value by 1 in the specified direction"""
        if control == 'zoom':
            current = self.zoom_slider.get()
            new_value = min(100, max(0, current + direction))
            self.zoom_slider.set(new_value)
            self.update_value('zoom', new_value)
        elif control == 'pan':
            current = self.pan_slider.get()
            new_value = min(100, max(0, current + direction))
            self.pan_slider.set(new_value)
            self.update_value('pan', new_value)
        elif control == 'tilt':
            current = self.tilt_slider.get()
            new_value = min(100, max(0, current + direction))
            self.tilt_slider.set(new_value)
            self.update_value('tilt', new_value)
    
    def update_value(self, control, value):
        """Update the display value for the given control"""
        if control == 'zoom':
            self.zoom_value.configure(text=f"{int(value)}")
        elif control == 'pan':
            self.pan_value.configure(text=f"{int(value)}")
        elif control == 'tilt':
            self.tilt_value.configure(text=f"{int(value)}")

class CameraSettingsPanel(ctk.CTkFrame):
    def __init__(self, parent, settings: CameraSettings, **kwargs):
        super().__init__(parent, **kwargs)
        self.settings = settings
        self.setup_ui()
    
    def setup_ui(self):
        ctk.CTkLabel(self, text="Camera Settings").pack()
        
        # Resolution dropdown
        res_frame = ctk.CTkFrame(self)
        res_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(res_frame, text="Resolution").pack(side="left", padx=5)
        self.resolution_menu = ctk.CTkOptionMenu(
            res_frame,
            values=["1920x1080", "1280x720", "854x480"],
            command=self.on_resolution_change
        )
        self.resolution_menu.pack(side="right", padx=5)
        
        # FPS dropdown
        fps_frame = ctk.CTkFrame(self)
        fps_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(fps_frame, text="FPS").pack(side="left", padx=5)
        self.fps_menu = ctk.CTkOptionMenu(
            fps_frame,
            values=["60", "30", "24"],
            command=self.on_fps_change
        )
        self.fps_menu.pack(side="right", padx=5)
    
    def on_resolution_change(self, value):
        self.settings.resolution = value
    
    def on_fps_change(self, value):
        self.settings.fps = int(value)

class VideoFeed(ctk.CTkFrame):
    def __init__(self, parent, video_source=None, log_console=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(fg_color="black")
        self.log_console = log_console

        # Video source setup
        self.video_source = video_source
        self.cap = None
        self.is_running = False
        
        # Single canvas for both video and overlay
        self.canvas = ctk.CTkCanvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")
        
        # Drawing setup
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        
        # Start video if source provided
        if video_source is not None:
            self.start_video()

    def start_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_source)
            
        if not self.cap.isOpened():
            print("Error: Could not open video source")
            return
            
        self.is_running = True
        self.update_frame()
    
    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # If end of video, loop back to start
                if self.video_source and not isinstance(self.video_source, int):
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 0 and canvas_height > 0:
                    frame = cv2.resize(frame, (canvas_width, canvas_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas - use tag for video frame
                self.canvas.delete("video")  # Delete old frame
                self.canvas.create_image(0, 0, image=photo, anchor="nw", tags="video")
                self.canvas.photo = photo  # Keep a reference
                
                # Make sure video stays behind drawings
                self.canvas.tag_lower("video")
            
            # Schedule next update
            self.after(30, self.update_frame)
    
    def start_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="black",
            width=2,
            tags="overlay"  # Add tag for rectangle
        )
        # Ensure overlay stays on top
        self.canvas.tag_raise("overlay")

    def drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        # Make sure rectangle stays visible
        self.canvas.tag_raise(self.rect)

    def end_drag(self, event):
        msg = "Box: (%d, %d) to (%d, %d)" % (self.start_x, self.start_y, event.x, event.y)
        if self.log_console:
            self.log_console.log(msg)
        self.canvas.delete(self.rect)
    
    def stop_video(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.stop_video()

class ModelSyncPanel(ctk.CTkFrame):
    def __init__(self, parent, on_model_download=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_model_download = on_model_download  # Callback for when model is downloaded
        self.available_models = []
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        # Header with current model info
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=5, pady=5)
        
        self.current_model_label = ctk.CTkLabel(
            header,
            text="No model loaded",
            font=("Arial", 12, "bold")
        )
        self.current_model_label.pack(side="left")
        
        self.sync_button = ctk.CTkButton(
            header,
            text="Check for Models",
            width=120,
            command=self.check_for_models
        )
        self.sync_button.pack(side="right", padx=5)
        
        # Model list container
        self.model_list = ctk.CTkScrollableFrame(self)
        self.model_list.pack(fill="both", expand=True, padx=5, pady=5)
    
    def check_for_models(self):
        # Simulate checking S3 for available models
        self.sync_button.configure(state="disabled", text="Checking...")
        self.after(1000, self.display_available_models)
    
    def display_available_models(self):
        # Clear existing model list
        for widget in self.model_list.winfo_children():
            widget.destroy()
        
        # Simulate retrieved models from S3
        models = [
            RaceModel(
                race_id="RACE_123",
                version="1.0.0",
                timestamp="2024-03-01 14:30",
                cars=["Ferrari #44", "McLaren #77", "Porsche #11"],
                model_size="250MB"
            ),
            RaceModel(
                race_id="RACE_124",
                version="1.0.0",
                timestamp="2024-03-02 15:45",
                cars=["Aston Martin #23", "Alpine #31"],
                model_size="200MB"
            ),
            RaceModel(
                race_id="RACE_125",
                version="1.0.0",
                timestamp="2024-03-03 15:45",
                cars=["Mazda_MX30", "LEXUS_NX_2014", "AUDI_A7_2017", "TOYOTA_Rav4_2018", "BMW_7_2022"],
                model_size="200MB"
            )
        ]
        
        # Display each model
        for model in models:
            model_frame = ctk.CTkFrame(self.model_list)
            model_frame.pack(fill="x", padx=5, pady=2)
            
            # Model info
            info_frame = ctk.CTkFrame(model_frame)
            info_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(
                info_frame,
                text=f"Race: {model.race_id}",
                font=("Arial", 12, "bold")
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                info_frame,
                text=f"Cars: {', '.join(model.cars)}"
            ).pack(anchor="w")
            
            details_frame = ctk.CTkFrame(info_frame)
            details_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(
                details_frame,
                text=f"Version: {model.version}"
            ).pack(side="left")
            
            ctk.CTkLabel(
                details_frame,
                text=f"Size: {model.model_size}"
            ).pack(side="right")
            
            # Download button
            download_btn = ctk.CTkButton(
                model_frame,
                text="Download",
                width=100,
                command=lambda m=model: self.download_model(m)
            )
            download_btn.pack(pady=5)
        
    # In ModelSyncPanel's download_model method:
    def download_model(self, model):
        download_dialog = ModelDownloadDialog(self, model)
        # After successful download:
        self.current_model = model
        self.update_current_model_display()
        # Notify parent about the downloaded model
        if self.on_model_download:
            self.on_model_download(model)

    def update_current_model_display(self):
        if self.current_model:
            self.current_model_label.configure(
                text=f"Current: {self.current_model.race_id} (v{self.current_model.version})"
            )

class ModelDownloadDialog(ctk.CTkToplevel):
    def __init__(self, parent, model):
        super().__init__(parent)
        self.title("Downloading Model")
        self.geometry("300x150")
        self.model = model
        
        # Make dialog unclosable while operation is in progress
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Wait for the dialog to be ready
        self.update_idletasks()
        
        # Calculate position x and y coordinates
        x = parent.winfo_toplevel().winfo_x() + (parent.winfo_toplevel().winfo_width()/2 - self.winfo_width()/2)
        y = parent.winfo_toplevel().winfo_y() + (parent.winfo_toplevel().winfo_height()/2 - self.winfo_height()/2)
        
        # Set the dialog position
        self.geometry(f"+{int(x)}+{int(y)}")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.setup_ui()

    def setup_ui(self):
        ctk.CTkLabel(
            self,
            text=f"Downloading model for {self.model.race_id}"
        ).pack(pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self, text="Starting download...")
        self.status_label.pack(pady=5)
        
        self.simulate_download()
    
    def simulate_download(self, progress=0):
        if progress <= 1:
            self.progress_bar.set(progress)
            self.status_label.configure(
                text=f"Downloading... ({int(progress * 100)}%)"
            )
            self.after(100, self.simulate_download, progress + 0.1)
        else:
            self.status_label.configure(text="Validating model...")
            self.after(1000, self.finish_download)
    
    def finish_download(self):
        self.status_label.configure(text="Download complete!")
        self.after(1000, self.destroy)

class ModelMetadataPanel(ctk.CTkFrame):
    def __init__(self, parent, ptz_controls=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.ptz_controls = ptz_controls
        self.current_model = None
        self.tracking_car = None
        self.setup_ui()

    def setup_ui(self):
        # Race Information Section
        race_info = ctk.CTkFrame(self)
        race_info.pack(fill="x", padx=10, pady=5)

        # Default Position Panel
        self.default_position_panel = DefaultPositionPanel(self, self.ptz_controls)
        self.default_position_panel.pack(fill="x", padx=10, pady=5)
        
        # Tracking Status Bar
        self.tracking_status = TrackingStatusBar(self)
        self.tracking_status.pack(fill="x", padx=10, pady=5)
                
        title = ctk.CTkLabel(
            race_info, 
            text="Race Information", 
            font=("Arial", 14, "bold")
        )
        title.pack(anchor="w", pady=5)

        self.race_info_frame = ctk.CTkFrame(race_info)
        self.race_info_frame.pack(fill="x", pady=5)
        
        # Initial empty state
        self.no_model_label = ctk.CTkLabel(
            self.race_info_frame,
            text="No model loaded. \nPlease download a model first.",
            text_color="gray"
        )
        self.no_model_label.pack(pady=10)

        # Car List Section
        car_section = ctk.CTkFrame(self)
        car_section.pack(fill="both", expand=True, padx=10, pady=5)
        
        car_title = ctk.CTkLabel(
            car_section, 
            text="Available Cars", 
            font=("Arial", 14, "bold")
        )
        car_title.pack(anchor="w", pady=5)

        self.car_list = ctk.CTkScrollableFrame(car_section)
        self.car_list.pack(fill="both", expand=True, pady=5)

    def set_tracking_car(self, car_name: str):
        self.tracking_car = car_name
        self.tracking_status.set_car(car_name)
        self.update_car_indicators()

    def update_car_indicators(self):
        for widget in self.car_list.winfo_children():
            car_name = widget.winfo_children()[1].winfo_children()[0].cget("text")
            status_indicator = widget.winfo_children()[0].winfo_children()[0]
            if car_name == self.tracking_car:
                status_indicator.configure(text_color="green")
            else:
                status_indicator.configure(text_color="red")

    def update_metadata(self, model: RaceModel):
        self.current_model = model
        
        # Clear no model label
        self.no_model_label.pack_forget()
        
        # Update race information
        for widget in self.race_info_frame.winfo_children():
            widget.destroy()

        # Create grid for race info
        info_items = [
            ("Race ID:", model.race_id),
            ("Version:", model.version),
            ("Timestamp:", model.timestamp),
            ("Model Size:", model.model_size),
            ("Status:", model.status)
        ]

        for i, (label, value) in enumerate(info_items):
            ctk.CTkLabel(
                self.race_info_frame,
                text=label,
                font=("Arial", 12, "bold")
            ).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            ctk.CTkLabel(
                self.race_info_frame,
                text=value,
                font=("Arial", 12)
            ).grid(row=i, column=1, sticky="w", padx=5, pady=2)

        # Update car list
        for widget in self.car_list.winfo_children():
            widget.destroy()

        for car in model.cars:
            car_frame = ctk.CTkFrame(self.car_list)
            car_frame.pack(fill="x", padx=5, pady=2)
            
            # Status indicator frame
            status_frame = ctk.CTkFrame(car_frame)
            status_frame.pack(side="left", padx=5, pady=5)
            
            status_indicator = ctk.CTkLabel(
                status_frame,
                text="●",
                text_color="red",
                font=("Arial", 16)
            )
            status_indicator.pack(padx=5)
            
            # Car info
            info_frame = ctk.CTkFrame(car_frame)
            info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            
            ctk.CTkLabel(
                info_frame,
                text=car,
                font=("Arial", 12, "bold")
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                info_frame,
                text="Available for tracking",
                text_color="gray",
                font=("Arial", 10)
            ).pack(anchor="w")
            
            # Track button
            track_btn = ctk.CTkButton(
                car_frame,
                text="Track",
                width=60,
                height=24,
                font=("Arial", 11),
                command=lambda c=car: self.set_tracking_car(c)
            )
            track_btn.pack(side="right", padx=5, pady=5)

class LogConsole(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.is_expanded = False
        self.setup_ui()
        
    def setup_ui(self):
        # Header bar with toggle button
        self.header = ctk.CTkFrame(self)
        self.header.pack(fill="x", pady=(0, 1))
        
        # Toggle button
        self.toggle_btn = ctk.CTkButton(
            self.header,
            text="▼ Show Logs",  # Will change to ▼ when expanded
            command=self.toggle_console,
            height=24,
            width=100
        )
        self.toggle_btn.pack(side="left", padx=5, pady=2)
        
        # Log text area
        self.log_area = ctk.CTkTextbox(
            self,
            height=150,  # Default height when expanded
            wrap="word",
            state="disabled"  # Make it read-only
        )
        
        # Initialize logger
        self.setup_logger()
        
    def setup_logger(self):
        # Create a custom logger
        self.logger = logging.getLogger('GUI_Logger')
        self.logger.setLevel(logging.INFO)
        
        # Create a custom handler that writes to our text widget
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.after(0, self.text_widget.insert_log, msg + '\n')
                
        # Add custom method to insert text while handling state
        def insert_log(text_widget, message):
            text_widget.configure(state="normal")
            text_widget.insert("end", message)
            text_widget.configure(state="disabled")
            text_widget.see("end")  # Auto-scroll to bottom
            
        self.log_area.insert_log = lambda msg: insert_log(self.log_area, msg)
        
        # Add the handler to the logger
        handler = TextHandler(self.log_area)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def toggle_console(self):
        if self.is_expanded:
            self.log_area.pack_forget()
            self.toggle_btn.configure(text="▼ Show Logs")
        else:
            self.log_area.pack(fill="both", expand=True, padx=5, pady=(0, 5))
            self.toggle_btn.configure(text="▲ Hide Logs")
        self.is_expanded = not self.is_expanded
        
    def log(self, message, level="info"):
        """Add a message to the log"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)

class InternetStatusIndicator(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.stop_thread = False
        self.setup_ui()
        self.start_monitoring()
    
    def setup_ui(self):
        # Simple row with indicator and status
        status_row = ctk.CTkFrame(self)
        status_row.pack(fill="x", padx=5, pady=5)
        
        # Internet status indicator dot
        self.status_indicator = ctk.CTkLabel(
            status_row,
            text="●",
            text_color="red",
            font=("Arial", 16)
        )
        self.status_indicator.pack(side="left", padx=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            status_row,
            text="Internet Status",
            font=("Arial", 12)
        )
        self.status_label.pack(side="left", padx=5)
    
    def check_internet(self):
        try:
            # Try to connect to Google's DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=1.0)
            return True
        except (socket.timeout, socket.error):
            return False
    
    def monitor_connection(self):
        while not self.stop_thread:
            connected = self.check_internet()
            self.status_indicator.configure(
                text_color="green" if connected else "red"
            )
            time.sleep(2)  # Check every 2 seconds
    
    def start_monitoring(self):
        self.stop_thread = False
        self.monitor_thread = threading.Thread(
            target=self.monitor_connection, 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.stop_thread = True