import customtkinter as ctk
from gui_components import (
    # VideoFeed,
    ImageFeed,
    RecordingControls,
    PTZControls,
    CameraSettingsPanel,
    ModelSyncPanel,
    ModelMetadataPanel,
    LogConsole, 
    InternetStatusIndicator
)
from models import CameraSettings

# harris_hill_video = "/home/alekxoo/Documents/f24_class/senior_design/vehicle_images_vault/harris_hills_car_video.MOV"
harris_hill_video="/home/alekxoo/Downloads/MVI_6206.MP4"

class PTZControlGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("PTZ Camera Control")
        self.root.geometry("1600x800")
        ctk.set_appearance_mode("dark")
        
        self.camera_settings = CameraSettings()
        self.current_mode = ctk.StringVar(value="manual")
        self.create_gui()

    def create_gui(self):
        # Create main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # logger
        self.log_console = LogConsole(self.root)
        self.log_console.pack(fill="x", padx=20, pady=(0, 20))
        
        # Left panel - Video Feed (60% of width instead of 70%)
        left_panel = ctk.CTkFrame(main_container, width=550)  # Reduced from 720
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)        

        # Video Feed
        # camera index for video feed / video file path for video 
        # self.video_feed = VideoFeed(left_panel, video_source=harris_hill_video, log_console=self.log_console)
        # self.video_feed = VideoFeed(left_panel, video_source=0, log_console=self.log_console)  # Default/built-in webcam
        # self.video_feed.pack(fill="both", expand=True)
        self.image_feed = ImageFeed(left_panel, image_path="screenshot1.png", log_console=self.log_console)
        self.image_feed.pack(fill="both", expand=True)
        
        # Recording Controls
        self.recording_controls = RecordingControls(left_panel, log_console=self.log_console)
        self.recording_controls.pack(fill="x", pady=(0, 10))
        
        # Right panel (40% of width instead of 30%)
        right_panel = ctk.CTkFrame(main_container, width=600)  # Increased from 480
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)   
             

        # Mode Selection
        mode_frame = ctk.CTkFrame(right_panel)
        mode_frame.pack(fill="x", pady=(0, 10))

        # Make mode buttons wider
        manual_btn = ctk.CTkButton(
            mode_frame,
            text="Properties",
            width=160,      # Increased from 120
            height=32,
            command=lambda: self.switch_mode("manual")
        )
        manual_btn.pack(side="left", expand=True, padx=5)

        auto_btn = ctk.CTkButton(
            mode_frame,
            text="Autonomous",
            width=160,      # Increased from 120
            height=32,
            command=lambda: self.switch_mode("autonomous")
        )
        auto_btn.pack(side="right", expand=True, padx=5)
        
        # Control panels container
        self.control_container = ctk.CTkFrame(right_panel)
        self.control_container.pack(fill="both", expand=True)

        # Internet status 
        properties_frame = ctk.CTkFrame(self.control_container)
        properties_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(properties_frame, text="Properties").pack()
        self.internet_status = InternetStatusIndicator(properties_frame)
        self.internet_status.pack(fill="x")

        # Create all panels (remove the duplicate ModelMetadataPanel)
        self.metadata_panel = ModelMetadataPanel(self.control_container)
        self.ptz_controls = PTZControls(self.control_container)
        self.camera_settings_panel = CameraSettingsPanel(
            self.control_container,
            settings=self.camera_settings
        )
        
        # Add ModelSyncPanel with callback
        self.model_sync_panel = ModelSyncPanel(
            self.control_container,
            on_model_download=self.on_model_downloaded
        )
        
        # Show default mode
        self.switch_mode("manual")

    def on_model_downloaded(self, model):
        # Update the metadata panel with the new model
        self.metadata_panel.update_metadata(model)
        
    def switch_mode(self, mode):
        self.current_mode = mode
        if mode == "manual":
            self.metadata_panel.pack_forget()
            self.model_sync_panel.pack_forget()
            self.ptz_controls.pack(fill="both", expand=True, padx=10, pady=10)
            self.camera_settings_panel.pack(fill="x", padx=10, pady=10)
        else:
            self.ptz_controls.pack_forget()
            self.camera_settings_panel.pack_forget()
            self.model_sync_panel.pack(fill="x", padx=10, pady=10)
            # Make sure the metadata_panel has access to ptz_controls even in auto mode
            if not hasattr(self.metadata_panel, 'ptz_controls') or self.metadata_panel.ptz_controls is None:
                self.metadata_panel.ptz_controls = self.ptz_controls
                self.metadata_panel.default_position_panel.ptz_controls = self.ptz_controls
            self.metadata_panel.pack(fill="both", expand=True, padx=10, pady=10)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PTZControlGUI()
    app.run()