import customtkinter as ctk
from gui_components import (
    VideoFeed,
    RecordingControls,
    PTZControls,
    CameraSettingsPanel,
    ModelSyncPanel,
    ModelMetadataPanel,
    LogConsole, 
    InternetStatusIndicator
)
from models import CameraSettings

class PTZControlGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("PTZ Camera Control")
        self.root.geometry("1600x800")
        ctk.set_appearance_mode("dark")
        
        self.camera_settings = CameraSettings()
        self.current_mode = ctk.StringVar(value="manual")

        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel - Video Feed
        left_panel = ctk.CTkFrame(main_container, width=550)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        self.video_feed = VideoFeed(left_panel, video_source=0, log_console=None)
        self.video_feed.pack(fill="both", expand=True, padx=0, pady=0)

        self.recording_controls = RecordingControls(left_panel, log_console=None)
        self.recording_controls.pack(fill="x", pady=(0, 10))

        # Right panel
        self.right_panel = ctk.CTkFrame(main_container, width=600)
        self.right_panel.pack(side="right", fill="both", padx=(10, 0), expand=True)
        self.right_panel.pack_propagate(False)

        # Mode selection buttons
        mode_frame = ctk.CTkFrame(self.right_panel)
        mode_frame.pack(fill="x", pady=(0, 10))

        manual_btn = ctk.CTkButton(mode_frame, text="Properties", width=160, height=32,
                                   command=lambda: self.switch_mode("manual"))
        manual_btn.pack(side="left", expand=True, padx=5)

        auto_btn = ctk.CTkButton(mode_frame, text="Autonomous", width=160, height=32,
                                 command=lambda: self.switch_mode("autonomous"))
        auto_btn.pack(side="right", expand=True, padx=5)

        # Dynamic content area for mode switching
        self.mode_content_frame = ctk.CTkFrame(self.right_panel)
        self.mode_content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Initialize components
        self.ptz_controls = PTZControls(self.mode_content_frame)
        self.camera_settings_panel = CameraSettingsPanel(self.mode_content_frame, settings=self.camera_settings)
        
        # Pass self.on_model_downloaded as a callback
        self.model_sync_panel = ModelSyncPanel(self.mode_content_frame, on_model_download=self.on_model_downloaded)
        self.metadata_panel = ModelMetadataPanel(self.mode_content_frame)

        # Start in manual mode
        self.switch_mode("manual")

    def switch_mode(self, mode):
        # Clear current mode content
        for widget in self.mode_content_frame.winfo_children():
            widget.pack_forget()

        # Display the correct widgets based on mode
        if mode == "manual":
            self.ptz_controls.pack(fill="both", expand=True, padx=10, pady=10)
            self.camera_settings_panel.pack(fill="x", padx=10, pady=10)
        else:
            self.model_sync_panel.pack(fill="x", padx=10, pady=10)

            # Ensure metadata_panel has access to ptz_controls in autonomous mode
            if not hasattr(self.metadata_panel, 'ptz_controls') or self.metadata_panel.ptz_controls is None:
                self.metadata_panel.ptz_controls = self.ptz_controls

            self.metadata_panel.pack(fill="both", expand=True, padx=10, pady=10)

    def on_model_downloaded(self, model):
        self.metadata_panel.update_metadata(model)
        self.metadata_panel.pack(fill="both", expand=True, padx=10, pady=10)  # Ensure visibility

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PTZControlGUI()
    app.run()
