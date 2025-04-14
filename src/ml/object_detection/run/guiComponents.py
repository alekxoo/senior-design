import customtkinter as ctk
import boto3, os  # import libraries to run retrieve and upload functions to S3 bucket
from tkinter import messagebox
import threading
from dotenv import load_dotenv

load_dotenv()


class ModelInfoComponents:
    """Class containing methods for model browsing and race information UI components"""

    def __init__(self, parent, on_model_download_success=None):
        """Initialize with parent container"""
        self.parent = parent
        self.model_list = None
        self.race_info = None

        self.on_model_download_success = on_model_download_success

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME")
        )

    def create_model_section(self, container):
        """Creates the model browser section with username input"""
        # Model section frame
        model_section = ctk.CTkFrame(container)
        model_section.pack(fill="x", pady=10, padx=10)

        # Header for model section
        model_header = ctk.CTkLabel(model_section, text="Search Models by User", font=("Arial", 14, "bold"))
        model_header.pack(anchor="w", padx=10, pady=5)

        # Username input
        username_frame = ctk.CTkFrame(model_section)
        username_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(username_frame, text="Username:").pack(side="left", padx=5)
        self.username_entry = ctk.CTkEntry(username_frame)
        self.username_entry.pack(side="left", fill="x", expand=True, padx=5)

        # Submit button
        submit_btn = ctk.CTkButton(model_section, text="Submit",
                                   command=self.handle_submit)
        submit_btn.pack(fill="x", padx=10, pady=5)

        # Scrollable frame for model list
        model_list_container = ctk.CTkFrame(model_section)
        model_list_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Create scrollable frame for models
        self.model_list = self.ModelListFrame(model_list_container, self)
        self.model_list.pack(fill="both", expand=True)

    def handle_submit(self):
        """Handles the submit button press - finds valid races and populates UI"""
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showerror("Input Error", "Please enter a Username.")
            return

        try:
            bucket_name = os.getenv("S3_RACES_BUCKET_NAME")
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{username}/")
            if 'Contents' not in response:
                messagebox.showinfo("No Races", "No race folders found for this user.")
                return

            races = {}

            for obj in response['Contents']:
                key = obj['Key']
                parts = key.split('/')
                if len(parts) >= 3:
                    race_name = parts[1]
                    subfolder = parts[2]
                    if subfolder in ['config', 'weights']:
                        if race_name not in races:
                            races[race_name] = set()
                        races[race_name].add(subfolder)

            valid_races = [race for race, folders in races.items() if {'config', 'weights'} <= folders]

            if not valid_races:
                messagebox.showinfo("No Valid Races", "No races with both config and weights found.")
            else:
                self.model_list.populate_races(username, valid_races)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch races: {str(e)}")

    def download_files(self, username, racename, download_path):
        """Download config and weights files from S3"""
        s3 = self.s3
        config_downloaded = False
        weights_downloaded = False

        bucket_name = os.getenv("S3_RACES_BUCKET_NAME")
        if not bucket_name:
            messagebox.showerror("Error", "S3_RACES_BUCKET_NAME environment variable is not set.")
            return

        config_key = f"{username}/{racename}/config/"
        weights_key = f"{username}/{racename}/weights/"

        config_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=config_key)
        weights_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=weights_key)

        if 'Contents' in config_objects:
            for obj in config_objects['Contents']:
                if obj['Key'].endswith('.yaml'):
                    config_filename = os.path.basename(obj['Key'])
                    config_filepath = os.path.join(download_path, config_filename)
                    s3.download_file(bucket_name, obj['Key'], config_filepath)
                    config_downloaded = True
                    break
        else:
            messagebox.showerror("Error:", "No config file found in S3.")

        if 'Contents' in weights_objects:
            for obj in weights_objects['Contents']:
                if obj['Key'].endswith('best.pt'):
                    weights_filepath = os.path.join(download_path, 'best.pt')
                    s3.download_file(bucket_name, obj['Key'], weights_filepath)
                    weights_downloaded = True
                    break
        else:
            messagebox.showerror("Error:", "No weights file found in S3.")

        if config_downloaded and weights_downloaded:
            messagebox.showinfo("Success", "Both model files downloaded successfully!")
            if self.on_model_download_success:
                self.on_model_download_success(
                username,
                racename,
                yaml_path=config_filepath,
                model_path=weights_filepath
            )

    def create_race_info_section(self, container):
        """Creates the race information section"""
        race_section = ctk.CTkFrame(container)
        race_section.pack(fill="x", pady=10, padx=10)

        race_header = ctk.CTkLabel(race_section, text="Race Information", font=("Arial", 14, "bold"))
        race_header.pack(anchor="w", padx=10, pady=5)

        self.race_info = self.RaceInfoFrame(race_section)
        self.race_info.pack(fill="both", expand=True, padx=5, pady=5)

        self.race_info.update_info({
            "Race_ID": "Not loaded",
            "Number of Cars": "Not loaded",
            "Version Number": "Not loaded"
        })

    class ModelListFrame(ctk.CTkScrollableFrame):
        """Custom scrollable frame for displaying race names with download buttons."""

        def __init__(self, master, outer, **kwargs):
            super().__init__(master, height=200, **kwargs)
            self.outer = outer
            self.race_buttons = []

        def populate_races(self, username, race_names):
            for widget in self.race_buttons:
                widget.destroy()
            self.race_buttons = []

            for race in race_names:
                row = ctk.CTkFrame(self)
                row.pack(fill="x", pady=2)

                race_label = ctk.CTkLabel(row, text=race, anchor="w")
                race_label.pack(side="left", fill="x", expand=True, padx=5)

                download_btn = ctk.CTkButton(row, text="Download", width=100,
                                             command=lambda r=race: self.download_for_race(username, r))
                download_btn.pack(side="right", padx=5)

                self.race_buttons.append(row)

        import threading

        def download_for_race(self, username, racename):
            current_dir = os.getcwd()
            download_path = os.path.join(current_dir, "config")
            os.makedirs(download_path, exist_ok=True)

            # Start download in a separate thread
            threading.Thread(
                target=self.outer.download_files,
                args=(username, racename, download_path),
                daemon=True  # Optional: terminates thread when main app exits
            ).start()


    class RaceInfoFrame(ctk.CTkScrollableFrame):
        """Custom scrollable frame for displaying race information."""

        def __init__(self, master, **kwargs):
            super().__init__(master, height=150, **kwargs)
            self.info_widgets = {}

        def update_info(self, info_dict):
            for label in self.info_widgets.values():
                label.destroy()
            self.info_widgets = {}

            for key, value in info_dict.items():
                label_frame = ctk.CTkFrame(self)
                label_frame.pack(fill="x", pady=2)

                key_label = ctk.CTkLabel(label_frame, text=f"{key}:", anchor="w", width=150)
                key_label.pack(side="left", padx=5)

                value_label = ctk.CTkLabel(label_frame, text=str(value), anchor="w")
                value_label.pack(side="left", fill="x", expand=True, padx=5)

                self.info_widgets[key] = label_frame
