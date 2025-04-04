import customtkinter as ctk
import boto3, os #import libraries to run retrieve and upload functions to S3 bucket
from tkinter import messagebox
import os
from dotenv import load_dotenv

load_dotenv()


"""
TODO: when retrieving model, instead of pulling all models, just include text box and use string input to search for model and retrieve pt and yaml file
should just be a race code


TODO: include a parsing function to parse the retrieved config yaml file
"""



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
        """Creates the model browser section with username and racename input"""
        # Model section frame
        model_section = ctk.CTkFrame(container)
        model_section.pack(fill="x", pady=10, padx=10)
        
        # Header for model section
        model_header = ctk.CTkLabel(model_section, text="Search Models by User & Race", font=("Arial", 14, "bold"))
        model_header.pack(anchor="w", padx=10, pady=5)
        
        # Username input
        username_frame = ctk.CTkFrame(model_section)
        username_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(username_frame, text="Username:").pack(side="left", padx=5)
        self.username_entry = ctk.CTkEntry(username_frame)
        self.username_entry.pack(side="left", fill="x", expand=True, padx=5)

        # Race name input
        racename_frame = ctk.CTkFrame(model_section)
        racename_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(racename_frame, text="Race Name:").pack(side="left", padx=5)
        self.racename_entry = ctk.CTkEntry(racename_frame)
        self.racename_entry.pack(side="left", fill="x", expand=True, padx=5)

        # Submit button
        submit_btn = ctk.CTkButton(model_section, text="Submit", 
                                command=self.handle_submit)
        submit_btn.pack(fill="x", padx=10, pady=5)

        # Scrollable frame for model list
        model_list_container = ctk.CTkFrame(model_section)
        model_list_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create scrollable frame for models
        self.model_list = self.ModelListFrame(model_list_container)
        self.model_list.pack(fill="both", expand=True)

    def handle_submit(self):
        #TODO: when submitting create handler to ensure that the old weights and yaml file are deleted before pulling new files from S3
        """Handles the submit button press"""
        username = self.username_entry.get().strip()
        racename = self.racename_entry.get().strip()
        
        if not username or not racename:
            print("Please enter both Username and Race Name.")
            return
        
        # Then download config and weights files
        current_dir = os.getcwd()
        download_path = os.path.join(current_dir, "config") #download files and store in config folder in object_detection directory
        os.makedirs(download_path, exist_ok=True)
        print(download_path)

        self.download_files(username, racename, download_path)

    def download_files(self, username, racename, download_path):

        #pass in aws client
        s3 = self.s3
        config_downloaded = False
        weights_downloaded = False

        """Download config and weights files from S3"""
        # Retrieve bucket name from environment variable
        bucket_name = os.getenv("S3_RACES_BUCKET_NAME")
        if not bucket_name:
            messagebox.showerror("Error", "S3_RACES_BUCKET_NAME environment variable is not set.")
            return
        
        config_key = f"{username}/{racename}/config/"
        weights_key = f"{username}/{racename}/weights/"
        
        # List objects in the config directory
        config_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=config_key)
        weights_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=weights_key)
        
        if 'Contents' in config_objects:
            for obj in config_objects['Contents']:
                if obj['Key'].endswith('.yaml'):
                    config_filename = os.path.basename(obj['Key'])
                    config_filepath = os.path.join(download_path, config_filename)
                    s3.download_file(bucket_name, obj['Key'], config_filepath)
                    # messagebox.showinfo("Message", f"Downloaded config file: {config_filepath}")
                    config_downloaded = True
                    break  # Only one config file expected
        else:
            messagebox.showerror("Error:","No config file found in S3.")
        
        if 'Contents' in weights_objects:
            for obj in weights_objects['Contents']:
                if obj['Key'].endswith('best.pt'):
                    weights_filepath = os.path.join(download_path, 'best.pt')
                    s3.download_file(bucket_name, obj['Key'], weights_filepath)
                    # messagebox.showinfo("Message", f"Downloaded weights file: {weights_filepath}")
                    weights_downloaded = True
                    break  # Only one weights file expected
        else:
            messagebox.showerror("Error:","No weights file found in S3.")

        if config_downloaded and weights_downloaded:
            messagebox.showinfo("Success", "Both model files downloaded successfully!")
            if self.on_model_download_success:
                self.on_model_download_success(username, racename)  

        #TODO: call parser to populate section with yaml file information

    def create_race_info_section(self, container):
        #TODO:update this section with parse info from yaml file once successfully downloaded
        """Creates the race information section"""
        # Race info section frame
        race_section = ctk.CTkFrame(container)
        race_section.pack(fill="x", pady=10, padx=10)
        
        # Header for race info
        race_header = ctk.CTkLabel(race_section, text="Race Information", font=("Arial", 14, "bold"))
        race_header.pack(anchor="w", padx=10, pady=5)
        
        # Create scrollable text widget for race information
        self.race_info = self.RaceInfoFrame(race_section)
        self.race_info.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Populate with some placeholder data
        self.race_info.update_info({
            "Race_ID": "Not loaded",
            "Number of Cars": "Not loaded",
            "Version Number": "Not loaded"
        })
    
    class ModelListFrame(ctk.CTkScrollableFrame):
        """Custom scrollable frame for displaying available models."""
        
        def __init__(self, master, **kwargs):
            super().__init__(master, height=150, **kwargs)
            self.model_rows = []
        
        def populate_models(self, models):
            """Populate the frame with model data."""
            # Clear existing widgets
            for row in self.model_rows:
                for widget in row:
                    widget.destroy()
            self.model_rows = []
            
            # Create header row
            header = ctk.CTkFrame(self)
            header.pack(fill="x", pady=(0, 5))
            
            ctk.CTkLabel(header, text="Model Name", width=150).pack(side="left", padx=5)
            ctk.CTkLabel(header, text="Size", width=80).pack(side="left", padx=5)
            ctk.CTkLabel(header, text="Date", width=100).pack(side="left", padx=5)
            
            # Add model rows
            for model in models:
                row_frame = ctk.CTkFrame(self)
                row_frame.pack(fill="x", pady=2)
                
                name_label = ctk.CTkLabel(row_frame, text=model["name"], width=150)
                name_label.pack(side="left", padx=5)
                
                size_label = ctk.CTkLabel(row_frame, text=model["size"], width=80)
                size_label.pack(side="left", padx=5)
                
                date_label = ctk.CTkLabel(row_frame, text=model["date"], width=100)
                date_label.pack(side="left", padx=5)
                
                download_btn = ctk.CTkButton(row_frame, text="Download", width=80, 
                                            command=lambda m=model["name"]: self.download_model(m))
                download_btn.pack(side="right", padx=5)
                
                self.model_rows.append([name_label, size_label, date_label, download_btn])
        
        def download_model(self, model_name):
            """Placeholder for model download functionality."""
            print(f"Downloading model: {model_name}")
            # In actual implementation, this would trigger S3 download


    class RaceInfoFrame(ctk.CTkScrollableFrame):
        """Custom scrollable frame for displaying race information."""
        
        def __init__(self, master, **kwargs):
            super().__init__(master, height=150, **kwargs)
            self.info_widgets = {}
        
        def update_info(self, info_dict):
            """Update race information display."""
            # Clear existing widgets
            for label in self.info_widgets.values():
                label.destroy()
            self.info_widgets = {}
            
            # Add new information
            row = 0
            for key, value in info_dict.items():
                label_frame = ctk.CTkFrame(self)
                label_frame.pack(fill="x", pady=2)
                
                key_label = ctk.CTkLabel(label_frame, text=f"{key}:", anchor="w", width=150)
                key_label.pack(side="left", padx=5)
                
                value_label = ctk.CTkLabel(label_frame, text=str(value), anchor="w")
                value_label.pack(side="left", fill="x", expand=True, padx=5)
                
                self.info_widgets[key] = label_frame
                row += 1