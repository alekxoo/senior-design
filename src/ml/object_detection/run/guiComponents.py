import customtkinter as ctk

class ModelInfoComponents:
    """Class containing methods for model browsing and race information UI components"""
    
    def __init__(self, parent):
        """Initialize with parent container"""
        self.parent = parent
        self.model_list = None
        self.race_info = None
    
    def create_model_section(self, container):
        """Creates the model browser section"""
        # Model section frame
        model_section = ctk.CTkFrame(container)
        model_section.pack(fill="x", pady=10, padx=10)
        
        # Header for model section
        model_header = ctk.CTkLabel(model_section, text="Available Models", font=("Arial", 14, "bold"))
        model_header.pack(anchor="w", padx=10, pady=5)
        
        # Search button
        search_btn = ctk.CTkButton(model_section, text="Search Models", 
                                  command=self.search_models)
        search_btn.pack(fill="x", padx=10, pady=5)
        
        # Scrollable frame for model list
        model_list_container = ctk.CTkFrame(model_section)
        model_list_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create scrollable frame for models
        self.model_list = self.ModelListFrame(model_list_container)
        self.model_list.pack(fill="both", expand=True)

    def create_race_info_section(self, container):
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

    def search_models(self):
        """Placeholder for model search functionality"""
        # This would connect to S3 in the actual implementation
        print("Searching for models...")
        
        # For demonstration, populate with dummy data
        sample_models = [
            {"name": "Model_2023_GT3", "size": "245MB", "date": "2023-10-15"},
            {"name": "Model_2024_F1", "size": "312MB", "date": "2024-01-20"},
            {"name": "Model_2024_Rally", "size": "189MB", "date": "2024-03-05"}
        ]
        self.model_list.populate_models(sample_models)
    
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