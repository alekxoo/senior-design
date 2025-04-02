import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from gui_controls import CameraPOV, GUIControls

class Visualization:
    def __init__(self, figsize=(16, 10)):
        """
        Initialize visualization manager
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        # Create figure with two subplots and control area
        self.fig = plt.figure(figsize=figsize)
        
        # Create grid for plots and controls
        gs = self.fig.add_gridspec(3, 2)
        
        # Setup main plot areas
        self.ax = self.fig.add_subplot(gs[0:2, 0])  # Main track view
        self.ax_pov = self.fig.add_subplot(gs[0:2, 1])  # Camera POV
        self.ax_controls = self.fig.add_subplot(gs[2, :])  # Controls area
        
        # Configure main view
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 500)
        self.ax.set_title("Track Simulation")
        
        # UI elements
        self.frame_counter = None
        self.status_text = None
        
        # Collection of bounding box patches
        self.bbox_patches = {}
        
        # Visibility settings
        self.show_bbox = True
        self.show_car = True
        self.show_camera = True
        self.show_tracking = True
        
        # Camera POV visualization
        self.camera_pov = None
        
        # GUI controls
        self.gui_controls = None
    
    def setup_camera_pov(self, camera_params):
        """
        Set up the camera point-of-view visualization
        
        Parameters:
        -----------
        camera_params : dict
            Dictionary of camera parameters
        """
        # Create default camera parameters if not provided
        if not isinstance(camera_params, dict):
            camera_params = {
                'position': (650, 100),
                'rotation': 135,
                'view_angle': 60,
                'focal_length': 35,
                'sensor_width': 400,
                'sensor_height': 300
            }
            
        # Initialize camera POV visualization
        self.camera_pov = CameraPOV(self.ax_pov, camera_params)
        
        # Return visualization elements for animation
        return self.camera_pov.get_visualization_elements()
    
    def setup_gui_controls(self, simulation):
        """
        Set up GUI controls for simulation
        
        Parameters:
        -----------
        simulation : Simulation
            The simulation object to control
        """
        self.gui_controls = GUIControls(self.ax_controls, simulation)
        
    def update_camera_pov(self, cars, camera):
        """
        Update the camera point-of-view visualization for all cars
        
        Parameters:
        -----------
        cars : list
            List of Car objects to visualize
        camera : Camera
            The camera object
            
        Returns:
        --------
        elements : list
            List of updated matplotlib artists
        """
        if self.camera_pov is None:
            return []
        
        # Process all cars
        for car in cars:
            # Check if car is in view
            in_camera_view = car.in_camera_view
            
            # Calculate distance if car is in view
            distance = None
            if in_camera_view:
                distance = np.linalg.norm(car.position - camera.position)
            
            # Update camera POV visualization for this car
            bbox_data = self.camera_pov.project_car(car, in_camera_view, distance)
        
        # Return updated visualization elements
        return self.camera_pov.get_visualization_elements()
    
    def setup_ui_elements(self):
        """Setup UI elements like frame counter and status texts"""
        # Add frame counter and status
        self.frame_counter = self.ax.text(50, 450, "Frame: 0", fontsize=10)
        self.status_text = self.ax.text(50, 430, "Tracking: No", fontsize=10)
        
        # Create custom legend
        car_patch = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Car')
        camera_patch = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Fixed Camera')
        bbox_patch = mlines.Line2D([], [], color='green', marker='s', linestyle='--',
                          markersize=10, fillstyle='none', label='Detection Bounding Box')
        
        self.ax.legend(handles=[car_patch, camera_patch, bbox_patch], loc='upper right')
    
    def create_bbox_patch(self, car_id):
        """
        Create a bounding box patch for a car
        
        Parameters:
        -----------
        car_id : int
            ID of the car the bounding box belongs to
        """
        bbox_patch = patches.Rectangle(
            (0, 0),  # Will be updated later
            0, 0,    # Will be updated later
            fill=False,
            ec='green',
            linewidth=2,
            linestyle='--',
            visible=False,
            label=f'Detection Box {car_id}'
        )
        self.ax.add_patch(bbox_patch)
        self.bbox_patches[car_id] = bbox_patch
        return bbox_patch
    
    def update_bbox_visualization(self, cars):
        """
        Update bounding box visualizations for all cars
        
        Parameters:
        -----------
        cars : list
            List of Car objects
        """
        updated_patches = []
        
        for car in cars:
            # Get the car's bounding box
            bbox = car.bounding_box
            
            # Create bbox patch if it doesn't exist
            if car.id not in self.bbox_patches:
                self.create_bbox_patch(car.id)
            
            # Get the bbox patch
            bbox_patch = self.bbox_patches[car.id]
            
            # Update the bbox patch
            if bbox is not None and self.show_bbox:
                bbox_patch.set_xy((bbox[0], bbox[1]))
                bbox_patch.set_width(bbox[2] - bbox[0])
                bbox_patch.set_height(bbox[3] - bbox[1])
                bbox_patch.set_visible(True)
                updated_patches.append(bbox_patch)
            else:
                bbox_patch.set_visible(False)
                updated_patches.append(bbox_patch)
        
        return updated_patches
    
    def update_ui(self, frame_num, camera):
        """
        Update UI elements
        
        Parameters:
        -----------
        frame_num : int
            Current frame number
        camera : Camera
            Camera object to get tracking status
        """
        # Update frame counter
        if self.frame_counter:
            self.frame_counter.set_text(f"Frame: {frame_num}")
        
        # Update tracking status
        if self.status_text:
            status = "Tracking: Yes" if camera.tracking else "Tracking: No"
            self.status_text.set_text(status)
        
        # If recording, record the current frame
        if self.gui_controls and hasattr(self.gui_controls, 'recording') and self.gui_controls.recording:
            self.gui_controls.record_frame(frame_num)
        
        return [self.frame_counter, self.status_text]
    
    def toggle_bbox_visibility(self, visible=None):
        """Toggle bounding box visibility"""
        if visible is not None:
            self.show_bbox = visible
        else:
            self.show_bbox = not self.show_bbox
        return self.show_bbox
    
    def toggle_car_visibility(self, visible=None):
        """Toggle car visibility"""
        if visible is not None:
            self.show_car = visible
        else:
            self.show_car = not self.show_car
        return self.show_car
    
    def toggle_camera_visibility(self, visible=None):
        """Toggle camera visualization visibility"""
        if visible is not None:
            self.show_camera = visible
        else:
            self.show_camera = not self.show_camera
        return self.show_camera
    
    def toggle_tracking_visibility(self, visible=None):
        """Toggle tracking history visibility"""
        if visible is not None:
            self.show_tracking = visible
        else:
            self.show_tracking = not self.show_tracking
        return self.show_tracking