import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from track import Track
from car import Car
from camera import Camera
from visualization import Visualization

class Simulation:
    def __init__(self):
        """Initialize the simulation"""
        # Create visualization manager with larger figure size for controls
        self.vis = Visualization(figsize=(16, 10))
        
        # Create track
        self.track = Track()
        
        # Draw track on the axes
        self.track.draw_track(self.vis.ax)
        
        # Create camera
        self.camera = Camera(position=(650, 100), view_angle=60, range=300, rotation=135)
        self.camera_elements = self.camera.create_visualization(self.vis.ax)
        
        # Create camera parameters dict for POV visualization
        self.camera_params = {
            'position': self.camera.position,
            'rotation': self.camera.rotation,
            'view_angle': self.camera.view_angle,
            'focal_length': 800,  # Default focal length
            'sensor_width': 640,
            'sensor_height': 480
        }
        
        # Setup camera POV visualization
        self.camera_pov_elements = self.vis.setup_camera_pov(self.camera_params)
        
        # Create cars collection
        self.cars = []
        self.add_car(id=0, speed=1.0, start_position=0, color='red')
        
        # Setup UI elements
        self.vis.setup_ui_elements()
        
        # Animation objects
        self.animation = None
        self.animation_running = False
        self.frame_num = 0
        
        # Default noise model settings
        self.noise_model = 'gaussian'
        self.noise_params = {'mean': 0, 'std': 2}
        
        # Set up GUI controls
        self.vis.setup_gui_controls(self)
        self.camera.sim = self

    def add_car(self, id=None, width=30, height=15, speed=1.0, start_position=0, color='red'):
        """
        Add a new car to the simulation
        
        Parameters:
        -----------
        id : int or None
            Car ID (auto-assigned if None)
        width : float
            Width of the car
        height : float
            Height of the car
        speed : float
            Movement speed along the track
        start_position : float
            Starting position on the track
        color : str
            Color of the car visualization
            
        Returns:
        --------
        car : Car
            The newly added car
        """
        # Auto-assign ID if not provided
        if id is None:
            id = len(self.cars)
        
        # Create new car
        car = Car(
            self.track, 
            id=id, 
            width=width, 
            height=height, 
            speed=speed, 
            start_position=start_position, 
            color=color
        )
        
        # Add to cars list
        self.cars.append(car)
        
        # Create visualization
        car.create_visualization(self.vis.ax)
        
        return car
    
    def remove_car(self, car_id):
        """
        Remove a car from the simulation
        
        Parameters:
        -----------
        car_id : int
            ID of the car to remove
            
        Returns:
        --------
        success : bool
            True if car was found and removed, False otherwise
        """
        for i, car in enumerate(self.cars):
            if car.id == car_id:
                # Remove car visualization
                if car.patch is not None:
                    car.patch.remove()
                
                # Remove car from list
                self.cars.pop(i)
                return True
        
        return False
    
    def animate(self, frame_num):
        """
        Animation function that updates the simulation for each frame
        
        Parameters:
        -----------
        frame_num : int
            Current frame number
            
        Returns:
        --------
        elements : list
            List of all updated artists
        """
        # Update frame counter
        self.frame_num = frame_num
        
        updated_elements = []
        
        # Calculate distance from camera to car for distance-based noise
        for car in self.cars:
            # Store distance to camera for use in noise models
            car.distance_to_camera = np.linalg.norm(car.position - self.camera.position)
        
        # Move all cars
        for car in self.cars:
            car.move()
            car.update_visualization()
            updated_elements.append(car.patch)
        
        # Update camera tracking
        self.camera.update_tracking(self.cars)
        
        # Update camera visualization
        updated_elements.extend(self.camera.update_visualization())
        
        # Update bounding box visualization
        updated_elements.extend(self.vis.update_bbox_visualization(self.cars))
        
        # Update UI elements
        updated_elements.extend(self.vis.update_ui(frame_num, self.camera))
        
        # Update camera parameters based on current camera state
        self.camera_params['position'] = self.camera.position
        self.camera_params['rotation'] = self.camera.rotation
        self.camera_params['view_angle'] = self.camera.view_angle
        
        # Update camera POV visualization for all cars
        camera_pov_elements = self.vis.update_camera_pov(self.cars, self.camera)
        updated_elements.extend(camera_pov_elements)
        
        return updated_elements
    
    def run(self, num_frames=1000, interval=50):
        """
        Run the simulation
        
        Parameters:
        -----------
        num_frames : int
            Number of frames to run
        interval : int
            Interval between frames in milliseconds
            
        Returns:
        --------
        animation : FuncAnimation
            The animation object
        """
        self.animation = FuncAnimation(
            self.vis.fig, 
            self.animate, 
            frames=None,  # Run indefinitely
            interval=interval, 
            blit=True,
            cache_frame_data=False
        )
        
        self.animation_running = True
        
        plt.tight_layout()
        plt.show()
        return self.animation
    
    def save_animation(self, filename, writer='ffmpeg', fps=20, dpi=100):
        """
        Save the animation to a file
        
        Parameters:
        -----------
        filename : str
            Filename to save to
        writer : str
            Animation writer to use
        fps : int
            Frames per second
        dpi : int
            DPI for the saved file
        """
        if self.animation is None:
            print("No animation to save. Run the simulation first.")
            return
        
        self.animation.save(
            filename,
            writer=writer,
            fps=fps,
            dpi=dpi
        )
        print(f"Animation saved to {filename}")
    
    def set_noise_model(self, model='gaussian', params=None):
        """
        Set the noise model for all cars
        
        Parameters:
        -----------
        model : str
            Name of the noise model to use ('gaussian', 'twitch', 'oscillation', 'occlusion')
        params : dict or None
            Parameters for the noise model
        """
        if params is None:
            if model == 'gaussian':
                params = {'mean': 0, 'std': 2}
            elif model == 'twitch':
                params = {'twitch_prob': 0.1, 'twitch_magnitude': 10}
            elif model == 'occlusion':
                params = {'occlusion_prob': 0.05, 'occlusion_duration': 10}
            elif model == 'oscillation':
                params = {'oscillation_amplitude': 5, 'oscillation_frequency': 0.1}
        
        self.noise_model = model
        self.noise_params = params
        
        # Update camera to use the new noise model when updating tracking
        for car in self.cars:
            # Reset any car-specific noise states if changing models
            if hasattr(car, 'occlusion_state'):
                delattr(car, 'occlusion_state')
            if hasattr(car, 'occlusion_counter'):
                delattr(car, 'occlusion_counter')
            if hasattr(car, 'oscillation_frame'):
                delattr(car, 'oscillation_frame')
    
    def toggle_bounding_box_visibility(self, visible=None):
        """
        Toggle visibility of bounding boxes
        
        Parameters:
        -----------
        visible : bool or None
            If None, toggle current state. Otherwise, set to specified state.
            
        Returns:
        --------
        visible : bool
            The new visibility state
        """
        return self.vis.toggle_bbox_visibility(visible)
    
    def toggle_car_visibility(self, visible=None):
        """
        Toggle visibility of cars
        
        Parameters:
        -----------
        visible : bool or None
            If None, toggle current state. Otherwise, set to specified state.
            
        Returns:
        --------
        visible : bool
            The new visibility state
        """
        if visible is not None:
            for car in self.cars:
                if car.patch:
                    car.patch.set_visible(visible)
        else:
            current_state = self.cars[0].patch.get_visible() if self.cars else True
            for car in self.cars:
                if car.patch:
                    car.patch.set_visible(not current_state)
            visible = not current_state
            
        return visible
    
    def toggle_camera_visibility(self, visible=None):
        """
        Toggle visibility of camera visualization
        
        Parameters:
        -----------
        visible : bool or None
            If None, toggle current state. Otherwise, set to specified state.
            
        Returns:
        --------
        visible : bool
            The new visibility state
        """
        return self.vis.toggle_camera_visibility(visible)
    
    def toggle_tracking_visibility(self, visible=None):
        """
        Toggle visibility of tracking history
        
        Parameters:
        -----------
        visible : bool or None
            If None, toggle current state. Otherwise, set to specified state.
            
        Returns:
        --------
        visible : bool
            The new visibility state
        """
        return self.vis.toggle_tracking_visibility(visible)
    
    def change_camera_position(self, position=None, rotation=None):
        """
        Change camera position or rotation
        
        Parameters:
        -----------
        position : tuple or None
            New (x, y) position for the camera
        rotation : float or None
            New rotation angle in degrees
        """
        if position is not None:
            self.camera.position = np.array(position)
            
        if rotation is not None:
            self.camera.rotation = rotation
            
        # Update camera parameters for POV view
        if position is not None or rotation is not None:
            self.camera_params['position'] = self.camera.position
            self.camera_params['rotation'] = self.camera.rotation
            
            # Update the CameraPOV object
            if hasattr(self.vis, 'camera_pov') and self.vis.camera_pov is not None:
                self.vis.camera_pov.update_camera_params(self.camera_params)
            
        # Update camera visualization
        self.camera.update_visualization()
    
    def get_state(self):
        """
        Get the current state of the simulation
        
        Returns:
        --------
        state : dict
            Dictionary containing simulation state
        """
        return {
            'frame': self.frame_num,
            'num_cars': len(self.cars),
            'car_positions': [car.position.tolist() for car in self.cars],
            'car_track_positions': [car.track_position for car in self.cars],
            'tracking': self.camera.tracking,
            'bbox_visible': self.vis.show_bbox,
            'car_visible': self.vis.show_car,
            'camera_visible': self.vis.show_camera,
            'tracking_visible': self.vis.show_tracking,
            'camera_params': self.camera_params
        }