import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines

class Camera:
    def __init__(self, position=(650, 100), view_angle=60, range=300, rotation=135):
        """
        Initialize a camera object
        
        Parameters:
        -----------
        position : tuple
            (x, y) coordinates of the camera
        view_angle : float
            Field of view angle in degrees
        range : float
            Maximum viewing distance
        rotation : float
            Camera rotation in degrees (clockwise from East)
        """
        self.position = np.array(position)
        self.view_angle = view_angle
        self.range = range
        self.rotation = rotation
        self.tracking = False
        
        # Tracking history
        self.tracking_history = []
        
        # Visualization elements
        self.camera_point = None
        self.fov_left = None
        self.fov_right = None
        self.fov_area = None
        self.tracking_points = None
        
        # Reference to simulation (will be set by simulation)
        self.sim = None
    
    def is_in_view(self, car):
        """
        Check if a car is within the camera's field of view
        
        Parameters:
        -----------
        car : Car
            The car to check
            
        Returns:
        --------
        in_view : bool
            True if the car is in view, False otherwise
        """
        # Get car position
        car_pos = car.position
        
        # Calculate vector from camera to car
        to_car = car_pos - self.position
        distance = np.linalg.norm(to_car)
        
        # Check if car is within camera range
        if distance > self.range:
            car.in_camera_view = False
            return False
        
        # Calculate angle to car (in radians)
        angle_to_car = np.arctan2(to_car[1], to_car[0])
        
        # Convert camera rotation to radians
        cam_angle = np.radians(self.rotation)
        
        # Calculate angular difference (ensure it's in [-pi, pi])
        angle_diff = np.abs(angle_to_car - cam_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # Check if car is within camera's field of view
        half_fov = np.radians(self.view_angle / 2)
        in_view = angle_diff <= half_fov
        
        # Update car's in_camera_view flag
        car.in_camera_view = in_view
        return in_view
    
    def update_tracking(self, cars, noise_model='gaussian', noise_params=None):
        """
        Update tracking for all cars
        
        Parameters:
        -----------
        cars : list
            List of Car objects to track
        noise_model : str
            Type of noise for bounding box detection
        noise_params : dict or None
            Parameters for the noise model
        """
        if noise_params is None:
            noise_params = {'mean': 0, 'std': 2}
            
        # Reset tracking flag
        self.tracking = False
        
        # Process each car
        for car in cars:
            # Check if car is in view
            in_view = self.is_in_view(car)
            
            # Get detection bounding box if car is in view
            if in_view:
                # Use the simulation's current noise model and parameters if available
                if hasattr(self, 'sim') and self.sim is not None:
                    car.bounding_box = car.get_bounding_box(
                        self.sim.noise_model, 
                        self.sim.noise_params
                    )
                else:
                    car.bounding_box = car.get_bounding_box(noise_model, noise_params)
            else:
                car.bounding_box = None
            
            # Update tracking status
            if car.bounding_box is not None:
                self.tracking = True
                
                # Calculate center of bounding box
                bbox = car.bounding_box
                bbox_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                self.tracking_history.append(bbox_center)
                
                # Keep history at reasonable size
                if len(self.tracking_history) > 50:
                    self.tracking_history.pop(0)
    
    def create_visualization(self, ax):
        """
        Create camera visualization elements
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw the camera on
        """
        # Draw camera point
        self.camera_point = ax.plot(self.position[0], self.position[1], 'bo', markersize=10)[0]
        
        # Calculate FOV edges
        half_angle = self.view_angle / 2
        base_angle = np.radians(self.rotation)
        
        # Calculate angles for FOV edges
        left_angle = base_angle - np.radians(half_angle)
        right_angle = base_angle + np.radians(half_angle)
        
        # Calculate endpoints for FOV edges
        range_val = self.range
        left_x = self.position[0] + range_val * np.cos(left_angle)
        left_y = self.position[1] + range_val * np.sin(left_angle)
        right_x = self.position[0] + range_val * np.cos(right_angle)
        right_y = self.position[1] + range_val * np.sin(right_angle)
        
        # Draw FOV edges
        self.fov_left = ax.plot([self.position[0], left_x], [self.position[1], left_y], 'b--', alpha=0.5)[0]
        self.fov_right = ax.plot([self.position[0], right_x], [self.position[1], right_y], 'b--', alpha=0.5)[0]
        
        # Create a polygon for the FOV area
        points = np.array([[self.position[0], self.position[1]], [left_x, left_y], [right_x, right_y]])
        self.fov_area = patches.Polygon(points, alpha=0.1, color='blue')
        ax.add_patch(self.fov_area)
        
        # Initialize tracking history visualization
        self.tracking_points = ax.plot([], [], 'g.', alpha=0.5)[0]
        
        # Return visualization elements
        return [self.camera_point, self.fov_left, self.fov_right, self.fov_area, self.tracking_points]
    
    def update_visualization(self):
        """Update camera visualization"""
        # Check if visualization exists
        if self.fov_left is None or self.fov_right is None or self.fov_area is None:
            return []
        
        # Calculate FOV edges
        half_angle = self.view_angle / 2
        base_angle = np.radians(self.rotation)
        
        # Calculate angles for FOV edges
        left_angle = base_angle - np.radians(half_angle)
        right_angle = base_angle + np.radians(half_angle)
        
        # Calculate endpoints for FOV edges
        cam_x, cam_y = self.position
        range_val = self.range
        left_x = cam_x + range_val * np.cos(left_angle)
        left_y = cam_y + range_val * np.sin(left_angle)
        right_x = cam_x + range_val * np.cos(right_angle)
        right_y = cam_y + range_val * np.sin(right_angle)
        
        # Update FOV edges
        self.fov_left.set_data([cam_x, left_x], [cam_y, left_y])
        self.fov_right.set_data([cam_x, right_x], [cam_y, right_y])
        
        # Update FOV area
        points = np.array([[cam_x, cam_y], [left_x, left_y], [right_x, right_y]])
        self.fov_area.set_xy(points)
        
        # Update tracking history visualization
        if self.tracking_history:
            x_points = [p[0] for p in self.tracking_history]
            y_points = [p[1] for p in self.tracking_history]
            self.tracking_points.set_data(x_points, y_points)
        
        # Return updated elements
        return [self.fov_left, self.fov_right, self.fov_area, self.tracking_points]