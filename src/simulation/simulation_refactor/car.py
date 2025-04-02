import numpy as np
import matplotlib.patches as patches

class Car:
    def __init__(self, track, id=0, width=30, height=15, speed=1.0, start_position=0, color='red'):
        """
        Initialize a car object
        
        Parameters:
        -----------
        track : Track
            The track the car will drive on
        id : int
            Unique identifier for the car
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
        """
        self.id = id
        self.track = track
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        
        # Initialize position
        self.track_position = start_position
        self.position, self.angle = self.track.get_position_at(self.track_position)
        
        # Tracking related attributes
        self.in_camera_view = False
        self.bounding_box = None
        
        # Visualization elements
        self.patch = None
    
    def move(self):
        """Update car position along the track"""
        # Update track position
        self.track_position = (self.track_position + self.speed) % len(self.track.track_points)
        
        # Update position and angle
        self.position, self.angle = self.track.get_position_at(self.track_position)
    
    def create_visualization(self, ax):
        """
        Create car visualization
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw the car on
        """
        # Create car visualization (as a rectangle)
        car_x, car_y = self.position
        self.patch = patches.Rectangle(
            (car_x - self.width/2, car_y - self.height/2),
            self.width, self.height,
            angle=0,
            fc=self.color,
            ec='black',
            label=f'Car {self.id}'
        )
        ax.add_patch(self.patch)
        return self.patch
    
    def update_visualization(self):
        """Update car visualization position and orientation"""
        if self.patch is not None:
            car_x, car_y = self.position
            self.patch.set_xy((car_x - self.width/2, car_y - self.height/2))
            self.patch.angle = np.degrees(self.angle)
            
    def get_bounding_box(self, noise_model='gaussian', noise_params=None):
        """
        Get car's bounding box with optional noise to simulate detection
        
        Parameters:
        -----------
        noise_model : str
            Type of noise to add ('gaussian', 'twitch', 'oscillation', 'occlusion', etc.)
        noise_params : dict
            Parameters for the noise model
            
        Returns:
        --------
        bbox : list or None
            Bounding box as [x1, y1, x2, y2] or None if not detected
        """
        # Default noise parameters
        if noise_params is None:
            noise_params = {
                'mean': 0, 
                'std': 2,
                'twitch_prob': 0.1,
                'twitch_magnitude': 10,
                'occlusion_prob': 0.05,
                'occlusion_duration': 10,
                'oscillation_amplitude': 5,
                'oscillation_frequency': 0.1
            }
        
        # If car is not in view, return None
        if not self.in_camera_view:
            return None
        
        # Simulate occlusion if using that model
        if noise_model == 'occlusion':
            # Use ID and frame number for deterministic but pseudo-random occlusions
            if not hasattr(self, 'occlusion_state'):
                self.occlusion_state = False
                self.occlusion_counter = 0
            
            # Manage occlusion state
            if self.occlusion_state:
                self.occlusion_counter -= 1
                if self.occlusion_counter <= 0:
                    self.occlusion_state = False
            elif np.random.rand() < noise_params.get('occlusion_prob', 0.05):
                self.occlusion_state = True
                self.occlusion_counter = int(noise_params.get('occlusion_duration', 10))
            
            # Return None during occlusion
            if self.occlusion_state:
                return None
        
        # Add random chance of detection failure (5%)
        elif np.random.rand() < 0.05:
            return None
        
        # Get car position
        car_x, car_y = self.position
        
        # Basic bounding box (without rotation for simplicity)
        width, height = self.width, self.height
        bbox = [
            car_x - width/2, 
            car_y - height/2,
            car_x + width/2, 
            car_y + height/2
        ]
        
        # Apply noise model
        if noise_model == 'gaussian':
            noise = np.random.normal(
                noise_params.get('mean', 0),
                noise_params.get('std', 2),
                4
            )
            bbox = [
                bbox[0] + noise[0],
                bbox[1] + noise[1],
                bbox[2] + noise[2],
                bbox[3] + noise[3]
            ]
        
        elif noise_model == 'twitch':
            # Simulate twitching by occasionally adding larger jumps
            if np.random.rand() < noise_params.get('twitch_prob', 0.1):
                magnitude = noise_params.get('twitch_magnitude', 10)
                direction = np.random.rand(2) * 2 - 1  # Random direction
                bbox[0] += direction[0] * magnitude
                bbox[1] += direction[1] * magnitude
                bbox[2] += direction[0] * magnitude
                bbox[3] += direction[1] * magnitude
        
        elif noise_model == 'oscillation':
            # Track frame number for oscillation if not already tracking
            if not hasattr(self, 'oscillation_frame'):
                self.oscillation_frame = 0
            
            # Increment frame counter
            self.oscillation_frame += 1
            
            # Calculate oscillation effect
            amplitude = noise_params.get('oscillation_amplitude', 5)
            frequency = noise_params.get('oscillation_frequency', 0.1)
            
            # Use sine waves with different phases for each edge
            # This creates a more natural "breathing" effect for the bounding box
            bbox[0] += amplitude * np.sin(frequency * self.oscillation_frame)
            bbox[1] += amplitude * np.sin(frequency * self.oscillation_frame + np.pi/2)
            bbox[2] += amplitude * np.sin(frequency * self.oscillation_frame + np.pi)
            bbox[3] += amplitude * np.sin(frequency * self.oscillation_frame + 3*np.pi/2)
        
        # Add distance-based noise 
        # (farther objects have more jittery bounding boxes)
        if hasattr(self, 'distance_to_camera') and self.distance_to_camera is not None:
            distance_factor = self.distance_to_camera / 100  # Scale based on distance
            distance_noise = np.random.normal(0, distance_factor, 4)
            
            bbox[0] += distance_noise[0]
            bbox[1] += distance_noise[1]
            bbox[2] += distance_noise[2]
            bbox[3] += distance_noise[3]
        
        # Return the bounding box
        return bbox