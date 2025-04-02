import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Track:
    def __init__(self, center=(400, 250), dimensions=(300, 150), track_width=40, num_points=100):
        """
        Initialize track with customizable parameters
        
        Parameters:
        -----------
        center : tuple
            Center coordinates (x, y) of the track
        dimensions : tuple
            Width and height of the oval track
        track_width : float
            Width of the track (distance between inner and outer edge)
        num_points : int
            Number of points used to define the track curve
        """
        self.center_x, self.center_y = center
        self.width, self.height = dimensions
        self.track_width = track_width
        self.num_points = num_points
        
        # Generate track points and boundaries
        self.generate_track_points()
        self.generate_track_boundaries()
    
    def generate_track_points(self):
        """Generate points along the center of the oval track"""
        self.track_points = []
        for angle in np.linspace(0, 2*np.pi, self.num_points):
            x = self.center_x + self.width * np.cos(angle)
            y = self.center_y + self.height * np.sin(angle)
            self.track_points.append(np.array([x, y]))
        
        # Convert to numpy array
        self.track_points = np.array(self.track_points)
    
    def generate_track_boundaries(self):
        """Generate inner and outer track boundaries"""
        self.track_outer = []
        self.track_inner = []
        
        # Calculate inner and outer track points
        for i in range(len(self.track_points)):
            # Get direction vector
            prev_idx = (i - 1) % len(self.track_points)
            next_idx = (i + 1) % len(self.track_points)
            
            # Calculate tangent
            tangent = self.track_points[next_idx] - self.track_points[prev_idx]
            tangent = tangent / np.linalg.norm(tangent)
            
            # Normal vector (perpendicular to tangent)
            normal = np.array([-tangent[1], tangent[0]])
            
            # Calculate inner and outer track points
            self.track_inner.append(self.track_points[i] - (self.track_width/2) * normal)
            self.track_outer.append(self.track_points[i] + (self.track_width/2) * normal)
        
        # Convert to numpy arrays
        self.track_inner = np.array(self.track_inner)
        self.track_outer = np.array(self.track_outer)
    
    def draw_track(self, ax):
        """
        Draw the track on a given matplotlib axis
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw the track on
        """
        # Draw track center line
        ax.plot(self.track_points[:,0], self.track_points[:,1], 'k--', alpha=0.3)
        
        # Draw track boundaries
        ax.fill(self.track_outer[:,0], self.track_outer[:,1], 'gray', alpha=0.3)
        ax.fill(self.track_inner[:,0], self.track_inner[:,1], 'white')
        
        # Draw start/finish line
        start_idx = 0
        finish_line_start = self.track_inner[start_idx]
        finish_line_end = self.track_outer[start_idx]
        ax.plot([finish_line_start[0], finish_line_end[0]], 
                [finish_line_start[1], finish_line_end[1]], 
                'k-', linewidth=2)
        
        # Add track label
        ax.text(self.center_x, self.center_y, "Track", ha='center')
        
        return ax
    
    def get_position_at(self, track_position):
        """
        Get position coordinates at a specific track position
        
        Parameters:
        -----------
        track_position : float
            Position along the track (can be fractional)
            
        Returns:
        --------
        position : numpy.ndarray
            (x, y) coordinates at the specified track position
        angle : float
            Orientation angle at the specified track position (in radians)
        """
        # Get integer position and next position
        pos_idx = int(track_position) % len(self.track_points)
        next_idx = (pos_idx + 1) % len(self.track_points)
        
        # Interpolate between points for smooth movement
        frac = track_position - int(track_position)
        position = (1 - frac) * self.track_points[pos_idx] + frac * self.track_points[next_idx]
        
        # Calculate orientation
        direction = self.track_points[next_idx] - self.track_points[pos_idx]
        angle = np.arctan2(direction[1], direction[0])
        
        return position, angle