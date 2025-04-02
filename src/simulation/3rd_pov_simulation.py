import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

class RealWorldTrackSimulator:
    def __init__(self):
        # Set up figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 500)
        
        # Create oval track
        self.create_oval_track()
        
        # Initialize the car
        self.car = {
            'track_position': 0,  # Position index along track
            'position': self.track_points[0],  # Initial position (x,y)
            'speed': 1.0,  # Movement speed
            'width': 30,  # Car width (for bounding box)
            'height': 15,  # Car height (for bounding box)
            'angle': 0,  # Car orientation
            'in_camera_view': False  # Whether the car is currently visible to the camera
        }
        
        # Set up fixed camera
        self.camera = {
            'position': np.array([650, 100]),  # Camera position (bottom right corner)
            'view_angle': 60,  # Camera field of view in degrees
            'range': 300,  # How far the camera can see
            'rotation': 135,  # Camera rotation in degrees (clockwise from East)
            'tracking': False  # Whether camera is currently tracking the car
        }
        
        # Initialize bounding box
        self.bbox = None
        
        # Initialize tracking
        self.tracking_history = []  # Store positions for visualization
        
        # Set up visualization
        self.setup_visualization()
        
    def create_oval_track(self):
        """Create a simple oval track"""
        # Track center and dimensions
        center_x, center_y = 400, 250
        width, height = 300, 150
        
        # Generate points along the oval
        self.track_points = []
        for angle in np.linspace(0, 2*np.pi, 100):
            x = center_x + width * np.cos(angle)
            y = center_y + height * np.sin(angle)
            self.track_points.append(np.array([x, y]))
        
        # Convert to numpy array
        self.track_points = np.array(self.track_points)
        
        # Create track boundaries
        self.track_width = 40
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
    
    def setup_visualization(self):
        """Set up visualization elements"""
        # Draw track
        self.ax.plot(self.track_points[:,0], self.track_points[:,1], 'k--', alpha=0.3)
        self.ax.fill(self.track_outer[:,0], self.track_outer[:,1], 'gray', alpha=0.3)
        self.ax.fill(self.track_inner[:,0], self.track_inner[:,1], 'white')
        
        # Start/finish line
        start_idx = 0
        finish_line_start = self.track_inner[start_idx]
        finish_line_end = self.track_outer[start_idx]
        self.ax.plot([finish_line_start[0], finish_line_end[0]], 
                    [finish_line_start[1], finish_line_end[1]], 
                    'k-', linewidth=2)
        
        # Create car visualization (as a rectangle)
        car_x, car_y = self.car['position']
        self.car_patch = patches.Rectangle(
            (car_x - self.car['width']/2, car_y - self.car['height']/2),
            self.car['width'], self.car['height'],
            angle=0,
            fc='red',
            ec='black',
            label='Car'
        )
        self.ax.add_patch(self.car_patch)
        
        # Create bounding box visualization (rectangle around the car)
        self.bbox_patch = patches.Rectangle(
            (0, 0),  # Will be updated in animate
            0, 0,  # Will be updated in animate
            fill=False,
            ec='green',
            linewidth=2,
            linestyle='--',
            visible=False,
            label='Detection Bounding Box'
        )
        self.ax.add_patch(self.bbox_patch)
        
        # Draw camera
        camera_x, camera_y = self.camera['position']
        self.camera_point = self.ax.plot(camera_x, camera_y, 'bo', markersize=10)[0]
        
        # Draw camera field of view
        self.draw_camera_fov()
        
        # Tracking status and history
        self.tracking_points = self.ax.plot([], [], 'g.', alpha=0.5)[0]
        
        # Add frame counter and status
        self.frame_counter = self.ax.text(50, 450, "Frame: 0", fontsize=10)
        self.status_text = self.ax.text(50, 430, "Tracking: No", fontsize=10)
        
        # Add legends and labels
        self.ax.set_title("Realistic Oval Track Simulation with Fixed Camera")
        self.ax.text(400, 250, "Track", ha='center')
        
        # Create custom legend
        car_patch = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Car')
        camera_patch = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Fixed Camera')
        bbox_patch = mlines.Line2D([], [], color='green', marker='s', linestyle='--',
                          markersize=10, fillstyle='none', label='Detection Bounding Box')
        
        self.ax.legend(handles=[car_patch, camera_patch, bbox_patch], loc='upper right')
    
    def draw_camera_fov(self):
        """Draw camera field of view as a cone"""
        # Camera position
        cam_x, cam_y = self.camera['position']
        
        # Calculate FOV edges
        half_angle = self.camera['view_angle'] / 2
        base_angle = np.radians(self.camera['rotation'])
        
        # Calculate angles for FOV edges
        left_angle = base_angle - np.radians(half_angle)
        right_angle = base_angle + np.radians(half_angle)
        
        # Calculate endpoints for FOV edges
        range_val = self.camera['range']
        left_x = cam_x + range_val * np.cos(left_angle)
        left_y = cam_y + range_val * np.sin(left_angle)
        right_x = cam_x + range_val * np.cos(right_angle)
        right_y = cam_y + range_val * np.sin(right_angle)
        
        # Draw FOV edges
        self.fov_left = self.ax.plot([cam_x, left_x], [cam_y, left_y], 'b--', alpha=0.5)[0]
        self.fov_right = self.ax.plot([cam_x, right_x], [cam_y, right_y], 'b--', alpha=0.5)[0]
        
        # Create a polygon for the FOV area
        points = np.array([[cam_x, cam_y], [left_x, left_y], [right_x, right_y]])
        self.fov_area = patches.Polygon(points, alpha=0.1, color='blue')
        self.ax.add_patch(self.fov_area)
    
    def update_camera_fov(self):
        """Update camera field of view visualization"""
        # Camera position
        cam_x, cam_y = self.camera['position']
        
        # Calculate FOV edges
        half_angle = self.camera['view_angle'] / 2
        base_angle = np.radians(self.camera['rotation'])
        
        # Calculate angles for FOV edges
        left_angle = base_angle - np.radians(half_angle)
        right_angle = base_angle + np.radians(half_angle)
        
        # Calculate endpoints for FOV edges
        range_val = self.camera['range']
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
    
    def move_car(self):
        """Update car position along the track"""
        # Update track position
        self.car['track_position'] = (self.car['track_position'] + self.car['speed']) % len(self.track_points)
        
        # Get integer position and next position
        pos_idx = int(self.car['track_position'])
        next_idx = (pos_idx + 1) % len(self.track_points)
        
        # Interpolate between points for smooth movement
        frac = self.car['track_position'] - pos_idx
        self.car['position'] = (1 - frac) * self.track_points[pos_idx] + frac * self.track_points[next_idx]
        
        # Calculate car orientation
        direction = self.track_points[next_idx] - self.track_points[pos_idx]
        self.car['angle'] = np.arctan2(direction[1], direction[0])
    
    def check_if_in_camera_view(self):
        """Check if the car is within the camera's field of view"""
        # Get car and camera positions
        car_pos = self.car['position']
        cam_pos = self.camera['position']
        
        # Calculate vector from camera to car
        to_car = car_pos - cam_pos
        distance = np.linalg.norm(to_car)
        
        # Check if car is within camera range
        if distance > self.camera['range']:
            self.car['in_camera_view'] = False
            return False
        
        # Calculate angle to car (in radians)
        angle_to_car = np.arctan2(to_car[1], to_car[0])
        
        # Convert camera rotation to radians
        cam_angle = np.radians(self.camera['rotation'])
        
        # Calculate angular difference (ensure it's in [-pi, pi])
        angle_diff = np.abs(angle_to_car - cam_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # Check if car is within camera's field of view
        half_fov = np.radians(self.camera['view_angle'] / 2)
        in_view = angle_diff <= half_fov
        
        self.car['in_camera_view'] = in_view
        return in_view
    
    def get_bounding_box(self):
        """Get the car's bounding box with noise to simulate detection"""
        # If car is not in view, return None
        if not self.car['in_camera_view']:
            return None
        
        # Get car position
        car_x, car_y = self.car['position']
        
        # Add random chance of detection failure (5%)
        if np.random.rand() < 0.05:
            return None
        
        # Basic bounding box (without rotation for simplicity)
        width, height = self.car['width'], self.car['height']
        
        # Add random noise to simulate detection variations
        noise = np.random.normal(0, 2, 4)  # Gaussian noise
        
        # Return as [x1, y1, x2, y2]
        return [
            car_x - width/2 + noise[0], 
            car_y - height/2 + noise[1],
            car_x + width/2 + noise[2], 
            car_y + height/2 + noise[3]
        ]
    
    def update_tracking(self):
        """Update car tracking based on camera view"""
        # Check if car is in camera view
        in_view = self.check_if_in_camera_view()
        
        # Get detection bounding box if car is in view
        self.bbox = self.get_bounding_box() if in_view else None
        
        # Update tracking status
        self.camera['tracking'] = self.bbox is not None
        
        # If tracking, add to history
        if self.camera['tracking']:
            # Calculate center of bounding box
            bbox_center = [(self.bbox[0] + self.bbox[2])/2, (self.bbox[1] + self.bbox[3])/2]
            self.tracking_history.append(bbox_center)
            
            # Keep history at reasonable size
            if len(self.tracking_history) > 50:
                self.tracking_history.pop(0)
    
    def update_visualization(self, frame_num):
        """Update visualization elements"""
        # Update car patch position and orientation
        car_x, car_y = self.car['position']
        self.car_patch.set_xy((car_x - self.car['width']/2, car_y - self.car['height']/2))
        self.car_patch.angle = np.degrees(self.car['angle'])
        
        # Update bounding box if tracking
        if self.bbox is not None:
            self.bbox_patch.set_xy((self.bbox[0], self.bbox[1]))
            self.bbox_patch.set_width(self.bbox[2] - self.bbox[0])
            self.bbox_patch.set_height(self.bbox[3] - self.bbox[1])
            self.bbox_patch.set_visible(True)
        else:
            self.bbox_patch.set_visible(False)
        
        # Update tracking history visualization
        if self.tracking_history:
            x_points = [p[0] for p in self.tracking_history]
            y_points = [p[1] for p in self.tracking_history]
            self.tracking_points.set_data(x_points, y_points)
        
        # Update frame counter and status
        self.frame_counter.set_text(f"Frame: {frame_num}")
        status = "Tracking: Yes" if self.camera['tracking'] else "Tracking: No"
        self.status_text.set_text(status)
        
        # Update camera FOV
        self.update_camera_fov()
    
    def animate(self, frame_num):
        """Animation function"""
        # Move the car
        self.move_car()
        
        # Update tracking
        self.update_tracking()
        
        # Update visualization
        self.update_visualization(frame_num)
        
        return (self.car_patch, self.bbox_patch, self.frame_counter, 
                self.status_text, self.fov_left, self.fov_right, 
                self.fov_area, self.tracking_points)
    
    def run_simulation(self, num_frames=200):
        """Run the simulation"""
        ani = FuncAnimation(self.fig, self.animate, frames=num_frames,
                           interval=50, blit=True)
        plt.show()
        return ani

# Run the simulation
if __name__ == "__main__":
    simulator = RealWorldTrackSimulator()
    ani = simulator.run_simulation(num_frames=200)
    
    # If you want to save the animation
    # ani.save('oval_track_simulation.mp4', writer='ffmpeg')