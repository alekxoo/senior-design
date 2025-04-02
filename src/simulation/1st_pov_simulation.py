import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
import os
from datetime import datetime
import pickle

class CameraPOVSimulator:
    def __init__(self):
        # Set up figure with two subplots: overhead view and camera POV
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create grid for plots and controls
        gs = self.fig.add_gridspec(3, 2)
        
        # Setup main plot areas
        self.ax_overhead = self.fig.add_subplot(gs[0:2, 0])
        self.ax_camera = self.fig.add_subplot(gs[0:2, 1])
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        
        # Configure overhead view
        self.ax_overhead.set_xlim(0, 800)
        self.ax_overhead.set_ylim(0, 500)
        self.ax_overhead.set_title("Overhead View")
        
        # Configure camera POV view
        self.ax_camera.set_xlim(0, 640)
        self.ax_camera.set_ylim(0, 480)
        self.ax_camera.set_title("Camera POV")
        self.ax_camera.set_facecolor('black')  # Black background for camera view
        
        # Turn off controls area for sliders
        self.ax_controls.axis('off')
        
        # Define default parameters
        self.default_camera = {
            'position': np.array([650, 100]),  # Camera position (bottom right corner)
            'view_angle': 70,  # Camera field of view in degrees
            'range': 400,  # How far the camera can see
            'rotation': 135,  # Camera rotation in degrees (clockwise from East)
            'focal_length': 800,  # Simulated focal length for perspective
            'sensor_width': 640,  # Width of camera sensor/image
            'sensor_height': 480,  # Height of camera sensor/image
        }
        
        self.default_car = {
            'track_position': 0,  # Position index along track
            'speed': 2.0,  # Movement speed
            'width': 30,  # Car width
            'height': 15,  # Car height
        }
        
        # Camera properties (copy from defaults)
        self.camera = self.default_camera.copy()
        self.camera['position'] = np.array(self.default_camera['position'])
        
        # Create oval track
        self.create_oval_track()
        
        # Initialize the car (copy from defaults)
        self.car = self.default_car.copy()
        self.car['position'] = self.track_points[0]  # Initial position (x,y)
        self.car['angle'] = 0  # Car orientation
        self.car['in_camera_view'] = False  # Whether the car is currently visible
        
        # Animation state
        self.animation_running = False
        self.animation = None
        self.current_frame = 0
        
        # Recording state
        self.recording = False
        self.frames_data = []
        
        # Set up visualization
        self.setup_visualization()
        
        # Add GUI controls
        self.setup_gui_controls()
        
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
        """Set up visualization elements for both views"""
        # --- Overhead View Setup ---
        # Draw track
        self.ax_overhead.plot(self.track_points[:,0], self.track_points[:,1], 'k--', alpha=0.3)
        self.ax_overhead.fill(self.track_outer[:,0], self.track_outer[:,1], 'gray', alpha=0.3)
        self.ax_overhead.fill(self.track_inner[:,0], self.track_inner[:,1], 'white')
        
        # Start/finish line
        start_idx = 0
        finish_line_start = self.track_inner[start_idx]
        finish_line_end = self.track_outer[start_idx]
        self.ax_overhead.plot([finish_line_start[0], finish_line_end[0]], 
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
        )
        self.ax_overhead.add_patch(self.car_patch)
        
        # Draw camera in overhead view
        camera_x, camera_y = self.camera['position']
        self.camera_point = self.ax_overhead.plot(camera_x, camera_y, 'bo', markersize=10)[0]
        
        # Draw camera field of view
        self.draw_camera_fov()
        
        # --- Camera POV Setup ---
        # Create car visualization in camera view (initially off-screen)
        self.car_pov = patches.Rectangle(
            (-100, -100),  # Off-screen
            0, 0,  # Will be updated in animate
            angle=0,
            fc='red',
            ec='black',
            visible=False
        )
        self.ax_camera.add_patch(self.car_pov)
        
        # Create bounding box in camera view
        self.bbox_pov = patches.Rectangle(
            (-100, -100),  # Off-screen
            0, 0,  # Will be updated in animate
            fill=False,
            ec='green',
            linewidth=2,
            linestyle='--',
            visible=False
        )
        self.ax_camera.add_patch(self.bbox_pov)
        
        # Add info text
        self.info_text = self.ax_camera.text(10, 30, "", color='white', fontsize=10)
        self.distance_text = self.ax_camera.text(10, 10, "", color='white', fontsize=10)
        
        # Add frame counter
        self.frame_counter = self.ax_overhead.text(50, 450, "Frame: 0", fontsize=10)
        
        # Add recording indicator
        self.recording_text = self.ax_overhead.text(650, 450, "", color='red', fontsize=10, fontweight='bold')
        
        # Add camera position and settings text
        self.camera_info = self.ax_overhead.text(50, 50, self.get_camera_info_text(), fontsize=8)
        self.car_info = self.ax_overhead.text(50, 30, self.get_car_info_text(), fontsize=8)
    
    def get_camera_info_text(self):
        """Get camera information text"""
        return (f"Camera: pos=({self.camera['position'][0]:.0f}, {self.camera['position'][1]:.0f}), "
                f"FOV={self.camera['view_angle']:.0f}°, "
                f"rot={self.camera['rotation']:.0f}°, "
                f"range={self.camera['range']:.0f}")
    
    def get_car_info_text(self):
        """Get car information text"""
        return (f"Car: speed={self.car['speed']:.1f}, "
                f"size={self.car['width']:.0f}x{self.car['height']:.0f}")
    
    def setup_gui_controls(self):
        """Set up GUI controls for simulation parameters"""
        # Add sliders for camera parameters
        plt.subplots_adjust(bottom=0.35)
        
        # Camera X position slider
        self.ax_cam_x = plt.axes([0.1, 0.26, 0.25, 0.03])
        self.slider_cam_x = Slider(
            ax=self.ax_cam_x,
            label='Camera X',
            valmin=0,
            valmax=800,
            valinit=self.camera['position'][0],
        )
        self.slider_cam_x.on_changed(self.update_camera_x)
        
        # Camera Y position slider
        self.ax_cam_y = plt.axes([0.1, 0.22, 0.25, 0.03])
        self.slider_cam_y = Slider(
            ax=self.ax_cam_y,
            label='Camera Y',
            valmin=0,
            valmax=500,
            valinit=self.camera['position'][1],
        )
        self.slider_cam_y.on_changed(self.update_camera_y)
        
        # Camera FOV slider
        self.ax_cam_fov = plt.axes([0.1, 0.18, 0.25, 0.03])
        self.slider_cam_fov = Slider(
            ax=self.ax_cam_fov,
            label='FOV (°)',
            valmin=10,
            valmax=180,
            valinit=self.camera['view_angle'],
        )
        self.slider_cam_fov.on_changed(self.update_camera_fov_angle)
        
        # Camera rotation slider
        self.ax_cam_rot = plt.axes([0.1, 0.14, 0.25, 0.03])
        self.slider_cam_rot = Slider(
            ax=self.ax_cam_rot,
            label='Rotation (°)',
            valmin=0,
            valmax=360,
            valinit=self.camera['rotation'],
        )
        self.slider_cam_rot.on_changed(self.update_camera_rotation)
        
        # Camera range slider
        self.ax_cam_range = plt.axes([0.1, 0.1, 0.25, 0.03])
        self.slider_cam_range = Slider(
            ax=self.ax_cam_range,
            label='Range',
            valmin=100,
            valmax=800,
            valinit=self.camera['range'],
        )
        self.slider_cam_range.on_changed(self.update_camera_range)
        
        # Car speed slider
        self.ax_car_speed = plt.axes([0.45, 0.22, 0.25, 0.03])
        self.slider_car_speed = Slider(
            ax=self.ax_car_speed,
            label='Car Speed',
            valmin=0.5,
            valmax=5.0,
            valinit=self.car['speed'],
        )
        self.slider_car_speed.on_changed(self.update_car_speed)
        
        # Car width slider
        self.ax_car_width = plt.axes([0.45, 0.18, 0.25, 0.03])
        self.slider_car_width = Slider(
            ax=self.ax_car_width,
            label='Car Width',
            valmin=10,
            valmax=50,
            valinit=self.car['width'],
        )
        self.slider_car_width.on_changed(self.update_car_width)
        
        # Car height slider
        self.ax_car_height = plt.axes([0.45, 0.14, 0.25, 0.03])
        self.slider_car_height = Slider(
            ax=self.ax_car_height,
            label='Car Height',
            valmin=5,
            valmax=30,
            valinit=self.car['height'],
        )
        self.slider_car_height.on_changed(self.update_car_height)
        
        # Add buttons
        button_width = 0.11
        button_spacing = 0.02
        
        # Play/pause button
        self.ax_play_button = plt.axes([0.1, 0.04, button_width, 0.04])
        self.play_button = Button(self.ax_play_button, 'Play')
        self.play_button.on_clicked(self.toggle_animation)
        
        # Reset button
        self.ax_reset_button = plt.axes([0.1 + button_width + button_spacing, 0.04, button_width, 0.04])
        self.reset_button = Button(self.ax_reset_button, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)
        
        # Save parameters button
        self.ax_save_params_button = plt.axes([0.45, 0.04, button_width, 0.04])
        self.save_params_button = Button(self.ax_save_params_button, 'Save Parameters')
        self.save_params_button.on_clicked(self.save_parameters)
        
        # Record simulation button
        self.ax_record_button = plt.axes([0.45 + button_width + button_spacing, 0.04, button_width, 0.04])
        self.record_button = Button(self.ax_record_button, 'Start Recording')
        self.record_button.on_clicked(self.toggle_recording)
    
    def update_camera_x(self, val):
        """Update camera X position"""
        self.camera['position'][0] = val
        self.camera_point.set_data([val], [self.camera['position'][1]])
        self.update_camera_fov()
        self.camera_info.set_text(self.get_camera_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_camera_y(self, val):
        """Update camera Y position"""
        self.camera['position'][1] = val
        self.camera_point.set_data([self.camera['position'][0]], [val])
        self.update_camera_fov()
        self.camera_info.set_text(self.get_camera_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_camera_fov_angle(self, val):
        """Update camera field of view angle"""
        self.camera['view_angle'] = val
        self.update_camera_fov()
        self.camera_info.set_text(self.get_camera_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_camera_rotation(self, val):
        """Update camera rotation"""
        self.camera['rotation'] = val
        self.update_camera_fov()
        self.camera_info.set_text(self.get_camera_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_camera_range(self, val):
        """Update camera range"""
        self.camera['range'] = val
        self.update_camera_fov()
        self.camera_info.set_text(self.get_camera_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_car_speed(self, val):
        """Update car speed"""
        self.car['speed'] = val
        self.car_info.set_text(self.get_car_info_text())
        self.fig.canvas.draw_idle()
    
    def update_car_width(self, val):
        """Update car width"""
        self.car['width'] = val
        car_x, car_y = self.car['position']
        self.car_patch.set_width(val)
        self.car_patch.set_x(car_x - val/2)
        self.car_info.set_text(self.get_car_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def update_car_height(self, val):
        """Update car height"""
        self.car['height'] = val
        car_x, car_y = self.car['position']
        self.car_patch.set_height(val)
        self.car_patch.set_y(car_y - val/2)
        self.car_info.set_text(self.get_car_info_text())
        # Update camera view
        self.project_to_camera_view()
        self.fig.canvas.draw_idle()
    
    def save_parameters(self, event):
        """Save current parameter settings as defaults"""
        # Update default camera settings
        self.default_camera = self.camera.copy()
        self.default_camera['position'] = np.array(self.camera['position'])  # Create a copy of the position array
        
        # Update default car settings
        self.default_car = {
            'track_position': 0,  # Reset track position
            'speed': self.car['speed'],
            'width': self.car['width'],
            'height': self.car['height']
        }
        
        # Save parameters to file
        parameters = {
            'camera': {
                'position_x': float(self.camera['position'][0]),
                'position_y': float(self.camera['position'][1]),
                'view_angle': float(self.camera['view_angle']),
                'range': float(self.camera['range']),
                'rotation': float(self.camera['rotation']),
                'focal_length': float(self.camera['focal_length']),
                'sensor_width': float(self.camera['sensor_width']),
                'sensor_height': float(self.camera['sensor_height']),
            },
            'car': {
                'speed': float(self.car['speed']),
                'width': float(self.car['width']),
                'height': float(self.car['height']),
            }
        }
        
        # Create parameters directory if it doesn't exist
        if not os.path.exists('parameters'):
            os.makedirs('parameters')
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parameters/camera_params_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Parameters saved to {filename}")
    
    def toggle_recording(self, event):
        """Toggle recording of simulation data"""
        if not self.recording:
            # Start recording
            self.recording = True
            self.frames_data = []
            self.record_button.label.set_text('Stop Recording')
            self.recording_text.set_text("● RECORDING")
        else:
            # Stop recording and save data
            self.recording = False
            self.record_button.label.set_text('Start Recording')
            self.recording_text.set_text("")
            
            if self.frames_data:
                self.save_recording()
        
        self.fig.canvas.draw_idle()
    
    def save_recording(self):
        """Save recorded simulation data"""
        # Create recordings directory if it doesn't exist
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/simulation_{timestamp}.pkl"
        
        # Record parameter state at end of recording
        parameters = {
            'camera': {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in self.camera.items()},
            'car': {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                   for k, v in self.car.items() if k != 'position' and k != 'angle' and k != 'in_camera_view'},
            'frames': self.frames_data
        }
        
        # Save data
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
        
        print(f"Recording saved to {filename}")
    
    def toggle_animation(self, event):
        """Toggle animation play/pause"""
        if self.animation_running:
            if self.animation is not None:
                self.animation.event_source.stop()
            self.animation_running = False
            self.play_button.label.set_text('Play')
        else:
            if self.animation is None:
                # Use blitting for improved performance
                self.animation = FuncAnimation(self.fig, self.animate, frames=None,
                                             interval=50, blit=True, cache_frame_data=False)
            else:
                self.animation.event_source.start()
            self.animation_running = True
            self.play_button.label.set_text('Pause')
        self.fig.canvas.draw_idle()
    
    def reset_simulation(self, event):
        """Reset simulation to initial state"""
        # Stop animation if running
        if self.animation_running and self.animation is not None:
            self.animation.event_source.stop()
            self.animation_running = False
            self.play_button.label.set_text('Play')
        
        # Reset car position
        self.car['track_position'] = 0
        self.car['position'] = self.track_points[0]
        self.current_frame = 0
        
        # Update car visualization
        car_x, car_y = self.car['position']
        self.car_patch.set_xy((car_x - self.car['width']/2, car_y - self.car['height']/2))
        
        # Update frame counter
        self.frame_counter.set_text("Frame: 0")
        
        # Update camera view
        self.project_to_camera_view()
        
        self.fig.canvas.draw_idle()
    
    def draw_camera_fov(self):
        """Draw camera field of view as a cone in overhead view"""
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
        self.fov_left = self.ax_overhead.plot([cam_x, left_x], [cam_y, left_y], 'b--', alpha=0.5)[0]
        self.fov_right = self.ax_overhead.plot([cam_x, right_x], [cam_y, right_y], 'b--', alpha=0.5)[0]
        
        # Create a polygon for the FOV area
        points = np.array([[cam_x, cam_y], [left_x, left_y], [right_x, right_y]])
        self.fov_area = patches.Polygon(points, alpha=0.1, color='blue')
        self.ax_overhead.add_patch(self.fov_area)
    
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
            return False, None
        
        # Calculate angle to car (in radians)
        angle_to_car = np.arctan2(to_car[1], to_car[0])
        
        # Convert camera rotation to radians
        cam_angle = np.radians(self.camera['rotation'])
        
        # Calculate angular difference (ensure it's in [-pi, pi])
        angle_diff = angle_to_car - cam_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Check if car is within camera's field of view
        half_fov = np.radians(self.camera['view_angle'] / 2)
        in_view = abs(angle_diff) <= half_fov
        
        self.car['in_camera_view'] = in_view
        return in_view, distance
    
    def project_to_camera_view(self):
        """Project car from 3D world to 2D camera view with perspective"""
        # Check if car is in camera view
        in_view, distance = self.check_if_in_camera_view()
        
        if not in_view:
            # Car is not visible to camera
            self.car_pov.set_visible(False)
            self.bbox_pov.set_visible(False)
            self.info_text.set_text("Car not in view")
            self.distance_text.set_text("")
            return None
        
        # Get car and camera positions
        car_pos = self.car['position']
        cam_pos = self.camera['position']
        
        # Vector from camera to car
        to_car = car_pos - cam_pos
        
        # Calculate angle to car relative to camera direction
        angle_to_car = np.arctan2(to_car[1], to_car[0])
        cam_angle = np.radians(self.camera['rotation'])
        
        # Normalize angle_diff to be between -pi and pi
        angle_diff = angle_to_car - cam_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Calculate horizontal position in image based on angle
        # Map from [-fov/2, fov/2] to [0, sensor_width]
        # Flipped the mapping to fix the direction issue
        half_fov = np.radians(self.camera['view_angle'] / 2)
        normalized_x = 1.0 - (angle_diff + half_fov) / (2 * half_fov)  # Flipped 
        screen_x = normalized_x * self.camera['sensor_width']
        
        # Scale size based on distance (perspective effect)
        # Size is inversely proportional to distance
        scale_factor = self.camera['focal_length'] / max(distance, 1)
        
        # Calculate projected size of car
        projected_width = self.car['width'] * scale_factor
        projected_height = self.car['height'] * scale_factor
        
        # Center car vertically in camera view (simplified)
        # In a real camera, vertical position would depend on 3D coordinates
        screen_y = self.camera['sensor_height'] / 2
        
        # Add some random noise to bounding box (detection noise)
        bbox_noise = np.random.normal(0, 2, 4)  # Gaussian noise
        
        # Calculate car and bounding box coordinates in camera view
        car_rect = [
            screen_x - projected_width/2,  # left
            screen_y - projected_height/2,  # top
            projected_width,  # width
            projected_height  # height
        ]
        
        bbox_rect = [
            car_rect[0] + bbox_noise[0],  # left with noise
            car_rect[1] + bbox_noise[1],  # top with noise
            car_rect[2] + bbox_noise[2],  # width with noise
            car_rect[3] + bbox_noise[3]   # height with noise
        ]
        
        # Occasionally fail detection (5% chance)
        if np.random.rand() < 0.05:
            detected = False
            self.bbox_pov.set_visible(False)
        else:
            detected = True
            self.bbox_pov.set_visible(True)
            self.bbox_pov.set_xy((bbox_rect[0], bbox_rect[1]))
            self.bbox_pov.set_width(bbox_rect[2])
            self.bbox_pov.set_height(bbox_rect[3])
        
        # Update car visualization in camera view
        self.car_pov.set_visible(True)
        self.car_pov.set_xy((car_rect[0], car_rect[1]))
        self.car_pov.set_width(car_rect[2])
        self.car_pov.set_height(car_rect[3])
        
        # Set car angle in camera view (simplified)
        # In reality, this would involve 3D rotation and projection
        self.car_pov.angle = 0
        
        # Update info text
        self.info_text.set_text("Car in view" + (" (Detected)" if detected else " (Not Detected)"))
        self.distance_text.set_text(f"Distance: {distance:.1f} units")
        
        return bbox_rect if detected else None
    
    def update_visualization(self, frame_num):
            """Update visualization elements"""
            # Update car patch in overhead view
            car_x, car_y = self.car['position']
            self.car_patch.set_xy((car_x - self.car['width']/2, car_y - self.car['height']/2))
            self.car_patch.angle = np.degrees(self.car['angle'])
            
            # Update camera FOV
            self.update_camera_fov()
            
            # Update frame counter
            self.frame_counter.set_text(f"Frame: {frame_num}")
        
    def animate(self, frame_num):
        """Animation function"""
        # Increment frame counter
        self.current_frame = frame_num
        
        # Move the car
        self.move_car()
        
        # Update overhead view
        self.update_visualization(frame_num)
        
        # Project car to camera view
        bbox_data = self.project_to_camera_view()
        
        # Record frame data if recording is active
        if self.recording:
            # Record the current state for replay
            frame_data = {
                'frame': self.current_frame,
                'car_position': self.car['position'].copy(),
                'car_angle': self.car['angle'],
                'in_camera_view': self.car['in_camera_view'],
                'bbox': bbox_data
            }
            self.frames_data.append(frame_data)
        
        return (self.car_patch, self.car_pov, self.bbox_pov, 
                self.info_text, self.distance_text, self.frame_counter,
                self.recording_text, self.fov_left, self.fov_right, self.fov_area)

    def run_simulation(self):
        """Run the simulation"""
        # Initial updates
        self.update_visualization(0)
        self.project_to_camera_view()
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, bottom=0.3, top=0.95)
        plt.show()

    @staticmethod
    def load_recording(filename):
        """Load a saved recording and replay it"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded recording with {len(data['frames'])} frames")
        return data


class RecordingPlayer:
    """Class to play back recorded simulations"""
    def __init__(self, recording_data):
        self.data = recording_data
        self.frames = self.data['frames']
        self.current_frame_idx = 0
        
        # Set up figure with two subplots: overhead view and camera POV
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create grid for plots and controls
        gs = self.fig.add_gridspec(3, 2)
        
        # Setup main plot areas
        self.ax_overhead = self.fig.add_subplot(gs[0:2, 0])
        self.ax_camera = self.fig.add_subplot(gs[0:2, 1])
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        
        # Configure overhead view
        self.ax_overhead.set_xlim(0, 800)
        self.ax_overhead.set_ylim(0, 500)
        self.ax_overhead.set_title("Replay - Overhead View")
        
        # Configure camera POV view
        self.ax_camera.set_xlim(0, 640)
        self.ax_camera.set_ylim(0, 480)
        self.ax_camera.set_title("Replay - Camera POV")
        self.ax_camera.set_facecolor('black')  # Black background for camera view
        
        # Turn off controls area for sliders
        self.ax_controls.axis('off')
        
        # Setup GUI controls
        self.setup_controls()
        
        # Extract camera properties
        self.camera = self.data['camera']
        try:
            # Convert position from list to numpy array if needed
            if isinstance(self.camera['position'], list):
                self.camera['position'] = np.array(self.camera['position'])
        except KeyError:
            # Handle case where position might be stored as x,y coordinates
            self.camera['position'] = np.array([
                self.camera.get('position_x', 650),
                self.camera.get('position_y', 100)
            ])
        
        # Extract car properties
        self.car = self.data['car']
        
        # Setup visualization elements
        self.setup_visualization()
        
        # Animation state
        self.animation_running = False
        self.animation = None

    def setup_controls(self):
        """Set up replay controls"""
        # Add playback control buttons
        plt.subplots_adjust(bottom=0.35)
        
        # Frame slider
        self.ax_frame = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.slider_frame = Slider(
            ax=self.ax_frame,
            label='Frame',
            valmin=0,
            valmax=len(self.frames) - 1,
            valinit=0,
            valstep=1
        )
        self.slider_frame.on_changed(self.update_frame)
        
        # Play/pause button
        self.ax_play_button = plt.axes([0.3, 0.05, 0.15, 0.04])
        self.play_button = Button(self.ax_play_button, 'Play')
        self.play_button.on_clicked(self.toggle_playback)
        
        # Reset button
        self.ax_reset_button = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.reset_button = Button(self.ax_reset_button, 'Reset')
        self.reset_button.on_clicked(self.reset_playback)

    def setup_visualization(self):
        """Set up visualization elements"""
        # Create oval track (simplified for replay)
        # This would ideally be saved in the recording, but we'll recreate it for now
        center_x, center_y = 400, 250
        width, height = 300, 150
        track_width = 40
        
        # Draw simplified track outline
        theta = np.linspace(0, 2*np.pi, 100)
        outer_x = center_x + (width + track_width/2) * np.cos(theta)
        outer_y = center_y + (height + track_width/2) * np.sin(theta)
        inner_x = center_x + (width - track_width/2) * np.cos(theta)
        inner_y = center_y + (height - track_width/2) * np.sin(theta)
        
        self.ax_overhead.fill(outer_x, outer_y, 'gray', alpha=0.3)
        self.ax_overhead.fill(inner_x, inner_y, 'white')
        
        # Create car visualization (as a rectangle)
        first_frame = self.frames[0]
        car_x, car_y = first_frame['car_position']
        self.car_patch = patches.Rectangle(
            (car_x - self.car['width']/2, car_y - self.car['height']/2),
            self.car['width'], self.car['height'],
            angle=np.degrees(first_frame['car_angle']),
            fc='red',
            ec='black',
        )
        self.ax_overhead.add_patch(self.car_patch)
        
        # Draw camera in overhead view
        camera_x, camera_y = self.camera['position']
        self.camera_point = self.ax_overhead.plot(camera_x, camera_y, 'bo', markersize=10)[0]
        
        # Draw camera field of view
        self.draw_camera_fov()
        
        # Create car visualization in camera view
        self.car_pov = patches.Rectangle(
            (-100, -100),  # Off-screen
            0, 0,          # Will be updated
            angle=0,
            fc='red',
            ec='black',
            visible=False
        )
        self.ax_camera.add_patch(self.car_pov)
        
        # Create bounding box in camera view
        self.bbox_pov = patches.Rectangle(
            (-100, -100),  # Off-screen
            0, 0,          # Will be updated
            fill=False,
            ec='green',
            linewidth=2,
            linestyle='--',
            visible=False
        )
        self.ax_camera.add_patch(self.bbox_pov)
        
        # Add info text
        self.info_text = self.ax_camera.text(10, 30, "", color='white', fontsize=10)
        self.distance_text = self.ax_camera.text(10, 10, "", color='white', fontsize=10)
        
        # Add frame counter
        self.frame_counter = self.ax_overhead.text(50, 450, "Frame: 0", fontsize=10)
        
        # Add playback indicator
        self.playback_text = self.ax_overhead.text(650, 450, "REPLAY MODE", 
                                                color='green', fontsize=10, fontweight='bold')

    def draw_camera_fov(self):
        """Draw camera field of view as a cone in overhead view"""
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
        self.fov_left = self.ax_overhead.plot([cam_x, left_x], [cam_y, left_y], 'b--', alpha=0.5)[0]
        self.fov_right = self.ax_overhead.plot([cam_x, right_x], [cam_y, right_y], 'b--', alpha=0.5)[0]
        
        # Create a polygon for the FOV area
        points = np.array([[cam_x, cam_y], [left_x, left_y], [right_x, right_y]])
        self.fov_area = patches.Polygon(points, alpha=0.1, color='blue')
        self.ax_overhead.add_patch(self.fov_area)

    def update_frame(self, val=None):
        """Update display to show the current frame"""
        if val is not None:
            self.current_frame_idx = int(val)
        
        frame_data = self.frames[self.current_frame_idx]
        
        # Update car position in overhead view
        car_x, car_y = frame_data['car_position']
        self.car_patch.set_xy((car_x - self.car['width']/2, car_y - self.car['height']/2))
        self.car_patch.angle = np.degrees(frame_data['car_angle'])
        
        # Update camera view based on recorded data
        if frame_data['in_camera_view'] and frame_data['bbox'] is not None:
            # Car is in view and detected
            bbox = frame_data['bbox']
            
            # Update car in camera view
            self.car_pov.set_visible(True)
            self.car_pov.set_xy((bbox[0], bbox[1]))
            self.car_pov.set_width(bbox[2])
            self.car_pov.set_height(bbox[3])
            
            # Update bounding box
            self.bbox_pov.set_visible(True)
            self.bbox_pov.set_xy((bbox[0], bbox[1]))
            self.bbox_pov.set_width(bbox[2])
            self.bbox_pov.set_height(bbox[3])
            
            # Update info text
            self.info_text.set_text("Car in view (Detected)")
            
            # Calculate approximate distance
            cam_pos = self.camera['position']
            car_pos = frame_data['car_position']
            distance = np.linalg.norm(np.array(car_pos) - cam_pos)
            self.distance_text.set_text(f"Distance: {distance:.1f} units")
        else:
            # Car is not in view or not detected
            self.car_pov.set_visible(False)
            self.bbox_pov.set_visible(False)
            self.info_text.set_text("Car not in view" if not frame_data['in_camera_view'] else "Car in view (Not Detected)")
            self.distance_text.set_text("")
        
        # Update frame counter
        self.frame_counter.set_text(f"Frame: {frame_data['frame']}")
        
        self.fig.canvas.draw_idle()

    def toggle_playback(self, event):
        """Toggle playback animation"""
        if self.animation_running:
            if self.animation is not None:
                self.animation.event_source.stop()
            self.animation_running = False
            self.play_button.label.set_text('Play')
        else:
            if self.animation is None:
                self.animation = FuncAnimation(self.fig, self.animate_playback, frames=None,
                                                interval=50, blit=False, cache_frame_data=False)
            else:
                self.animation.event_source.start()
            self.animation_running = True
            self.play_button.label.set_text('Pause')
        self.fig.canvas.draw_idle()

    def animate_playback(self, frame_num):
        """Animation function for playback"""
        # Advance frame
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        
        # Update slider to match current frame
        self.slider_frame.set_val(self.current_frame_idx)
        
        # No need to call update_frame as the slider callback will do it
        
        # Return updated artists (not used with blit=False)
        return []

    def reset_playback(self, event):
        """Reset playback to first frame"""
        # Stop animation if running
        if self.animation_running and self.animation is not None:
            self.animation.event_source.stop()
            self.animation_running = False
            self.play_button.label.set_text('Play')
        
        # Reset to first frame
        self.current_frame_idx = 0
        self.slider_frame.set_val(0)
        
        # No need to call update_frame as the slider callback will do it
        
        self.fig.canvas.draw_idle()

    def run(self):
        """Run the playback interface"""
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, bottom=0.35, top=0.95)
        plt.show()


# Run the simulation
if __name__ == "__main__":
    # Check for command line arguments
    import sys
    import argparse

    # Create parameter directories if they don't exist
    for directory in ['parameters', 'recordings']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Camera POV Simulator')
    parser.add_argument('--replay', help='Path to recording file to replay')
    parser.add_argument('--interval', type=int, default=50, help='Animation interval in ms (higher values reduce CPU usage)')
    parser.add_argument('--load-params', help='Path to parameter file to load')

    args = parser.parse_args()

    if args.replay:
        # Replay mode
        try:
            data = CameraPOVSimulator.load_recording(args.replay)
            player = RecordingPlayer(data)
            player.run()
        except Exception as e:
            print(f"Error loading recording: {e}")
    else:
        # Normal simulation mode
        simulator = CameraPOVSimulator()
        
        # Load parameters if specified
        if args.load_params:
            try:
                with open(args.load_params, 'r') as f:
                    params = json.load(f)
                
                # Update camera settings
                cam_params = params['camera']
                simulator.camera['position'][0] = cam_params['position_x']
                simulator.camera['position'][1] = cam_params['position_y']
                simulator.camera['view_angle'] = cam_params['view_angle']
                simulator.camera['range'] = cam_params['range']
                simulator.camera['rotation'] = cam_params['rotation']
                
                # Update car settings
                car_params = params['car']
                simulator.car['speed'] = car_params['speed']
                simulator.car['width'] = car_params['width']
                simulator.car['height'] = car_params['height']
                
                # Update sliders to match loaded parameters
                simulator.slider_cam_x.set_val(simulator.camera['position'][0])
                simulator.slider_cam_y.set_val(simulator.camera['position'][1])
                simulator.slider_cam_fov.set_val(simulator.camera['view_angle'])
                simulator.slider_cam_range.set_val(simulator.camera['range'])
                simulator.slider_cam_rot.set_val(simulator.camera['rotation'])
                simulator.slider_car_speed.set_val(simulator.car['speed'])
                simulator.slider_car_width.set_val(simulator.car['width'])
                simulator.slider_car_height.set_val(simulator.car['height'])
                
                print(f"Loaded parameters from {args.load_params}")
            except Exception as e:
                print(f"Error loading parameters: {e}")
        
        # Run the simulation
        simulator.run_simulation()

        
        