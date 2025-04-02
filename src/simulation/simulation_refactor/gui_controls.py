import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, AxesWidget
import matplotlib.patches as patches
import os
import json
from datetime import datetime
import pickle

# Simple dropdown selector widget
class Dropdown(AxesWidget):
    def __init__(self, ax, options, active=0, color='0.95', hovercolor='0.7'):
        """
        Create a simple dropdown selector widget
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to place the dropdown on
        options : list
            List of string options to display
        active : int
            Index of initially selected option
        color : str
            Background color
        hovercolor : str
            Background color when mouse hovers
        """
        AxesWidget.__init__(self, ax)
        self.options = options
        self.active = active
        self.color = color
        self.hovercolor = hovercolor
        self.observers = {}
        
        # Set up the axes
        ax.set_navigate(False)
        ax.set_frame_on(True)
        ax.axis('off')
        
        # Initialize the dropdown menu
        self.draw_menu()
        self.clicked = False
        self.expanded = False
        
        # Connect events
        self.connect_event('button_press_event', self._click)
        
    def draw_menu(self):
        """Draw the dropdown menu"""
        self.ax.clear()
        self.ax.axis('off')
        
        # Draw the selected option
        self.ax.text(0.05, 0.5, self.options[self.active], va='center', ha='left')
        
        # Draw a dropdown arrow
        self.ax.text(0.9, 0.5, '▼', va='center', ha='right')
        
        # Draw a border
        rect = patches.Rectangle((0, 0), 1, 1, fill=True, 
                                 facecolor=self.color, edgecolor='black', 
                                 linewidth=1, alpha=0.5)
        self.ax.add_patch(rect)
        
        self.ax.figure.canvas.draw()
    
    def _click(self, event):
        """Handle click events"""
        if event.inaxes != self.ax or not self.ax.get_visible():
            return
        
        if not self.expanded:
            # Show dropdown options
            self.show_options()
        else:
            # Check if click is on an option
            option_index = int((1 - event.ydata) * len(self.options))
            if 0 <= option_index < len(self.options):
                self.active = option_index
                self.draw_menu()
                self.expanded = False
                
                # Call observers
                for cid, func in self.observers.items():
                    func(self.options[self.active])
    
    def show_options(self):
        """Show the dropdown options"""
        self.expanded = True
        fig = self.ax.figure
        
        # Calculate position of dropdown menu
        x0, y0, w, h = self.ax.get_position().bounds
        
        # Create temporary axes for dropdown
        height = min(0.3, len(self.options) * 0.04)  # Limit height for many options
        ax_options = fig.add_axes([x0, y0 - height, w, height])
        ax_options.set_navigate(False)
        ax_options.set_frame_on(True)
        ax_options.set_axis_off()
        
        # Draw options
        for i, option in enumerate(self.options):
            # Position from bottom to top
            y_pos = 1 - (i + 1) / len(self.options) * 0.9 - 0.05
            ax_options.text(0.05, y_pos, option, va='center', ha='left')
            
            # Draw separator lines
            if i < len(self.options) - 1:
                ax_options.axhline(y=1 - (i + 1) / len(self.options), color='gray', linestyle='-', linewidth=0.5)
        
        # Draw background
        rect = patches.Rectangle((0, 0), 1, 1, fill=True, 
                               facecolor=self.color, edgecolor='black', 
                               linewidth=1, alpha=0.5)
        ax_options.add_patch(rect)
        
        # Store reference to options axes
        self.ax_options = ax_options
        
        # Add event to close dropdown when clicking elsewhere
        self.click_cid = fig.canvas.mpl_connect('button_press_event', self.close_options)
        
        fig.canvas.draw()
    
    def close_options(self, event):
        """Close the dropdown options"""
        if hasattr(self, 'ax_options'):
            if event.inaxes != self.ax_options:
                # Remove dropdown axes
                self.ax.figure.delaxes(self.ax_options)
                delattr(self, 'ax_options')
                self.expanded = False
                
                # Disconnect click event
                if hasattr(self, 'click_cid'):
                    self.ax.figure.canvas.mpl_disconnect(self.click_cid)
                
                self.ax.figure.canvas.draw()
    
    def on_changed(self, func):
        """
        Connect a callback function to dropdown change event
        
        Parameters:
        -----------
        func : callable
            Function to call when dropdown selection changes
            Will be called with the selected option text
            
        Returns:
        --------
        cid : int
            Connection id for disconnecting
        """
        cid = max(self.observers.keys()) + 1 if self.observers else 0
        self.observers[cid] = func
        return cid

class CameraPOV:
    def __init__(self, ax, camera_params):
        """
        Initialize the camera point-of-view visualization
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw the camera POV on
        camera_params : dict
            Dictionary containing camera parameters
        """
        self.ax = ax
        self.camera_params = camera_params
        
        # Configure camera POV view
        self.ax.set_xlim(0, self.camera_params['sensor_width'])
        self.ax.set_ylim(0, self.camera_params['sensor_height'])
        self.ax.set_title("Camera POV")
        self.ax.set_facecolor('black')  # Black background for camera view
        
        # Dictionary to store car visualizations in POV
        self.car_povs = {}
        self.bbox_povs = {}
        
        # Add info text elements
        self.info_text = self.ax.text(10, 30, "", color='white', fontsize=10)
        self.distance_text = self.ax.text(10, 10, "", color='white', fontsize=10)
        
        # Car priority for display (lower z-order = appears on top)
        self.car_priorities = {}
    
    def update_camera_params(self, camera_params):
        """
        Update camera parameters
        
        Parameters:
        -----------
        camera_params : dict
            Dictionary containing new camera parameters
        """
        self.camera_params = camera_params
    
    def set_car_priority(self, car_id, priority):
        """
        Set the display priority for a car
        
        Parameters:
        -----------
        car_id : int
            ID of the car
        priority : int
            Priority value (lower = appears on top)
        """
        self.car_priorities[car_id] = priority
        
        # Update z-orders of existing car POVs
        if car_id in self.car_povs:
            self.car_povs[car_id].set_zorder(-priority)  # Negative because lower priority should be on top
            if car_id in self.bbox_povs:
                self.bbox_povs[car_id].set_zorder(-priority)
    
    def project_car(self, car, in_camera_view, distance=None):
        """
        Project car from 3D world to 2D camera view with perspective
        
        Parameters:
        -----------
        car : Car object
            The car object being visualized
        in_camera_view : bool
            Whether the car is in the camera's field of view
        distance : float or None
            Distance from camera to car, if known
            
        Returns:
        --------
        bbox_data : list or None
            Bounding box data [x, y, width, height] if car is detected, None otherwise
        """
        car_id = car.id
        
        # Create car POV visualization if it doesn't exist
        if car_id not in self.car_povs:
            self.car_povs[car_id] = patches.Rectangle(
                (-100, -100),  # Off-screen
                0, 0,  # Will be updated
                angle=0,
                fc=car.color,  # Use car's color
                ec='black',
                visible=False
            )
            self.ax.add_patch(self.car_povs[car_id])
            
            # Create bounding box for this car
            self.bbox_povs[car_id] = patches.Rectangle(
                (-100, -100),  # Off-screen
                0, 0,  # Will be updated
                fill=False,
                ec='green',
                linewidth=2,
                linestyle='--',
                visible=False
            )
            self.ax.add_patch(self.bbox_povs[car_id])
            
            # Set priority (default to car_id if not explicitly set)
            if car_id not in self.car_priorities:
                self.car_priorities[car_id] = car_id
                
            # Set z-order based on priority
            self.car_povs[car_id].set_zorder(-self.car_priorities[car_id])
            self.bbox_povs[car_id].set_zorder(-self.car_priorities[car_id])
        
        if not in_camera_view:
            # Car is not visible to camera
            self.car_povs[car_id].set_visible(False)
            self.bbox_povs[car_id].set_visible(False)
            
            # Only update info text if this is the highest priority car
            if self.is_highest_priority_car(car_id):
                self.info_text.set_text(f"Car {car_id} not in view")
                self.distance_text.set_text("")
            
            return None
        
        # Calculate distance if not provided
        if distance is None:
            camera_pos = self.camera_params['position']
            car_pos = car.position  # Access as attribute, not as dictionary
            distance = np.linalg.norm(car_pos - camera_pos)
        
        # Calculate angle to car relative to camera direction
        camera_pos = self.camera_params['position']
        car_pos = car.position  # Access as attribute, not as dictionary
        to_car = car_pos - camera_pos
        
        angle_to_car = np.arctan2(to_car[1], to_car[0])
        cam_angle = np.radians(self.camera_params['rotation'])
        
        # Normalize angle_diff to be between -pi and pi
        angle_diff = angle_to_car - cam_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Calculate horizontal position in image based on angle
        # Map from [-fov/2, fov/2] to [0, sensor_width]
        half_fov = np.radians(self.camera_params['view_angle'] / 2)
        normalized_x = 1.0 - (angle_diff + half_fov) / (2 * half_fov)  # Flipped direction
        screen_x = normalized_x * self.camera_params['sensor_width']
        
        # Scale size based on distance (perspective effect)
        # Size is inversely proportional to distance
        scale_factor = self.camera_params['focal_length'] / max(distance, 1)
        
        # Calculate projected size of car
        projected_width = car.width * scale_factor  # Access as attribute
        projected_height = car.height * scale_factor  # Access as attribute
        
        # Center car vertically in camera view (simplified)
        # In a real camera, vertical position would depend on 3D coordinates
        screen_y = self.camera_params['sensor_height'] / 2
        
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
            self.bbox_povs[car_id].set_visible(False)
        else:
            detected = True
            self.bbox_povs[car_id].set_visible(True)
            self.bbox_povs[car_id].set_xy((bbox_rect[0], bbox_rect[1]))
            self.bbox_povs[car_id].set_width(bbox_rect[2])
            self.bbox_povs[car_id].set_height(bbox_rect[3])
        
        # Update car visualization in camera view
        self.car_povs[car_id].set_visible(True)
        self.car_povs[car_id].set_xy((car_rect[0], car_rect[1]))
        self.car_povs[car_id].set_width(car_rect[2])
        self.car_povs[car_id].set_height(car_rect[3])
        
        # Set car angle in camera view (simplified)
        # In reality, this would involve 3D rotation and projection
        self.car_povs[car_id].angle = 0
        
        # Only update info text if this is the highest priority car
        if self.is_highest_priority_car(car_id):
            # Update info text
            self.info_text.set_text(f"Car {car_id} in view" + (" (Detected)" if detected else " (Not Detected)"))
            self.distance_text.set_text(f"Distance: {distance:.1f} units")
        
        # Return the bounding box data if detected
        return bbox_rect if detected else None
    
    def is_highest_priority_car(self, car_id):
        """Check if the car has the highest priority (lowest value)"""
        if not self.car_priorities:
            return True
            
        # Get visible cars
        visible_cars = [cid for cid, pov in self.car_povs.items() if pov.get_visible()]
        
        if not visible_cars:
            return True
            
        # Get priorities of visible cars
        visible_priorities = [self.car_priorities.get(cid, cid) for cid in visible_cars]
        
        # Check if this car has the highest priority (lowest value)
        return self.car_priorities.get(car_id, car_id) == min(visible_priorities)
    
    def get_visualization_elements(self):
        """
        Get all visualization elements for animation updates
        
        Returns:
        --------
        elements : list
            List of matplotlib artists that need to be updated
        """
        elements = [self.info_text, self.distance_text]
        elements.extend(self.car_povs.values())
        elements.extend(self.bbox_povs.values())
        return elements


class GUIControls:
    def __init__(self, ax, simulation):
        """
        Initialize GUI controls for simulation parameters
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to place controls on
        simulation : Simulation
            The simulation object to control
        """
        self.ax = ax
        self.sim = simulation
        
        # Turn off axis for controls area
        self.ax.axis('off')
        
        # Currently selected car
        self.selected_car_idx = 0
        
        # Setup controls
        self.setup_sliders()
        self.setup_buttons()
        self.setup_car_selector()
        
        # Recording state
        self.recording = False
        self.frames_data = []
        
        # Create recording directory if it doesn't exist
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
            
        # Create parameters directory if it doesn't exist
        if not os.path.exists('parameters'):
            os.makedirs('parameters')
    
    def setup_car_selector(self):
        """Set up car selection dropdown and priority controls"""
        # Get number of cars
        num_cars = len(self.sim.cars)
        
        # Create car selection options
        car_options = [f"Car {i}" for i in range(num_cars)]
        
        # Car selection dropdown
        self.ax_car_selector = plt.axes([0.45, 0.22, 0.15, 0.03])
        self.car_selector = Dropdown(
            self.ax_car_selector,
            car_options,
            active=0
        )
        self.car_selector.on_changed(self.select_car)
        
        # Car priority label
        self.ax_priority_label = plt.axes([0.45, 0.18, 0.08, 0.03])
        self.ax_priority_label.text(0.5, 0.5, "Car Priority:", ha='center', va='center')
        self.ax_priority_label.axis('off')
        
        # Priority up button
        self.ax_priority_up = plt.axes([0.54, 0.18, 0.03, 0.03])
        self.priority_up_button = Button(
            self.ax_priority_up, 
            '▲',
            color='lightgreen'
        )
        self.priority_up_button.on_clicked(self.increase_priority)
        
        # Priority down button
        self.ax_priority_down = plt.axes([0.57, 0.18, 0.03, 0.03])
        self.priority_down_button = Button(
            self.ax_priority_down, 
            '▼',
            color='lightcoral'
        )
        self.priority_down_button.on_clicked(self.decrease_priority)
    
    def select_car(self, label):
        """
        Handle car selection
        
        Parameters:
        -----------
        label : str
            Label of the selected car option
        """
        # Extract car index from label (e.g., "Car 0" -> 0)
        self.selected_car_idx = int(label.split()[1])
        
        # Update sliders to match selected car's parameters
        self.update_sliders_for_selected_car()
        
        self.sim.vis.fig.canvas.draw_idle()
    
    def increase_priority(self, event):
        """Increase priority of selected car (smaller number = higher priority)"""
        car_id = self.selected_car_idx
        current_priority = self.sim.vis.camera_pov.car_priorities.get(car_id, car_id)
        
        # Increase priority (lower the number)
        new_priority = max(0, current_priority - 1)
        self.sim.vis.camera_pov.set_car_priority(car_id, new_priority)
        self.sim.vis.fig.canvas.draw_idle()
    
    def decrease_priority(self, event):
        """Decrease priority of selected car (larger number = lower priority)"""
        car_id = self.selected_car_idx
        current_priority = self.sim.vis.camera_pov.car_priorities.get(car_id, car_id)
        
        # Decrease priority (increase the number)
        new_priority = current_priority + 1
        self.sim.vis.camera_pov.set_car_priority(car_id, new_priority)
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_sliders_for_selected_car(self):
        """Update slider values to match the selected car"""
        if self.selected_car_idx < len(self.sim.cars):
            car = self.sim.cars[self.selected_car_idx]
            
            # Update car speed slider
            self.slider_car_speed.set_val(car.speed)
    
    def setup_sliders(self):
        """Set up slider controls"""
        camera = self.sim.camera
        car = self.sim.cars[self.selected_car_idx]  # Currently selected car
        
        # Get figure to determine positions
        fig = self.ax.figure
        
        # Add sliders for camera parameters
        plt.subplots_adjust(bottom=0.35)
        
        # Camera X position slider
        self.ax_cam_x = plt.axes([0.1, 0.26, 0.25, 0.03])
        self.slider_cam_x = Slider(
            ax=self.ax_cam_x,
            label='Camera X',
            valmin=0,
            valmax=800,
            valinit=camera.position[0],
        )
        self.slider_cam_x.on_changed(self.update_camera_x)
        
        # Camera Y position slider
        self.ax_cam_y = plt.axes([0.1, 0.22, 0.25, 0.03])
        self.slider_cam_y = Slider(
            ax=self.ax_cam_y,
            label='Camera Y',
            valmin=0,
            valmax=500,
            valinit=camera.position[1],
        )
        self.slider_cam_y.on_changed(self.update_camera_y)
        
        # Camera FOV slider
        self.ax_cam_fov = plt.axes([0.1, 0.18, 0.25, 0.03])
        self.slider_cam_fov = Slider(
            ax=self.ax_cam_fov,
            label='FOV (°)',
            valmin=10,
            valmax=180,
            valinit=camera.view_angle,
        )
        self.slider_cam_fov.on_changed(self.update_camera_fov)
        
        # Camera rotation slider
        self.ax_cam_rot = plt.axes([0.1, 0.14, 0.25, 0.03])
        self.slider_cam_rot = Slider(
            ax=self.ax_cam_rot,
            label='Rotation (°)',
            valmin=0,
            valmax=360,
            valinit=camera.rotation,
        )
        self.slider_cam_rot.on_changed(self.update_camera_rotation)
        
        # Camera range slider
        self.ax_cam_range = plt.axes([0.1, 0.1, 0.25, 0.03])
        self.slider_cam_range = Slider(
            ax=self.ax_cam_range,
            label='Range',
            valmin=100,
            valmax=800,
            valinit=camera.range,
        )
        self.slider_cam_range.on_changed(self.update_camera_range)
        
        # Car speed slider
        self.ax_car_speed = plt.axes([0.45, 0.14, 0.15, 0.03])
        self.slider_car_speed = Slider(
            ax=self.ax_car_speed,
            label='Car Speed',
            valmin=0.5,
            valmax=5.0,
            valinit=car.speed,
        )
        self.slider_car_speed.on_changed(self.update_car_speed)
    
    def setup_buttons(self):
        """Set up button controls"""
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
        
        # Add recording indicator (initially empty)
        self.recording_text = self.sim.vis.ax.text(650, 450, "", color='red', fontsize=10, fontweight='bold')
    
    def update_camera_x(self, val):
        """Update camera X position"""
        self.sim.change_camera_position(position=(val, self.sim.camera.position[1]))
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_camera_y(self, val):
        """Update camera Y position"""
        self.sim.change_camera_position(position=(self.sim.camera.position[0], val))
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_camera_fov(self, val):
        """Update camera field of view"""
        self.sim.camera.view_angle = val
        # Update camera params for POV
        self.sim.camera_params['view_angle'] = val
        self.sim.vis.camera_pov.update_camera_params(self.sim.camera_params)
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_camera_rotation(self, val):
        """Update camera rotation"""
        self.sim.change_camera_position(rotation=val)
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_camera_range(self, val):
        """Update camera range"""
        self.sim.camera.range = val
        # Update camera params for POV
        self.sim.camera_params['range'] = val
        self.sim.vis.camera_pov.update_camera_params(self.sim.camera_params)
        self.sim.vis.fig.canvas.draw_idle()
    
    def update_car_speed(self, val):
        """Update selected car speed"""
        self.sim.cars[self.selected_car_idx].speed = val
        self.sim.vis.fig.canvas.draw_idle()
    
    def toggle_animation(self, event):
        """Toggle animation play/pause"""
        if self.sim.animation_running:
            if self.sim.animation is not None:
                self.sim.animation.event_source.stop()
            self.sim.animation_running = False
            self.play_button.label.set_text('Play')
        else:
            if self.sim.animation is None:
                self.sim.run()
            else:
                self.sim.animation.event_source.start()
            self.sim.animation_running = True
            self.play_button.label.set_text('Pause')
        self.sim.vis.fig.canvas.draw_idle()
    
    def reset_simulation(self, event):
        """Reset simulation to initial state"""
        # Stop animation if running
        if self.sim.animation_running and self.sim.animation is not None:
            self.sim.animation.event_source.stop()
            self.sim.animation_running = False
            self.play_button.label.set_text('Play')
        
        # Reset car positions
        for car in self.sim.cars:
            car.track_position = 0
            car.position, car.angle = car.track.get_position_at(0)
            car.update_visualization()
        
        # Reset frame counter
        self.sim.frame_num = 0
        
        self.sim.vis.fig.canvas.draw_idle()
    
    def save_parameters(self, event):
        """Save current parameter settings"""
        # Get camera parameters
        camera = self.sim.camera
        
        # Parameters object - includes all cars
        parameters = {
            'camera': {
                'position_x': float(camera.position[0]),
                'position_y': float(camera.position[1]),
                'view_angle': float(camera.view_angle),
                'range': float(camera.range),
                'rotation': float(camera.rotation),
                'focal_length': float(self.sim.camera_params['focal_length']),
                'sensor_width': float(self.sim.camera_params['sensor_width']),
                'sensor_height': float(self.sim.camera_params['sensor_height']),
            },
            'cars': []
        }
        
        # Add parameters for all cars
        for car in self.sim.cars:
            car_params = {
                'id': car.id,
                'speed': float(car.speed),
                'width': float(car.width),
                'height': float(car.height),
                'color': car.color,
                'start_position': float(car.track_position)
            }
            parameters['cars'].append(car_params)
        
        # Add car priorities
        if hasattr(self.sim.vis, 'camera_pov') and hasattr(self.sim.vis.camera_pov, 'car_priorities'):
            parameters['priorities'] = {str(k): v for k, v in self.sim.vis.camera_pov.car_priorities.items()}
        
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
        
        self.sim.vis.fig.canvas.draw_idle()
    
    def save_recording(self):
        """Save recorded simulation data"""
        # Get camera parameters
        camera = self.sim.camera
        
        # Create parameter object - includes all cars
        parameters = {
            'camera': {
                'position_x': float(camera.position[0]),
                'position_y': float(camera.position[1]),
                'view_angle': float(camera.view_angle),
                'range': float(camera.range),
                'rotation': float(camera.rotation),
                'focal_length': float(self.sim.camera_params['focal_length']),
                'sensor_width': float(self.sim.camera_params['sensor_width']),
                'sensor_height': float(self.sim.camera_params['sensor_height']),
            },
            'cars': []
        }
        
        # Add parameters for all cars
        for car in self.sim.cars:
            car_params = {
                'id': car.id,
                'speed': float(car.speed),
                'width': float(car.width),
                'height': float(car.height),
                'color': car.color
            }
            parameters['cars'].append(car_params)
            
        # Add car priorities
        if hasattr(self.sim.vis, 'camera_pov') and hasattr(self.sim.vis.camera_pov, 'car_priorities'):
            parameters['priorities'] = {str(k): v for k, v in self.sim.vis.camera_pov.car_priorities.items()}
        
        # Add frames data
        parameters['frames'] = self.frames_data
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/simulation_{timestamp}.pkl"
        
        # Save data
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
        
        print(f"Recording saved to {filename}")
    
    def record_frame(self, frame_num):
        """Record current frame data"""
        if not self.recording:
            return
        
        # Frame data will include all cars
        frame_data = {
            'frame': frame_num,
            'cars': {}
        }
        
        # Record data for each car
        for car in self.sim.cars:
            car_data = {
                'position': car.position.copy(),
                'angle': car.angle,
                'in_camera_view': car.in_camera_view,
                'bbox': car.bounding_box
            }
            frame_data['cars'][car.id] = car_data
        
        # Add to frames data
        self.frames_data.append(frame_data)





