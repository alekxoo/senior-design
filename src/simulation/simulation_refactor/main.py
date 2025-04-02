#!/usr/bin/env python3
"""
Main entry point for the track simulation.
Run this file to start the simulation.
"""

import os
import argparse
import json
from simulation import Simulation

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Camera POV Simulator')
    parser.add_argument('--interval', type=int, default=50, 
                        help='Animation interval in ms (higher values reduce CPU usage)')
    parser.add_argument('--load-params', help='Path to parameter file to load')
    
    args = parser.parse_args()
    
    # Create parameters directory if it doesn't exist
    for directory in ['parameters', 'recordings']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Create and initialize simulation
    sim = Simulation()
    
    # Load parameters if specified
    if args.load_params and os.path.exists(args.load_params):
        try:
            with open(args.load_params, 'r') as f:
                params = json.load(f)
            
            # Update camera settings
            cam_params = params['camera']
            sim.camera.position[0] = cam_params['position_x']
            sim.camera.position[1] = cam_params['position_y']
            sim.camera.view_angle = cam_params['view_angle']
            sim.camera.range = cam_params['range']
            sim.camera.rotation = cam_params['rotation']
            
            # Update camera parameters for POV
            sim.camera_params['position'] = sim.camera.position
            sim.camera_params['rotation'] = sim.camera.rotation
            sim.camera_params['view_angle'] = sim.camera.view_angle
            sim.camera_params['range'] = sim.camera.range
            
            # Update car settings for the first car
            car_params = params['car']
            if sim.cars:
                sim.cars[0].speed = car_params['speed']
                sim.cars[0].width = car_params['width']
                sim.cars[0].height = car_params['height']
            
            # Update sliders to match loaded parameters
            gui = sim.vis.gui_controls
            if gui:
                gui.slider_cam_x.set_val(sim.camera.position[0])
                gui.slider_cam_y.set_val(sim.camera.position[1])
                gui.slider_cam_fov.set_val(sim.camera.view_angle)
                gui.slider_cam_range.set_val(sim.camera.range)
                gui.slider_cam_rot.set_val(sim.camera.rotation)
                if sim.cars:
                    gui.slider_car_speed.set_val(sim.cars[0].speed)
                    gui.slider_car_width.set_val(sim.cars[0].width)
                    gui.slider_car_height.set_val(sim.cars[0].height)
            
            print(f"Loaded parameters from {args.load_params}")
        except Exception as e:
            print(f"Error loading parameters: {e}")
    
    # Add a second car (optional)
    sim.add_car(id=1, speed=1.2, start_position=50, color='blue')
    
    # Set noise model (optional)
    sim.set_noise_model('gaussian', {'mean': 0, 'std': 2})
    
    # Run the simulation - the GUI controls will handle play/pause/reset
    sim.run(interval=args.interval)
    
if __name__ == "__main__":
    main()