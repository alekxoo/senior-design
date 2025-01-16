# backend/video_processor.py

import torch
import cv2
import numpy as np
from typing import Tuple, List, Optional

class VideoProcessor:
    def __init__(self, model_path: str = "yolov5s.pt"):
        """Initialize the video processor with YOLOv5 model.
        
        Args:
            model_path (str): Path to YOLOv5 model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLOv5 model
        print("before load model")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.eval()
        self.model.to(self.device)
        # Set model parameters
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.classes = [2]  # Only detect cars (class 2 in COCO dataset)

    def process_frame(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """Process a single frame and return detections.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            Tuple containing:
            - List of detections, each with keys: 'bbox', 'confidence', 'class'
            - Annotated frame with detections drawn
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(frame_rgb)
        print("After YOLO inference")
        
        # Process results
        detections = []
        for det in results.xyxy[0]:  # det: (x1, y1, x2, y2, confidence, class)
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class': int(cls)
            })
        
        print("Post Process results")

        # Draw detections on frame
        annotated_frame = frame.copy()
        for det in detections:
            bbox = det['bbox']
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        (0, 255, 0), 2)
            # Add confidence score
            label = f"{det['confidence']:.2f}"
            cv2.putText(annotated_frame, 
                       label, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)
            
        print("Draw annotated frames")

        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        return detections, annotated_frame
    

