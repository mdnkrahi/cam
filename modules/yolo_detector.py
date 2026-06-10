import cv2
import numpy as np
from ultralytics import YOLO
from config import get_config
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self):
        """Initialize YOLO face detector"""
        self.config = get_config()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model for face detection"""
        try:
            # Use yolov8n for face detection (nano version - fast and lightweight)
            self.model = YOLO('yolov8n-face.pt')
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def detect_faces(self, frame, conf_threshold=None):
        """
        Detect faces in a frame
        
        Args:
            frame: Input image/frame
            conf_threshold: Confidence threshold (uses config default if None)
        
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        if conf_threshold is None:
            conf_threshold = self.config.CONFIDENCE_THRESHOLD
        
        try:
            # Run inference
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract face region
                    face_region = frame[y1:y2, x1:x2]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'face_region': face_region,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
            
            logger.debug(f"Detected {len(detections)} faces")
            return detections
        
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []
    
    def draw_detections(self, frame, detections, labels=None):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            labels: Optional labels for each detection
        
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for unknown faces
            thickness = 2
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"Face {i+1}: {conf:.2f}"
            if labels and i < len(labels):
                label = f"{labels[i]}: {conf:.2f}"
            
            cv2.putText(output_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output_frame
    
    def filter_detections(self, detections, min_size=20):
        """
        Filter out small or low-confidence detections
        
        Args:
            detections: List of detections
            min_size: Minimum face size in pixels
        
        Returns:
            Filtered detections
        """
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Check size and confidence
            if width >= min_size and height >= min_size:
                if detection['confidence'] >= self.config.CONFIDENCE_THRESHOLD:
                    filtered.append(detection)
        
        return filtered
    
    def get_face_roi(self, frame, detection, padding=10):
        """
        Extract face region of interest with optional padding
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            padding: Padding around face in pixels
        
        Returns:
            Face ROI
        """
        x1, y1, x2, y2 = detection['bbox']
        h, w = frame.shape[:2]
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return frame[y1:y2, x1:x2]