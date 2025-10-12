#!/usr/bin/env python3
"""
YOLO-based Activity Detection Script
Combines YOLO person detection with activity classification using the 3-class dataset.

This script:
1. Uses YOLO to detect persons in video frames
2. Crops detected person regions
3. Classifies activities using the trained 3-class model
4. Outputs results with bounding boxes and activity labels

Author: Generated for YOLO + Activity Detection integration
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import argparse
from pathlib import Path
import json
from ultralytics import YOLO
import time

class Simple3DCNN(nn.Module):
    """3D CNN for activity classification (same as training script)"""
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        
        self.conv3d1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        self.fc = nn.Linear(128 * 7 * 7, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (batch, channels, frames, height, width)
        x = self.relu(self.conv3d1(x))
        x = self.pool1(x)
        x = self.relu(self.conv3d2(x))
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class YOLOActivityDetector:
    def __init__(self, yolo_model_path="yolov8n.pt", activity_model_path=None, device='cuda'):
        """
        Initialize YOLO + Activity Detection pipeline
        
        Args:
            yolo_model_path: Path to YOLO model (will download if not exists)
            activity_model_path: Path to trained activity classification model
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO model for person detection
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load activity classification model
        if activity_model_path and Path(activity_model_path).exists():
            print("Loading activity classification model...")
            self.activity_model = Simple3DCNN(num_classes=3).to(self.device)
            checkpoint = torch.load(activity_model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.activity_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.activity_model.load_state_dict(checkpoint)
            self.activity_model.eval()
        else:
            print("No activity model provided, will only do person detection")
            self.activity_model = None
        
        # Activity class labels
        self.activity_labels = {
            0: "Normal",
            1: "Abnormal Physical", 
            2: "Abnormal Biological"
        }
        
        # Colors for different activities
        self.activity_colors = {
            0: (0, 255, 0),    # Green for normal
            1: (0, 0, 255),    # Red for abnormal physical
            2: (255, 0, 255)   # Magenta for abnormal biological
        }
        
        # Transform for activity classification
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Frame buffer for temporal analysis
        self.frame_buffer = []
        self.max_frames = 16
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLO"""
        results = self.yolo_model(frame, classes=[0], verbose=False)  # Class 0 is 'person'
        
        persons = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                if confidence > 0.5:  # Confidence threshold
                    persons.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    })
        
        return persons
    
    def crop_person_region(self, frame, bbox, margin=0.1):
        """Crop person region from frame with some margin"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        return frame[y1:y2, x1:x2]
    
    def classify_activity(self, person_frames):
        """Classify activity from sequence of person crops"""
        if self.activity_model is None:
            return 0, 0.0  # Default to normal if no model
        
        if len(person_frames) < self.max_frames:
            # Pad with last frame
            while len(person_frames) < self.max_frames:
                if person_frames:
                    person_frames.append(person_frames[-1])
                else:
                    person_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Take only max_frames
        person_frames = person_frames[:self.max_frames]
        
        # Transform frames
        transformed_frames = []
        for frame in person_frames:
            if frame.size > 0:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            else:
                # Create black frame if crop is empty
                black_frame = torch.zeros(3, 224, 224)
                transformed_frames.append(black_frame)
        
        # Stack frames and add batch dimension
        video_tensor = torch.stack(transformed_frames).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.activity_model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence
    
    def process_video(self, video_path, output_path=None, show_video=True):
        """Process video file and detect activities"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        person_frame_buffers = {}  # Track frames for each person
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect persons
            persons = self.detect_persons(frame)
            
            # Process each detected person
            for i, person in enumerate(persons):
                bbox = person['bbox']
                confidence = person['confidence']
                
                # Crop person region
                person_crop = self.crop_person_region(frame, bbox)
                
                # Resize crop for consistency
                if person_crop.size > 0:
                    person_crop = cv2.resize(person_crop, (224, 224))
                
                # Maintain frame buffer for this person (simple tracking by position)
                person_id = f"person_{i}"
                if person_id not in person_frame_buffers:
                    person_frame_buffers[person_id] = []
                
                person_frame_buffers[person_id].append(person_crop)
                
                # Keep only recent frames
                if len(person_frame_buffers[person_id]) > self.max_frames:
                    person_frame_buffers[person_id] = person_frame_buffers[person_id][-self.max_frames:]
                
                # Classify activity if we have enough frames
                if len(person_frame_buffers[person_id]) >= 8:  # Minimum frames for classification
                    activity_class, activity_confidence = self.classify_activity(person_frame_buffers[person_id])
                    activity_label = self.activity_labels[activity_class]
                    color = self.activity_colors[activity_class]
                else:
                    activity_label = "Analyzing..."
                    activity_confidence = 0.0
                    color = (128, 128, 128)  # Gray
                
                # Draw bounding box and label
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                label_text = f"Person: {confidence:.2f}"
                activity_text = f"{activity_label}: {activity_confidence:.2f}"
                
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, activity_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            # Show frame
            if show_video:
                cv2.imshow('YOLO Activity Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing completed! Processed {frame_count} frames")
    
    def process_webcam(self):
        """Process live webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam processing. Press 'q' to quit.")
        
        person_frame_buffers = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons
            persons = self.detect_persons(frame)
            
            # Process each detected person
            for i, person in enumerate(persons):
                bbox = person['bbox']
                confidence = person['confidence']
                
                # Crop person region
                person_crop = self.crop_person_region(frame, bbox)
                
                # Resize crop for consistency
                if person_crop.size > 0:
                    person_crop = cv2.resize(person_crop, (224, 224))
                
                # Maintain frame buffer for this person
                person_id = f"person_{i}"
                if person_id not in person_frame_buffers:
                    person_frame_buffers[person_id] = []
                
                person_frame_buffers[person_id].append(person_crop)
                
                # Keep only recent frames
                if len(person_frame_buffers[person_id]) > self.max_frames:
                    person_frame_buffers[person_id] = person_frame_buffers[person_id][-self.max_frames:]
                
                # Classify activity if we have enough frames
                if len(person_frame_buffers[person_id]) >= 8:
                    activity_class, activity_confidence = self.classify_activity(person_frame_buffers[person_id])
                    activity_label = self.activity_labels[activity_class]
                    color = self.activity_colors[activity_class]
                else:
                    activity_label = "Analyzing..."
                    activity_confidence = 0.0
                    color = (128, 128, 128)
                
                # Draw bounding box and label
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                label_text = f"Person: {confidence:.2f}"
                activity_text = f"{activity_label}: {activity_confidence:.2f}"
                
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, activity_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show frame
            cv2.imshow('YOLO Activity Detection - Live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO-based Activity Detection')
    parser.add_argument('--video', '-v', help='Path to input video file')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam input')
    parser.add_argument('--output', '-o', help='Path to output video file')
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--activity-model', help='Activity classification model path')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--no-display', action='store_true', help='Do not display video')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOActivityDetector(
        yolo_model_path=args.yolo_model,
        activity_model_path=args.activity_model,
        device=args.device
    )
    
    if args.webcam:
        detector.process_webcam()
    elif args.video:
        detector.process_video(
            video_path=args.video,
            output_path=args.output,
            show_video=not args.no_display
        )
    else:
        print("Please specify either --video or --webcam")

if __name__ == "__main__":
    main()
