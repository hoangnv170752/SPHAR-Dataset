#!/usr/bin/env python3
"""
Integrated Person Detection + Action Recognition Testing
- YOLO for person detection with bounding boxes
- Action recognition only on detected persons
- Real-time performance metrics
- Better accuracy by focusing on persons
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from collections import deque
import time
from ultralytics import YOLO

# Import action recognition models
try:
    from action_recognition_trainer import SlowFastActionModel
except ImportError:
    SlowFastActionModel = None

try:
    from image_action_trainer import OptimizedSlowFastModel
except ImportError:
    OptimizedSlowFastModel = None

class IntegratedPersonActionInference:
    """Integrated person detection and action recognition"""
    
    def __init__(self, action_model_path, class_mapping_path, yolo_model_path=None, 
                 sequence_length=8, confidence_threshold=0.5, person_conf_threshold=0.3):
        
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.person_conf_threshold = person_conf_threshold
        
        # Load YOLO for person detection
        if yolo_model_path is None:
            yolo_model_path = Path(__file__).parent.parent / 'models' / 'yolo11n.pt'
        
        self.yolo_model = YOLO(str(yolo_model_path))
        print(f"✅ YOLO model loaded: {yolo_model_path}")
        
        # Load action recognition class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Load action recognition model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load checkpoint first
        checkpoint = torch.load(action_model_path, map_location=self.device)
        
        # Try OptimizedSlowFastModel first
        if OptimizedSlowFastModel is not None:
            try:
                self.action_model = OptimizedSlowFastModel(self.num_classes, sequence_length)
                self.action_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded OptimizedSlowFastModel")
            except Exception as e:
                print(f"⚠️ Failed to load OptimizedSlowFastModel: {e}")
                self.action_model = None
        
        # Fallback to original model
        if self.action_model is None and SlowFastActionModel is not None:
            try:
                self.action_model = SlowFastActionModel(self.num_classes, sequence_length)
                self.action_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded SlowFastActionModel")
            except Exception as e:
                print(f"❌ Failed to load SlowFastActionModel: {e}")
                raise e
        
        if self.action_model is None:
            raise RuntimeError("Could not load any action model")
        
        self.action_model.to(self.device)
        self.action_model.eval()
        
        # Frame buffers for each detected person (track by ID)
        self.person_buffers = {}
        self.person_actions = {}
        self.person_confidences = {}
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.action_times = deque(maxlen=30)
        
        print(f"✅ Integrated system loaded")
        print(f"📊 Action classes: {list(self.class_mapping.keys())}")
        print(f"⚡ Device: {self.device}")
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLO"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person (class 0 in COCO)
                    if int(box.cls) == 0 and float(box.conf) > self.person_conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        
                        persons.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence
                        })
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        return persons
    
    def extract_person_crop(self, frame, bbox):
        """Extract and preprocess person crop for action recognition"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        padding = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract crop
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return np.zeros((224, 224, 3), dtype=np.float32)
        
        # Resize to standard size
        person_crop = cv2.resize(person_crop, (224, 224))
        
        # Normalize
        person_crop = person_crop.astype(np.float32) / 255.0
        
        return person_crop
    
    def predict_action_for_person(self, person_id, person_crop):
        """Predict action for a specific person"""
        start_time = time.time()
        
        # Initialize buffer for new person
        if person_id not in self.person_buffers:
            self.person_buffers[person_id] = deque(maxlen=self.sequence_length)
            self.person_actions[person_id] = "normal"
            self.person_confidences[person_id] = 0.0
        
        # Add frame to person's buffer
        self.person_buffers[person_id].append(person_crop)
        
        # Predict if buffer is full
        if len(self.person_buffers[person_id]) == self.sequence_length:
            frames = np.array(list(self.person_buffers[person_id]))
            
            # Prepare tensor
            frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.action_model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.idx_to_class[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Update person's action if confidence is high enough
                if confidence_score > self.confidence_threshold:
                    self.person_actions[person_id] = predicted_class
                    self.person_confidences[person_id] = confidence_score
        
        action_time = time.time() - start_time
        self.action_times.append(action_time)
        
        return self.person_actions[person_id], self.person_confidences[person_id]
    
    def get_action_color(self, action):
        """Get color for action visualization"""
        colors = {
            'fall': (0, 0, 255),      # Red - Emergency
            'hitting': (0, 100, 255), # Orange - Violence
            'running': (0, 255, 255), # Yellow - Alert
            'warning': (0, 165, 255), # Orange - Warning
            'normal': (0, 255, 0),    # Green - Normal
            'unknown': (128, 128, 128) # Gray - Unknown
        }
        return colors.get(action, (255, 255, 255))
    
    def get_action_priority(self, action):
        """Get priority level for action"""
        priorities = {
            'fall': 3,      # Highest priority - Emergency
            'hitting': 3,   # Highest priority - Violence
            'running': 2,   # Medium priority - Alert
            'warning': 1,   # Low priority - Suspicious
            'normal': 0,    # No priority - Normal
            'unknown': 0    # No priority
        }
        return priorities.get(action, 0)
    
    def process_frame(self, frame):
        """Process frame with integrated person detection and action recognition"""
        frame_start_time = time.time()
        
        # Detect persons
        persons = self.detect_persons(frame)
        
        # Process each detected person
        annotated_frame = frame.copy()
        
        for i, person in enumerate(persons):
            bbox = person['bbox']
            person_conf = person['confidence']
            
            # Extract person crop
            person_crop = self.extract_person_crop(frame, bbox)
            
            # Predict action for this person
            action, action_conf = self.predict_action_for_person(i, person_crop)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            color = self.get_action_color(action)
            priority = self.get_action_priority(action)
            
            # Box thickness based on priority
            thickness = 2 + priority
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Action label
            label = f"{action.upper()}: {action_conf:.2f}"
            person_label = f"Person: {person_conf:.2f}"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 50), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, person_label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Priority indicator
            if priority > 0:
                priority_text = "⚠️" * priority
                cv2.putText(annotated_frame, priority_text, (x2 - 60, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Calculate FPS
        frame_time = time.time() - frame_start_time
        self.fps_counter.append(frame_time)
        
        # Add performance info
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection stats
        if len(self.detection_times) > 0 and len(self.action_times) > 0:
            avg_det_time = sum(self.detection_times) / len(self.detection_times)
            avg_act_time = sum(self.action_times) / len(self.action_times)
            
            det_text = f"Detection: {avg_det_time*1000:.1f}ms"
            act_text = f"Action: {avg_act_time*1000:.1f}ms"
            
            cv2.putText(annotated_frame, det_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(annotated_frame, act_text, (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add person count
        person_count_text = f"Persons: {len(persons)}"
        cv2.putText(annotated_frame, person_count_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame, persons

def test_integrated_webcam(action_model_path, class_mapping_path, sequence_length=8, confidence_threshold=0.5):
    """Test integrated system on webcam"""
    print("🎬 Testing Integrated Person Detection + Action Recognition")
    print("Press 'q' to quit")
    
    # Initialize system
    system = IntegratedPersonActionInference(
        action_model_path, class_mapping_path, None, 
        sequence_length, confidence_threshold
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, persons = system.process_frame(frame)
            
            # Show frame
            cv2.imshow('Integrated Person Action Recognition', annotated_frame)
            
            # Print detected actions
            if persons:
                actions = []
                for i, person in enumerate(persons):
                    if i in system.person_actions:
                        action = system.person_actions[i]
                        conf = system.person_confidences[i]
                        actions.append(f"{action}({conf:.2f})")
                
                if actions:
                    print(f"Detected actions: {', '.join(actions)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def test_integrated_video(action_model_path, class_mapping_path, video_path, output_path=None, 
                         sequence_length=8, confidence_threshold=0.5):
    """Test integrated system on video"""
    print(f"🎬 Testing Integrated System on Video: {video_path}")
    
    # Initialize system
    system = IntegratedPersonActionInference(
        action_model_path, class_mapping_path, None, 
        sequence_length, confidence_threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, persons = system.process_frame(frame)
            
            # Write to output
            if writer:
                writer.write(annotated_frame)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                actions = []
                for i in range(len(persons)):
                    if i in system.person_actions:
                        action = system.person_actions[i]
                        conf = system.person_confidences[i]
                        actions.append(f"{action}({conf:.2f})")
                
                action_str = ', '.join(actions) if actions else 'No actions'
                print(f"Progress: {progress:.1f}% - {action_str}")
            
            # Show frame
            cv2.imshow('Integrated System', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"✅ Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Integrated Person Detection + Action Recognition')
    parser.add_argument('--action-model', default='ultra_fast_action_model.pt',
                       help='Action recognition model')
    parser.add_argument('--class-mapping', default=r'D:\SPHAR-Dataset\action_recognition_optimized\class_mapping.json',
                       help='Class mapping file')
    parser.add_argument('--source', default='webcam',
                       help='Source: "webcam" or video path')
    parser.add_argument('--output', default=None,
                       help='Output video path')
    parser.add_argument('--sequence-length', type=int, default=8,
                       help='Sequence length for action recognition')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                       help='Action confidence threshold')
    
    args = parser.parse_args()
    
    # Check model path
    model_path = Path(args.action_model)
    if not model_path.is_absolute():
        models_dir = Path(__file__).parent.parent / 'models'
        model_path = models_dir / args.action_model
    
    if not model_path.exists():
        print(f"❌ Action model not found: {model_path}")
        return
    
    # Check class mapping
    if not Path(args.class_mapping).exists():
        print(f"❌ Class mapping not found: {args.class_mapping}")
        return
    
    # Run test
    if args.source.lower() == 'webcam':
        test_integrated_webcam(str(model_path), args.class_mapping, 
                              args.sequence_length, args.confidence_threshold)
    else:
        test_integrated_video(str(model_path), args.class_mapping, args.source, args.output,
                             args.sequence_length, args.confidence_threshold)

if __name__ == "__main__":
    main()
