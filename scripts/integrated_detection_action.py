#!/usr/bin/env python3
"""
Integrated Human Detection + Action Recognition System
Combines YOLO11s-detect.pt for human detection with SlowFast for action recognition
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import json
import argparse
import time
from collections import deque, defaultdict
from ultralytics import YOLO
from test_action_recognition import ActionRecognitionInference

class IntegratedDetectionAction:
    """Integrated system for human detection and action recognition"""
    
    def __init__(self, yolo_model_path, action_model_path, class_mapping_path, 
                 conf_threshold=0.5, sequence_length=16):
        
        self.conf_threshold = conf_threshold
        self.sequence_length = sequence_length
        
        # Load YOLO model for human detection
        print("🔥 Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Action Recognition model
        print("🔥 Loading Action Recognition model...")
        self.action_inference = ActionRecognitionInference(
            action_model_path, class_mapping_path, sequence_length
        )
        
        # Track individual persons and their actions
        self.person_tracks = {}  # track_id -> person info
        self.next_track_id = 1
        
        # Frame buffers for each person
        self.person_frame_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
        
        # Action history for each person
        self.person_action_history = defaultdict(lambda: deque(maxlen=10))
        
        print("✅ Integrated system initialized")
        print(f"⚡ Device: {self.device}")
    
    def detect_humans(self, frame):
        """Detect humans in frame using YOLO"""
        results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    # Only keep high confidence detections
                    if conf > 0.5:
                        x1, y1, x2, y2 = box
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf)
                        })
        
        return detections
    
    def extract_person_crop(self, frame, bbox, margin=0.1):
        """Extract person crop from frame with margin"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        crop = frame[y1:y2, x1:x2]
        return crop
    
    def simple_tracking(self, current_detections, previous_tracks, iou_threshold=0.3):
        """Simple tracking based on IoU overlap"""
        def calculate_iou(box1, box2):
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            
            # Calculate intersection
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Match current detections with previous tracks
        matched_tracks = {}
        unmatched_detections = list(current_detections)
        
        for track_id, track_info in previous_tracks.items():
            best_iou = 0
            best_detection = None
            
            for detection in unmatched_detections:
                iou = calculate_iou(detection['bbox'], track_info['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_detection = detection
            
            if best_detection:
                matched_tracks[track_id] = {
                    'bbox': best_detection['bbox'],
                    'confidence': best_detection['confidence'],
                    'last_seen': time.time()
                }
                unmatched_detections.remove(best_detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            matched_tracks[self.next_track_id] = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'last_seen': time.time()
            }
            self.next_track_id += 1
        
        return matched_tracks
    
    def process_frame(self, frame):
        """Process single frame with detection and action recognition"""
        # Detect humans
        detections = self.detect_humans(frame)
        
        # Update tracks
        self.person_tracks = self.simple_tracking(detections, self.person_tracks)
        
        # Remove old tracks (not seen for 2 seconds)
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_info in self.person_tracks.items():
            if current_time - track_info['last_seen'] > 2.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.person_tracks[track_id]
            if track_id in self.person_frame_buffers:
                del self.person_frame_buffers[track_id]
            if track_id in self.person_action_history:
                del self.person_action_history[track_id]
        
        # Process each person for action recognition
        person_actions = {}
        
        for track_id, track_info in self.person_tracks.items():
            bbox = track_info['bbox']
            
            # Extract person crop
            person_crop = self.extract_person_crop(frame, bbox)
            
            if person_crop.size > 0:
                # Preprocess crop for action recognition
                processed_crop = self.action_inference.preprocess_frame(person_crop)
                
                # Add to person's frame buffer
                self.person_frame_buffers[track_id].append(processed_crop)
                
                # Predict action if buffer is full
                if len(self.person_frame_buffers[track_id]) == self.sequence_length:
                    predicted_action, confidence = self.action_inference.predict_action(
                        list(self.person_frame_buffers[track_id])
                    )
                    
                    if predicted_action and confidence > self.action_inference.confidence_threshold:
                        # Update person's action history
                        self.person_action_history[track_id].append(predicted_action)
                        
                        # Get smoothed action
                        if len(self.person_action_history[track_id]) > 0:
                            from collections import Counter
                            action_counts = Counter(self.person_action_history[track_id])
                            most_common_action = action_counts.most_common(1)[0][0]
                            smoothed_confidence = action_counts[most_common_action] / len(self.person_action_history[track_id])
                            
                            person_actions[track_id] = {
                                'action': most_common_action,
                                'confidence': smoothed_confidence,
                                'raw_confidence': confidence,
                                'bbox': bbox
                            }
        
        return person_actions
    
    def annotate_frame(self, frame, person_actions):
        """Annotate frame with detections and actions"""
        annotated_frame = frame.copy()
        
        # Draw each person
        for track_id, action_info in person_actions.items():
            bbox = action_info['bbox']
            action = action_info['action']
            confidence = action_info['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Get action color and priority
            color = self.action_inference.get_action_color(action)
            priority = self.action_inference.get_action_priority(action)
            
            # Draw bounding box
            thickness = 3 if priority >= 2 else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Person label
            label = f"Person #{track_id}"
            action_label = f"{action.upper()} ({confidence:.2f})"
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            action_size = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            max_width = max(label_size[0], action_size[0])
            
            cv2.rectangle(annotated_frame, (x1, y1 - 50), (x1 + max_width + 10, y1), (0, 0, 0), -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, action_label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Priority indicator
            if priority >= 3:
                cv2.putText(annotated_frame, "🚨 EMERGENCY", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif priority >= 2:
                cv2.putText(annotated_frame, "⚠️ ALERT", (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # System info
        info_text = f"Tracking {len(person_actions)} people"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Action summary
        if person_actions:
            actions_summary = {}
            for action_info in person_actions.values():
                action = action_info['action']
                if action not in actions_summary:
                    actions_summary[action] = 0
                actions_summary[action] += 1
            
            summary_text = " | ".join([f"{action}: {count}" for action, count in actions_summary.items()])
            cv2.putText(annotated_frame, summary_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated_frame

def test_on_video(yolo_model, action_model, class_mapping, video_path, output_path=None):
    """Test integrated system on video"""
    
    # Initialize system
    system = IntegratedDetectionAction(yolo_model, action_model, class_mapping)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {Path(video_path).name}")
    print(f"📊 Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    action_stats = defaultdict(int)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            person_actions = system.process_frame(frame)
            
            # Count actions
            for action_info in person_actions.values():
                action_stats[action_info['action']] += 1
            
            # Annotate frame
            annotated_frame = system.annotate_frame(frame, person_actions)
            
            # Show frame
            cv2.imshow('Integrated Detection + Action Recognition', annotated_frame)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames}")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 INTEGRATED SYSTEM SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print("\nAction distribution:")
    for action, count in sorted(action_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action:15s}: {count:6d} detections")
    
    if output_path:
        print(f"\n💾 Output saved: {output_path}")

def test_on_webcam(yolo_model, action_model, class_mapping):
    """Test integrated system on webcam"""
    
    # Initialize system
    system = IntegratedDetectionAction(yolo_model, action_model, class_mapping)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Starting integrated detection + action recognition...")
    print("🚀 Press 'q' to quit")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            person_actions = system.process_frame(frame)
            
            # Annotate frame
            annotated_frame = system.annotate_frame(frame, person_actions)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}")
            
            # Show frame
            cv2.imshow('Integrated System - Webcam', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Integrated Human Detection + Action Recognition')
    parser.add_argument('--yolo-model', default=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                       help='Path to YOLO model for human detection')
    parser.add_argument('--action-model', default=r'D:\SPHAR-Dataset\models\action_recognition_slowfast.pt',
                       help='Path to action recognition model')
    parser.add_argument('--class-mapping', default=r'D:\SPHAR-Dataset\action_recognition\class_mapping.json',
                       help='Path to class mapping file')
    parser.add_argument('--source', help='Video file path or "webcam"')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='YOLO confidence threshold')
    
    args = parser.parse_args()
    
    # Check models exist
    if not Path(args.yolo_model).exists():
        print(f"❌ YOLO model not found: {args.yolo_model}")
        return
    
    if not Path(args.action_model).exists():
        print(f"❌ Action model not found: {args.action_model}")
        print("Please train the action recognition model first")
        return
    
    if not Path(args.class_mapping).exists():
        print(f"❌ Class mapping not found: {args.class_mapping}")
        return
    
    if args.source:
        if args.source.lower() == 'webcam':
            test_on_webcam(args.yolo_model, args.action_model, args.class_mapping)
        else:
            test_on_video(args.yolo_model, args.action_model, args.class_mapping, 
                         args.source, args.output)
    else:
        print("❌ Please specify --source (video file or 'webcam')")
        print("\nExamples:")
        print("  python integrated_detection_action.py --source webcam")
        print("  python integrated_detection_action.py --source video.mp4 --output result.mp4")

if __name__ == "__main__":
    main()
