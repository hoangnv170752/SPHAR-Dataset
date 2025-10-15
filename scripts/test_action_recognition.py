#!/usr/bin/env python3
"""
Test Action Recognition Model
Real-time action classification using trained SlowFast model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from collections import deque
import time
from action_recognition_trainer import SlowFastActionModel

class ActionRecognitionInference:
    """Real-time action recognition inference"""
    
    def __init__(self, model_path, class_mapping_path, sequence_length=16, confidence_threshold=0.5):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SlowFastActionModel(self.num_classes, sequence_length)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Frame buffer for sequence
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Action history for smoothing
        self.action_history = deque(maxlen=10)
        
        print(f"✅ Action Recognition Model loaded")
        print(f"📊 Classes: {list(self.class_mapping.keys())}")
        print(f"⚡ Device: {self.device}")
    
    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        # Resize to model input size
        frame = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def predict_action(self, frames):
        """Predict action from sequence of frames"""
        if len(frames) < self.sequence_length:
            return None, 0.0
        
        # Convert to tensor
        frames_array = np.array(frames)  # (T, H, W, C)
        frames_tensor = torch.tensor(frames_array).unsqueeze(0).float() / 255.0  # (1, T, H, W, C)
        frames_tensor = frames_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(frames_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score
    
    def get_smoothed_prediction(self, current_prediction, confidence):
        """Smooth predictions using history"""
        if confidence > self.confidence_threshold:
            self.action_history.append(current_prediction)
        
        if len(self.action_history) == 0:
            return "unknown", 0.0
        
        # Get most common prediction in recent history
        from collections import Counter
        action_counts = Counter(self.action_history)
        most_common_action = action_counts.most_common(1)[0][0]
        
        # Calculate smoothed confidence
        smoothed_confidence = action_counts[most_common_action] / len(self.action_history)
        
        return most_common_action, smoothed_confidence
    
    def process_frame(self, frame):
        """Process single frame and return action prediction"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Predict if buffer is full
        if len(self.frame_buffer) == self.sequence_length:
            predicted_action, confidence = self.predict_action(list(self.frame_buffer))
            
            if predicted_action is not None:
                # Smooth prediction
                smoothed_action, smoothed_confidence = self.get_smoothed_prediction(predicted_action, confidence)
                return smoothed_action, smoothed_confidence, confidence
        
        return None, 0.0, 0.0
    
    def get_action_color(self, action):
        """Get color for action visualization"""
        colors = {
            'fall': (0, 0, 255),      # Red - Emergency
            'hitting': (0, 100, 255), # Orange - Violence
            'running': (0, 255, 255), # Yellow - Alert
            'warning': (0, 165, 255), # Orange - Warning
            'normal': (0, 255, 0),    # Green - Normal behavior
            'neutral': (0, 255, 0),   # Green - Normal
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
            'normal': 0,    # No priority - Normal behavior
            'neutral': 0,   # No priority - Normal
            'unknown': 0    # No priority
        }
        return priorities.get(action, 0)
    
    def annotate_frame(self, frame, action, confidence, raw_confidence):
        """Annotate frame with action prediction"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Action info panel
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Action text
        action_text = f"Action: {action.upper()}"
        confidence_text = f"Confidence: {confidence:.2f}"
        raw_conf_text = f"Raw Conf: {raw_confidence:.2f}"
        
        # Get action color and priority
        color = self.get_action_color(action)
        priority = self.get_action_priority(action)
        
        # Priority indicator
        priority_text = "🚨 EMERGENCY" if priority == 3 else "⚠️ ALERT" if priority == 2 else "⚡ WARNING" if priority == 1 else "✅ NORMAL"
        
        # Draw text
        cv2.putText(frame, action_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, confidence_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, raw_conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, priority_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Action history visualization
        if len(self.action_history) > 0:
            history_text = " -> ".join(list(self.action_history)[-5:])  # Last 5 actions
            cv2.putText(frame, f"History: {history_text}", (20, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence bar
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (width - 320, 20), (width - 20 + bar_width, 40), color, -1)
        cv2.rectangle(frame, (width - 320, 20), (width - 20, 40), (255, 255, 255), 2)
        
        return frame

def test_on_video(model_path, class_mapping_path, video_path, output_path=None):
    """Test action recognition on video file"""
    
    # Initialize inference
    inference = ActionRecognitionInference(model_path, class_mapping_path)
    
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
    
    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    action_counts = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            action, confidence, raw_confidence = inference.process_frame(frame)
            
            if action:
                # Count actions
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
                
                # Annotate frame
                annotated_frame = inference.annotate_frame(frame, action, confidence, raw_confidence)
            else:
                annotated_frame = frame
            
            # Show frame
            cv2.imshow('Action Recognition', annotated_frame)
            
            # Write frame if output specified
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
    print("📊 ACTION RECOGNITION SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print("\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / frame_count) * 100
        print(f"  {action:15s}: {count:6d} frames ({percentage:5.1f}%)")
    
    if output_path:
        print(f"\n💾 Output saved: {output_path}")

def test_on_webcam(model_path, class_mapping_path):
    """Test action recognition on webcam"""
    
    # Initialize inference
    inference = ActionRecognitionInference(model_path, class_mapping_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Starting webcam action recognition...")
    print("🚀 Press 'q' to quit")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            action, confidence, raw_confidence = inference.process_frame(frame)
            
            if action:
                # Annotate frame
                annotated_frame = inference.annotate_frame(frame, action, confidence, raw_confidence)
            else:
                annotated_frame = frame
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}")
            
            # Show frame
            cv2.imshow('Action Recognition - Webcam', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Action Recognition Model')
    parser.add_argument('--model', default=r'D:\SPHAR-Dataset\models\action_recognition_slowfast.pt',
                       help='Path to trained model')
    parser.add_argument('--class-mapping', default=r'D:\SPHAR-Dataset\action_recognition\class_mapping.json',
                       help='Path to class mapping file')
    parser.add_argument('--source', help='Video file path or "webcam"')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Please train the model first using action_recognition_trainer.py")
        return
    
    # Check if class mapping exists
    if not Path(args.class_mapping).exists():
        print(f"❌ Class mapping not found: {args.class_mapping}")
        print("Please organize dataset first using action_recognition_trainer.py --organize-only")
        return
    
    if args.source:
        if args.source.lower() == 'webcam':
            test_on_webcam(args.model, args.class_mapping)
        else:
            test_on_video(args.model, args.class_mapping, args.source, args.output)
    else:
        print("❌ Please specify --source (video file or 'webcam')")
        print("\nExamples:")
        print("  python test_action_recognition.py --source webcam")
        print("  python test_action_recognition.py --source video.mp4 --output result.mp4")

if __name__ == "__main__":
    main()
