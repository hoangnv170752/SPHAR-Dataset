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
try:
    from action_recognition_trainer import SlowFastActionModel
except ImportError:
    SlowFastActionModel = None

try:
    from image_action_trainer import OptimizedSlowFastModel
except ImportError:
    OptimizedSlowFastModel = None

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
        
        # Try to load checkpoint first to determine model type
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try OptimizedSlowFastModel first (for ultra_fast_action_model.pt)
        if OptimizedSlowFastModel is not None:
            try:
                self.model = OptimizedSlowFastModel(self.num_classes, sequence_length)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded OptimizedSlowFastModel")
            except Exception as e:
                print(f"⚠️ Failed to load OptimizedSlowFastModel: {e}")
                self.model = None
        
        # Fallback to original SlowFastActionModel
        if self.model is None and SlowFastActionModel is not None:
            try:
                self.model = SlowFastActionModel(self.num_classes, sequence_length)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded SlowFastActionModel")
            except Exception as e:
                print(f"❌ Failed to load SlowFastActionModel: {e}")
                raise e
        
        if self.model is None:
            raise RuntimeError("Could not load any model architecture")
        
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
    
    def get_smoothed_prediction(self, current_prediction, confidence, raw_confidence):
        """Smooth predictions using history"""
        if confidence > self.confidence_threshold:
            self.action_history.append(current_prediction)
        
        if len(self.action_history) == 0:
            return "normal", 0.0, 0.0
        
        if len(self.action_history) > 0:
            # Get most common action in recent history
            action_counts = {}
            for hist_action in self.action_history:
                action_counts[hist_action] = action_counts.get(hist_action, 0) + 1
            
            smoothed_action = max(action_counts, key=action_counts.get)
            return smoothed_action, confidence, raw_confidence
        
        return "normal", 0.0, 0.0
    
    def process_frame(self, frame):
        """Process single frame and return action prediction"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Predict if buffer is full
        if len(self.frame_buffer) == self.sequence_length:
            # Convert buffer to numpy array
            frames = np.array(list(self.frame_buffer))
            predicted_action, confidence = self.predict_action(frames)
            
            # Apply smoothing
            smoothed_action, smoothed_confidence, raw_confidence = self.get_smoothed_prediction(predicted_action, confidence, confidence)
            
            return smoothed_action, smoothed_confidence, raw_confidence
        
        return "normal", 0.0, 0.0
    
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
        
        # Handle None action
        if action is None:
            action = "unknown"
            confidence = 0.0
        
        # Create overlay
        overlay = frame.copy()
        
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

def test_on_webcam(model_path, class_mapping_path, sequence_length=8, confidence_threshold=0.5):
    """Test action recognition on webcam"""
    print("🎬 Testing Action Recognition on Webcam")
    print("Press 'q' to quit")
    
    # Initialize inference
    inference = ActionRecognitionInference(model_path, class_mapping_path, sequence_length, confidence_threshold)
    
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
            action, confidence, raw_confidence = inference.process_frame(frame)
            
            # Annotate frame
            annotated_frame = inference.annotate_frame(frame, action, confidence, raw_confidence)
            
            # Show frame
            cv2.imshow('Action Recognition', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def test_on_video(model_path, class_mapping_path, video_path, output_path=None, sequence_length=8, confidence_threshold=0.5):
    """Test action recognition on video file"""
    print(f"🎬 Testing Action Recognition on Video: {video_path}")
    
    # Initialize inference
    inference = ActionRecognitionInference(model_path, class_mapping_path, sequence_length, confidence_threshold)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Video writer for output
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
            action, confidence, raw_confidence = inference.process_frame(frame)
            
            # Annotate frame
            annotated_frame = inference.annotate_frame(frame, action, confidence, raw_confidence)
            
            # Write to output video
            if writer:
                writer.write(annotated_frame)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Action: {action} ({confidence:.2f})")
            
            # Show frame (optional)
            cv2.imshow('Action Recognition', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"✅ Output saved to: {output_path}")

def test_on_random_video(model_path, class_mapping_path, sequence_length=8, confidence_threshold=0.5):
    """Test action recognition on a random video from test folder"""
    import random
    
    # Find test videos
    test_dirs = [
        Path(r'D:\SPHAR-Dataset\action_recognition_optimized\test'),
        Path(r'D:\SPHAR-Dataset\action_recognition\test'),
        Path(r'D:\SPHAR-Dataset\videos')
    ]
    
    test_videos = []
    for test_dir in test_dirs:
        if test_dir.exists():
            # Find all video files in subdirectories
            for action_dir in test_dir.iterdir():
                if action_dir.is_dir():
                    videos = list(action_dir.glob('*.mp4')) + list(action_dir.glob('*.avi'))
                    for video in videos:
                        test_videos.append((video, action_dir.name))
    
    if not test_videos:
        print("❌ No test videos found in:")
        for test_dir in test_dirs:
            print(f"   {test_dir}")
        return
    
    # Select random video
    video_path, true_action = random.choice(test_videos)
    print(f"🎬 Testing on random video: {video_path.name}")
    print(f"📊 True action: {true_action}")
    print(f"Press 'q' to quit, 's' to select another video")
    
    # Test on selected video
    test_on_video(model_path, class_mapping_path, str(video_path), None, sequence_length, confidence_threshold)

def main():
    parser = argparse.ArgumentParser(description='Test Action Recognition Model')
    parser.add_argument('--model', default='action_recognition_slowfast.pt',
                       help='Path to trained model')
    parser.add_argument('--class-mapping', default=r'D:\SPHAR-Dataset\action_recognition_optimized\class_mapping.json',
                       help='Path to class mapping file')
    parser.add_argument('--source', default='webcam',
                       help='Source: "webcam", "random", or path to video file')
    parser.add_argument('--output', default=None,
                       help='Output video path (optional)')
    parser.add_argument('--sequence-length', type=int, default=8,
                       help='Sequence length for model (8 for ultra_fast, 16 for original)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Check model path - look in models directory if not absolute path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # Try in models directory
        models_dir = Path(__file__).parent.parent / 'models'
        model_path = models_dir / args.model
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        models_dir = Path(__file__).parent.parent / 'models'
        if models_dir.exists():
            for model_file in models_dir.glob('*.pt'):
                print(f"   {model_file.name}")
        print("Please train the model first or specify correct path")
        return
    
    args.model = str(model_path)
    
    # Check if class mapping exists
    if not Path(args.class_mapping).exists():
        print(f"❌ Class mapping not found: {args.class_mapping}")
        print("Please organize dataset first using action_recognition_trainer.py --organize-only")
        return
    
    # Run inference
    if args.source.lower() == 'webcam':
        test_on_webcam(args.model, args.class_mapping, args.sequence_length, args.confidence_threshold)
    elif args.source.lower() == 'random':
        test_on_random_video(args.model, args.class_mapping, args.sequence_length, args.confidence_threshold)
    else:
        test_on_video(args.model, args.class_mapping, args.source, args.output, args.sequence_length, args.confidence_threshold)
        print("  python test_action_recognition.py --source video.mp4 --output result.mp4")

if __name__ == "__main__":
    main()
