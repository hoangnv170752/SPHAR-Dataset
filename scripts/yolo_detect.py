#!/usr/bin/env python3
"""
YOLO Human Detection Script
Step 1 of the pipeline: Detect humans in video frames using YOLO

This script:
1. Loads YOLO model optimized for person detection
2. Processes video frames to detect humans
3. Outputs detection results with bounding boxes and confidence scores
4. Saves detection data for DeepSORT tracking

Author: Generated for YOLO + DeepSORT + Activity Detection pipeline
"""

import cv2
import torch
import numpy as np
import argparse
import json
import time
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import pickle

class YOLOHumanDetector:
    def __init__(self, model_path="yolov8n.pt", device='cuda', conf_threshold=0.5):
        """
        Initialize YOLO Human Detector
        
        Args:
            model_path: Path to YOLO model
            device: Device to run inference on
            conf_threshold: Confidence threshold for detections
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        print(f"Using device: {self.device}")
        print(f"Confidence threshold: {self.conf_threshold}")
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        
        # Warm up the model
        print("Warming up model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_img, classes=[0], verbose=False)
        print("Model ready!")
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'detection_counts_per_frame': []
        }
    
    def detect_humans(self, frame):
        """
        Detect humans in a single frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': 0}]
        """
        start_time = time.time()
        
        # Run YOLO inference (class 0 = person)
        results = self.model(frame, classes=[0], verbose=False, conf=self.conf_threshold)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Ensure bounding box is within frame bounds
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                
                # Only add valid detections
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
        
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        self.stats['detection_counts_per_frame'].append(len(detections))
        
        return detections
    
    def process_video(self, video_path, output_dir=None, save_frames=False, visualize=True):
        """
        Process entire video and detect humans in each frame
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            save_frames: Whether to save annotated frames
            visualize: Whether to display video during processing
            
        Returns:
            Dictionary containing all detection results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if save_frames:
                frames_dir = output_dir / 'annotated_frames'
                frames_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        video_results = {
            'video_info': {
                'path': str(video_path),
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            },
            'detections': {},  # frame_id: [detections]
            'processing_stats': {}
        }
        
        frame_count = 0
        
        print("Starting detection...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect humans in current frame
            detections = self.detect_humans(frame)
            video_results['detections'][frame_count] = detections
            
            # Visualize detections
            if visualize or save_frames:
                annotated_frame = self.draw_detections(frame.copy(), detections, frame_count)
                
                if visualize:
                    cv2.imshow('YOLO Human Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if save_frames and output_dir:
                    frame_filename = frames_dir / f'frame_{frame_count:06d}.jpg'
                    cv2.imwrite(str(frame_filename), annotated_frame)
            
            frame_count += 1
            self.stats['total_frames'] = frame_count
            self.stats['total_detections'] += len(detections)
            
            # Print progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(self.stats['processing_times'][-100:])
                avg_detections = np.mean(self.stats['detection_counts_per_frame'][-100:])
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) | "
                      f"Avg time: {avg_time:.3f}s | Avg detections: {avg_detections:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Add final statistics
        video_results['processing_stats'] = {
            'total_frames_processed': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'avg_processing_time': np.mean(self.stats['processing_times']),
            'avg_detections_per_frame': np.mean(self.stats['detection_counts_per_frame']),
            'fps_achieved': 1.0 / np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        }
        
        print(f"\nDetection completed!")
        print(f"Processed {self.stats['total_frames']} frames")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Average processing time: {video_results['processing_stats']['avg_processing_time']:.3f}s per frame")
        print(f"Average detections per frame: {video_results['processing_stats']['avg_detections_per_frame']:.2f}")
        print(f"Processing FPS: {video_results['processing_stats']['fps_achieved']:.1f}")
        
        # Save results
        if output_dir:
            # Save as JSON
            json_path = output_dir / 'yolo_detections.json'
            with open(json_path, 'w') as f:
                json.dump(video_results, f, indent=2)
            print(f"Results saved to: {json_path}")
            
            # Save as pickle for faster loading
            pickle_path = output_dir / 'yolo_detections.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(video_results, f)
            print(f"Pickle results saved to: {pickle_path}")
        
        return video_results
    
    def draw_detections(self, frame, detections, frame_id):
        """Draw detection bounding boxes on frame"""
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {i+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw frame info
        info_text = f"Frame: {frame_id} | Detections: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_batch_videos(self, video_list, output_base_dir):
        """Process multiple videos"""
        results = {}
        
        for video_path in video_list:
            video_name = Path(video_path).stem
            output_dir = Path(output_base_dir) / video_name
            
            print(f"\n{'='*50}")
            print(f"Processing: {video_name}")
            print(f"{'='*50}")
            
            # Reset stats for each video
            self.stats = {
                'total_frames': 0,
                'total_detections': 0,
                'processing_times': [],
                'detection_counts_per_frame': []
            }
            
            video_results = self.process_video(video_path, output_dir, save_frames=False, visualize=False)
            results[video_name] = video_results
        
        # Save batch results
        batch_results_path = Path(output_base_dir) / 'batch_results.json'
        with open(batch_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing completed! Results saved to: {batch_results_path}")
        return results

def main():
    parser = argparse.ArgumentParser(description='YOLO Human Detection')
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--save-frames', action='store_true', help='Save annotated frames')
    parser.add_argument('--no-display', action='store_true', help='Do not display video')
    parser.add_argument('--batch', help='Process multiple videos from directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOHumanDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    if args.batch:
        # Process batch of videos
        video_dir = Path(args.batch)
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return
        
        output_dir = args.output or 'yolo_batch_results'
        detector.process_batch_videos(video_files, output_dir)
    
    else:
        # Process single video
        output_dir = args.output or f'yolo_results_{Path(args.video).stem}'
        detector.process_video(
            video_path=args.video,
            output_dir=output_dir,
            save_frames=args.save_frames,
            visualize=not args.no_display
        )

if __name__ == "__main__":
    main()
