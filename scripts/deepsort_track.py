#!/usr/bin/env python3
"""
DeepSORT Tracking Script
Step 2: Track detected humans using DeepSORT
"""

import cv2
import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "deep-sort-realtime"])
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError:
        # Fallback to alternative import
        from deep_sort_realtime import DeepSort

class DeepSORTTracker:
    def __init__(self, max_age=50, n_init=3):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        self.track_data = defaultdict(list)
        
    def track_detections(self, detections, frame, frame_id):
        """Track detections in frame"""
        detection_list = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            detection_list.append(([x1, y1, w, h], det['confidence'], 'person'))
        
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        track_results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            confidence = getattr(track, 'det_conf', None)
            if confidence is None:
                confidence = 1.0
            
            track_result = {
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence)
            }
            track_results.append(track_result)
            
            # Store track data
            self.track_data[track_id].append({
                'frame_id': frame_id,
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        return track_results
    
    def process_video(self, video_path, detection_results, output_dir=None):
        """Process video with tracking"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        tracking_results = {
            'video_info': {'fps': fps, 'total_frames': total_frames},
            'tracks': {},
            'track_data': {}
        }
        
        detections_data = detection_results.get('detections', {})
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_detections = detections_data.get(str(frame_count), [])
            tracks = self.track_detections(frame_detections, frame, frame_count)
            tracking_results['tracks'][frame_count] = tracks
            
            # Visualize
            annotated_frame = self.draw_tracks(frame.copy(), tracks)
            cv2.imshow('DeepSORT Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        cv2.destroyAllWindows()
        
        tracking_results['track_data'] = dict(self.track_data)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'tracking_results.json', 'w') as f:
                json.dump(tracking_results, f, indent=2)
            
            with open(output_dir / 'track_data.pkl', 'wb') as f:
                pickle.dump(dict(self.track_data), f)
        
        return tracking_results
    
    def draw_tracks(self, frame, tracks):
        """Draw tracking results on frame"""
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['confidence']
            
            # Different colors for different tracks
            color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

def main():
    parser = argparse.ArgumentParser(description='DeepSORT Tracking')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--detections', required=True, help='YOLO detection results file')
    parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    # Load detection results
    if args.detections.endswith('.pkl'):
        with open(args.detections, 'rb') as f:
            detection_results = pickle.load(f)
    else:
        with open(args.detections, 'r') as f:
            detection_results = json.load(f)
    
    # Initialize tracker
    tracker = DeepSORTTracker()
    
    # Process video
    output_dir = args.output or f'tracking_results_{Path(args.video).stem}'
    results = tracker.process_video(args.video, detection_results, output_dir)
    
    print(f"Tracking completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
