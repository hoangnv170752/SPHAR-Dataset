#!/usr/bin/env python3
"""
Test YOLOv11s fine-tuned model with DeepSORT tracking
Real-time human detection and tracking on video
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import time
from collections import defaultdict
from ultralytics import YOLO

# Try to import deep_sort_realtime
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    print("⚠️ DeepSORT not available. Install: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False

class YOLODeepSORTTracker:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, max_age=30):
        """
        Initialize YOLO + DeepSORT tracker
        
        Args:
            model_path: Path to fine-tuned YOLO model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IOU threshold for NMS
            max_age: Maximum frames to keep track alive
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        print(f"🔥 Loading YOLO model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ Using device: {self.device}")
        
        # Initialize DeepSORT
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=torch.cuda.is_available(),
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            print("✅ DeepSORT tracker initialized")
        else:
            self.tracker = None
            print("⚠️ DeepSORT not available, using simple tracking")
        
        # Tracking statistics
        self.track_history = defaultdict(list)
        self.track_colors = {}
        self.track_info = {}  # Store track information (first seen, last seen, etc)
        self.frame_count = 0
        self.total_detections = 0
        self.fps_history = []
        self.unique_tracks_seen = set()
        
    def generate_color(self, track_id):
        """Generate consistent color for each track ID"""
        if track_id not in self.track_colors:
            # Convert track_id to int if it's string
            seed = int(track_id) if isinstance(track_id, str) else track_id
            np.random.seed(seed)
            self.track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.track_colors[track_id]
    
    def process_frame(self, frame):
        """
        Process single frame with YOLO detection and DeepSORT tracking
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            annotated_frame: Frame with detections and tracks
            detections: List of detections
            tracks: List of tracks
        """
        start_time = time.time()
        
        # YOLO detection
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Extract detections
        detections = []
        boxes_for_tracking = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    # Only keep detections with confidence > 0.5
                    if conf <= 0.5:
                        continue
                    
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    detection = {
                        'bbox': [x1, y1, w, h],
                        'confidence': float(conf),
                        'class': 0,  # person class
                        'xyxy': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Format for DeepSORT: ([x1, y1, w, h], confidence, class)
                    boxes_for_tracking.append(([x1, y1, w, h], conf, 'person'))
        
        self.total_detections += len(detections)
        
        # DeepSORT tracking
        tracks = []
        detection_map = {}  # Map track to detection for confidence
        if self.tracker and boxes_for_tracking:
            tracks = self.tracker.update_tracks(boxes_for_tracking, frame=frame)
            
            # Map detections to tracks for confidence display
            for track in tracks:
                if track.is_confirmed():
                    track_bbox = track.to_ltrb()
                    # Find matching detection
                    for det in detections:
                        det_bbox = det['xyxy']
                        # Check if boxes match (simple overlap check)
                        iou = self._calculate_iou(track_bbox, det_bbox)
                        if iou > 0.5:
                            detection_map[track.track_id] = det['confidence']
                            break
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame.copy(), detections, tracks, fps, detection_map)
        
        self.frame_count += 1
        
        return annotated_frame, detections, tracks
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def annotate_frame(self, frame, detections, tracks, fps, detection_map=None):
        """Draw detections and tracks on frame"""
        h, w = frame.shape[:2]
        
        if detection_map is None:
            detection_map = {}
        
        # Draw tracks if available
        if tracks:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Ensure bounding box is within frame boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Skip if box is invalid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Track this person
                self.unique_tracks_seen.add(track_id)
                
                # Update track info
                if track_id not in self.track_info:
                    self.track_info[track_id] = {
                        'first_seen': self.frame_count,
                        'frames_tracked': 0
                    }
                self.track_info[track_id]['last_seen'] = self.frame_count
                self.track_info[track_id]['frames_tracked'] += 1
                
                # Get color for this track
                color = self.generate_color(track_id)
                
                # Draw thicker bounding box for better visibility
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw track ID with more info including confidence
                frames_alive = self.track_info[track_id]['frames_tracked']
                conf = detection_map.get(track_id, 0.0)
                
                label = f"Person #{track_id} [{conf:.2f}]"
                sub_label = f"Tracked: {frames_alive}f"
                
                # Main label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                sub_label_size, _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw labels background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - sub_label_size[1] - 20), 
                            (x1 + max(label_size[0], sub_label_size[0]) + 10, y1), color, -1)
                
                # Draw labels text
                cv2.putText(frame, label, (x1 + 5, y1 - sub_label_size[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, sub_label, (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Draw track history (trajectory)
                self.track_history[track_id].append((center_x, center_y))
                if len(self.track_history[track_id]) > 50:  # Longer trail
                    self.track_history[track_id].pop(0)
                
                # Draw trajectory with fading effect
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    # Thickness decreases for older points
                    thickness = int(np.sqrt(50 / float(i + 1)) * 2.5)
                    cv2.line(frame, points[i - 1], points[i], color, thickness)
        else:
            # Draw detections only (no tracking)
            for det in detections:
                x1, y1, x2, y2 = map(int, det['xyxy'])
                conf = det['confidence']
                
                # Ensure bounding box is within frame boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Skip if box is invalid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Draw bounding box (thicker)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw confidence with better visibility
                label = f"Person [{conf:.2f}]"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Background for label
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw enhanced statistics overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Title
        cv2.putText(frame, "=== TRACKING STATUS ===", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        stats_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {avg_fps:.1f}",
            f"Confidence Filter: > 0.5",
            f"Active Tracks: {len([t for t in tracks if t.is_confirmed()]) if tracks else 0}",
            f"Total People Seen: {len(self.unique_tracks_seen)}",
            f"DeepSORT: {'ACTIVE' if self.tracker and tracks else 'INACTIVE'}"
        ]
        
        y_offset = 55
        for text in stats_text:
            # Highlight important status
            if 'ACTIVE' in text:
                color = (0, 255, 0)  # Green for active
            elif 'Confidence Filter' in text:
                color = (0, 165, 255)  # Orange for filter info
            else:
                color = (255, 255, 255)  # White for normal
            
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        return frame
    
    def process_video(self, video_path, output_path=None, display=True, save_stats=True):
        """
        Process entire video with tracking
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Show video while processing
            save_stats: Save tracking statistics
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return
        
        print(f"📹 Processing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video writer
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        # Process frames
        print("🚀 Starting processing...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections, tracks = self.process_frame(frame)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('YOLO + DeepSORT Tracking', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("⏹️ Stopped by user")
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)  # Pause
                
                # Progress
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    avg_fps = np.mean(self.fps_history[-30:]) if self.fps_history else 0
                    print(f"Progress: {progress:.1f}% | Frame: {self.frame_count}/{total_frames} | FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print summary
            self.print_summary()
            
            # Save statistics
            if save_stats and output_path:
                self.save_statistics(output_path.parent / 'tracking_stats.txt')
    
    def process_webcam(self):
        """Process webcam stream with real-time tracking"""
        print("📹 Starting webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("🚀 Press 'q' to quit, 'p' to pause")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections, tracks = self.process_frame(frame)
                
                # Display
                cv2.imshow('YOLO + DeepSORT - Webcam', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_summary()
    
    def print_summary(self):
        """Print tracking summary"""
        print("\n" + "="*70)
        print("📊 DEEPSORT TRACKING SUMMARY")
        print("="*70)
        print(f"🎯 Confidence Filter: > 0.5 (High confidence only)")
        print(f"✅ Total frames processed: {self.frame_count}")
        print(f"✅ Total detections: {self.total_detections}")
        print(f"✅ Average detections/frame: {self.total_detections / max(1, self.frame_count):.2f}")
        if self.fps_history:
            print(f"✅ Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"\n👥 PEOPLE TRACKING:")
        print(f"   Total unique people tracked: {len(self.unique_tracks_seen)}")
        print(f"   Track IDs: {sorted(list(self.unique_tracks_seen))}")
        
        if self.track_info:
            print(f"\n📋 INDIVIDUAL TRACK DETAILS:")
            for track_id in sorted(self.track_info.keys()):
                info = self.track_info[track_id]
                frames = info['frames_tracked']
                duration_sec = frames / max(1, np.mean(self.fps_history) if self.fps_history else 30)
                print(f"   Person #{track_id}:")
                print(f"      - Tracked for: {frames} frames ({duration_sec:.1f}s)")
                print(f"      - First seen: Frame {info['first_seen']}")
                print(f"      - Last seen: Frame {info['last_seen']}")
        
        print("="*70)
    
    def save_statistics(self, stats_path):
        """Save tracking statistics to file"""
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("YOLO + DeepSORT Tracking Statistics\n")
            f.write("="*70 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Confidence Filter: > 0.5 (High confidence only)\n\n")
            
            f.write("OVERVIEW:\n")
            f.write(f"  Total frames: {self.frame_count}\n")
            f.write(f"  Total detections: {self.total_detections}\n")
            f.write(f"  Avg detections/frame: {self.total_detections / max(1, self.frame_count):.2f}\n")
            if self.fps_history:
                f.write(f"  Average FPS: {np.mean(self.fps_history):.2f}\n")
            
            f.write(f"\nPEOPLE TRACKING:\n")
            f.write(f"  Total unique people tracked: {len(self.unique_tracks_seen)}\n")
            f.write(f"  Track IDs: {sorted(list(self.unique_tracks_seen))}\n")
            
            if self.track_info:
                f.write(f"\nINDIVIDUAL TRACK DETAILS:\n")
                for track_id in sorted(self.track_info.keys()):
                    info = self.track_info[track_id]
                    frames = info['frames_tracked']
                    duration_sec = frames / max(1, np.mean(self.fps_history) if self.fps_history else 30)
                    f.write(f"\n  Person #{track_id}:\n")
                    f.write(f"    - Tracked for: {frames} frames ({duration_sec:.1f}s)\n")
                    f.write(f"    - First seen: Frame {info['first_seen']}\n")
                    f.write(f"    - Last seen: Frame {info['last_seen']}\n")
        
        print(f"📄 Statistics saved: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description='Test YOLO fine-tuned model with DeepSORT tracking')
    parser.add_argument('--model', '-m', 
                       default=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                       help='Path to fine-tuned YOLO model')
    parser.add_argument('--source', '-s',
                       help='Video file path (or "webcam" for webcam)')
    parser.add_argument('--output', '-o',
                       help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold (default: 0.45)')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum frames to keep track (default: 30)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video while processing')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = YOLODeepSORTTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_age=args.max_age
    )
    
    # Process video or webcam
    if args.source:
        if args.source.lower() == 'webcam':
            tracker.process_webcam()
        else:
            tracker.process_video(
                video_path=args.source,
                output_path=args.output,
                display=not args.no_display
            )
    else:
        print("❌ Please specify --source (video file or 'webcam')")
        print("\nExamples:")
        print("  python test_yolo_deepsort.py --source webcam")
        print("  python test_yolo_deepsort.py --source video.mp4 --output tracked.mp4")

if __name__ == "__main__":
    main()
