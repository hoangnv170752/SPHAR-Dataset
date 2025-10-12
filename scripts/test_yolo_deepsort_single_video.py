#!/usr/bin/env python3
"""
Test YOLO + DeepSORT cho 1 video abnormal
"""

import argparse
import cv2
import time
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
from yolo_detect import YOLOHumanDetector
from deepsort_track import DeepSORTTracker

def test_single_video(video_path, output_dir="test_output"):
    """Test YOLO + DeepSORT cho 1 video"""
    
    print(f"🎬 Testing video: {video_path}")
    
    # Khởi tạo
    detector = YOLOHumanDetector(model_path='yolov8n.pt', device='cuda')
    tracker = DeepSORTTracker()
    
    # Tạo output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    output_path = output_dir / f"tracked_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    start_time = time.time()
    
    print(f"📊 Processing {fps} FPS, {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO Detection
        detections = detector.detect_humans(frame)
        
        if detections:
            total_detections += len(detections)
        
        # DeepSORT Tracking
        tracks = tracker.track_detections(detections or [], frame, frame_count)
        total_tracks = max(total_tracks, len(tracks))
        
        # Draw results
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            
            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Tracks: {len(tracks)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
        
        if frame_count % 30 == 0:
            print(f"📹 Frame {frame_count}, Tracks: {len(tracks)}")
    
    # Cleanup
    cap.release()
    writer.release()
    
    processing_time = time.time() - start_time
    fps_achieved = frame_count / processing_time
    
    # Results
    results = {
        'video_path': video_path,
        'output_path': str(output_path),
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'max_tracks': total_tracks,
        'processing_time': processing_time,
        'fps_achieved': fps_achieved
    }
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ HOÀN THÀNH!")
    print(f"📹 Frames: {frame_count}")
    print(f"🎯 Detections: {total_detections}")
    print(f"👥 Max tracks: {total_tracks}")
    print(f"⏱️  Time: {processing_time:.2f}s")
    print(f"⚡ FPS: {fps_achieved:.1f}")
    print(f"💾 Output: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test YOLO + DeepSORT')
    parser.add_argument('--video', '-v', required=True, help='Video path')
    parser.add_argument('--output', '-o', default='test_output', help='Output dir')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Video không tồn tại: {args.video}")
        return
    
    test_single_video(args.video, args.output)

if __name__ == "__main__":
    main()
