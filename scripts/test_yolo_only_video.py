#!/usr/bin/env python3
"""
Test YOLO detection cho 1 video (không dùng DeepSORT)
"""

import argparse
import cv2
import time
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
from yolo_detect import YOLOHumanDetector

def test_yolo_video(video_path, output_dir="yolo_test_output"):
    """Test YOLO detection cho 1 video"""
    
    print(f"🎬 Testing YOLO on video: {video_path}")
    
    # Khởi tạo YOLO
    detector = YOLOHumanDetector(model_path='yolov8n.pt', device='cuda')
    
    # Tạo output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    output_path = output_dir / f"yolo_detected_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
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
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw confidence
                cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Detections: {len(detections) if detections else 0}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
        
        if frame_count % 30 == 0:
            print(f"📹 Frame {frame_count}, Detections: {len(detections) if detections else 0}")
    
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
        'processing_time': processing_time,
        'fps_achieved': fps_achieved,
        'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0
    }
    
    # Save results
    with open(output_dir / 'yolo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ YOLO TEST HOÀN THÀNH!")
    print(f"📹 Frames: {frame_count}")
    print(f"🎯 Total detections: {total_detections}")
    print(f"📊 Avg detections/frame: {total_detections/frame_count:.2f}")
    print(f"⏱️  Time: {processing_time:.2f}s")
    print(f"⚡ FPS: {fps_achieved:.1f}")
    print(f"💾 Output: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test YOLO detection only')
    parser.add_argument('--video', '-v', required=True, help='Video path')
    parser.add_argument('--output', '-o', default='yolo_test_output', help='Output dir')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"❌ Video không tồn tại: {args.video}")
        return
    
    test_yolo_video(args.video, args.output)

if __name__ == "__main__":
    main()
