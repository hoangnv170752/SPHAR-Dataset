#!/usr/bin/env python3
"""
Test YOLO Detection on Sample Images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_yolo_detection():
    """Test YOLO detection on sample images"""
    
    print("🔍 TESTING YOLO DETECTION")
    print("="*50)
    
    # Load YOLO model
    yolo_paths = [
        Path(r'D:\SPHAR-Dataset\models\yolo11n.pt'),
        Path(r'D:\SPHAR-Dataset\models\yolov8n.pt'),
        Path('yolo11n.pt')
    ]
    
    model = None
    for yolo_path in yolo_paths:
        try:
            if yolo_path.exists():
                model = YOLO(str(yolo_path))
                print(f"✅ YOLO loaded: {yolo_path.name}")
                break
        except:
            continue
    
    if model is None:
        print("❌ No YOLO model found")
        return
    
    # Get sample images from videos
    video_dirs = [Path(r'D:\SPHAR-Dataset\videos')]
    
    test_frames = []
    for video_dir in video_dirs:
        if video_dir.exists():
            for action_dir in video_dir.iterdir():
                if action_dir.is_dir():
                    videos = list(action_dir.glob('*.mp4'))[:2]
                    
                    for video_path in videos:
                        cap = cv2.VideoCapture(str(video_path))
                        
                        # Get frame from middle
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 50:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.resize(frame, (640, 480))
                                test_frames.append((frame, f"{video_path.name}"))
                        
                        cap.release()
                        
                        if len(test_frames) >= 4:  # Test 4 frames
                            break
                
                if len(test_frames) >= 4:
                    break
    
    print(f"📸 Testing {len(test_frames)} frames...")
    
    # Test each frame with different confidence thresholds
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (frame, name) in enumerate(test_frames[:4]):
        print(f"\n🔍 Testing frame {i+1}: {name}")
        
        # Test with multiple confidence thresholds
        for conf_thresh in [0.05, 0.1, 0.2, 0.3, 0.5]:
            results = model(frame, conf=conf_thresh, verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0:  # Person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf)
                            persons.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
            
            print(f"   Conf {conf_thresh}: {len(persons)} persons")
            for person in persons:
                print(f"      - Conf: {person['confidence']:.3f}, BBox: {person['bbox']}")
        
        # Display frame with best detection (lowest threshold)
        results = model(frame, conf=0.05, verbose=False)
        
        ax = axes[i]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{name}\nYOLO Detection Results', fontsize=10)
        ax.axis('off')
        
        # Draw all detections
        detection_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Color code by class
                    if cls == 0:  # Person
                        color = 'red'
                        label = f'Person: {conf:.2f}'
                        detection_count += 1
                    else:
                        color = 'blue'
                        label = f'Class{cls}: {conf:.2f}'
                    
                    # Draw bounding box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=2, edgecolor=color,
                                           facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(x1, y1-5, label, fontsize=8, color=color,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add detection summary
        ax.text(10, 30, f'Persons: {detection_count}', fontsize=12, color='red',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ YOLO Detection Test Complete!")
    print("📊 Check the visualization to see detection results")

if __name__ == "__main__":
    test_yolo_detection()
