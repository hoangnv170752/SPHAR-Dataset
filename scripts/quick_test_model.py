#!/usr/bin/env python3
"""
Quick test script for fine-tuned YOLO model
Simple detection without tracking - with confidence scores
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse

def test_on_image(model_path, image_path, output_path=None, conf=0.25):
    """Test model on single image"""
    print(f"🔥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📸 Processing image: {image_path}")
    results = model(image_path, conf=conf)
    
    # Show results
    for result in results:
        print(f"✅ Found {len(result.boxes)} humans")
        
        # Plot and show
        annotated = result.plot()
        
        if output_path:
            cv2.imwrite(str(output_path), annotated)
            print(f"💾 Saved to: {output_path}")
        
        cv2.imshow('Detection Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_on_video(model_path, video_path, output_path=None, conf=0.25):
    """Test model on video"""
    print(f"🔥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📹 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=conf, verbose=False)
            
            # Annotate
            for result in results:
                annotated = result.plot()
                
                if writer:
                    writer.write(annotated)
                
                cv2.imshow('Detection', annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Processed {frame_count} frames")
        if output_path:
            print(f"💾 Saved to: {output_path}")

def test_on_webcam(model_path, conf=0.25):
    """Test model on webcam"""
    print(f"🔥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("📹 Starting webcam... Press 'q' to quit")
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, conf=conf, verbose=False)
            
            # Show
            for result in results:
                annotated = result.plot()
                cv2.imshow('Webcam Detection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Quick test for fine-tuned YOLO model')
    parser.add_argument('--model', '-m',
                       default=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                       help='Path to model')
    parser.add_argument('--source', '-s',
                       help='Image/video path or "webcam"')
    parser.add_argument('--output', '-o',
                       help='Output path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not args.source:
        print("❌ Please specify --source")
        print("\nExamples:")
        print("  python quick_test_model.py --source webcam")
        print("  python quick_test_model.py --source image.jpg")
        print("  python quick_test_model.py --source video.mp4 --output result.mp4")
        return
    
    source = Path(args.source) if args.source != 'webcam' else args.source
    
    if args.source == 'webcam':
        test_on_webcam(args.model, args.conf)
    elif source.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        test_on_image(args.model, source, args.output, args.conf)
    else:
        test_on_video(args.model, source, args.output, args.conf)

if __name__ == "__main__":
    main()
