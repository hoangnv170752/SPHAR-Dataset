#!/usr/bin/env python3
"""
Quick demo script to test YOLO + DeepSORT tracking
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_yolo_deepsort import YOLODeepSORTTracker

def demo_webcam():
    """Demo on webcam"""
    print("="*70)
    print("🚀 YOLO + DEEPSORT TRACKING DEMO - WEBCAM")
    print("="*70)
    print("\n📋 Features:")
    print("  ✅ Real-time human detection")
    print("  ✅ Multi-person tracking with unique IDs")
    print("  ✅ Trajectory visualization (đường đi)")
    print("  ✅ Track statistics")
    print("\n🎮 Controls:")
    print("  Press 'q' to quit")
    print("  Press 'p' to pause")
    print("\n" + "="*70 + "\n")
    
    # Initialize tracker
    tracker = YOLODeepSORTTracker(
        model_path=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
        conf_threshold=0.3,  # Higher confidence for webcam
        max_age=50  # Keep tracks longer
    )
    
    # Run webcam tracking
    tracker.process_webcam()

def demo_video():
    """Demo on video file"""
    print("="*70)
    print("🚀 YOLO + DEEPSORT TRACKING DEMO - VIDEO")
    print("="*70)
    
    # Find a sample video
    videos_dir = Path(r'D:\SPHAR-Dataset\videos')
    sample_videos = list(videos_dir.rglob('*.mp4'))[:5] + list(videos_dir.rglob('*.avi'))[:5]
    
    if not sample_videos:
        print("\n❌ No sample videos found in D:\\SPHAR-Dataset\\videos")
        print("Please provide video path manually:")
        print('  python demo_tracking.py --video "path/to/video.mp4"')
        return
    
    print("\n📹 Sample videos available:")
    for i, video in enumerate(sample_videos[:5], 1):
        print(f"  {i}. {video.relative_to(videos_dir)}")
    
    print("\nSelect video (1-5) or press Enter to use first: ", end='')
    try:
        choice = input().strip()
        idx = int(choice) - 1 if choice else 0
        video_path = sample_videos[idx]
    except (ValueError, IndexError):
        video_path = sample_videos[0]
    
    print(f"\n✅ Selected: {video_path.name}")
    print("="*70 + "\n")
    
    # Initialize tracker
    tracker = YOLODeepSORTTracker(
        model_path=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
        conf_threshold=0.25,
        max_age=30
    )
    
    # Output path
    output_path = Path(r'D:\SPHAR-Dataset\output') / f'tracked_{video_path.name}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run video tracking
    tracker.process_video(
        video_path=video_path,
        output_path=output_path,
        display=True
    )

def print_usage():
    """Print usage information"""
    print("="*70)
    print("🎯 YOLO + DEEPSORT TRACKING DEMO")
    print("="*70)
    print("\nUsage:")
    print("  python demo_tracking.py                    # Demo on webcam")
    print("  python demo_tracking.py --webcam           # Demo on webcam")
    print("  python demo_tracking.py --video            # Demo on sample video")
    print('  python demo_tracking.py --video "path"     # Demo on specific video')
    print("\nWhat you'll see:")
    print("  🟦 Bounding boxes with unique colors for each person")
    print("  🔢 Person ID (e.g., 'Person #1', 'Person #2')")
    print("  📊 Number of frames tracked")
    print("  🎨 Trajectory lines showing movement path")
    print("  📈 Real-time statistics (FPS, active tracks, etc.)")
    print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO + DeepSORT Tracking Demo')
    parser.add_argument('--webcam', action='store_true', help='Demo on webcam')
    parser.add_argument('--video', nargs='?', const=True, help='Demo on video (optional: path)')
    
    args = parser.parse_args()
    
    if args.video:
        if isinstance(args.video, str):
            # Specific video provided
            from test_yolo_deepsort import YOLODeepSORTTracker
            tracker = YOLODeepSORTTracker(
                model_path=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                conf_threshold=0.25
            )
            tracker.process_video(args.video, display=True)
        else:
            demo_video()
    elif args.webcam:
        demo_webcam()
    else:
        # Default: show usage and run webcam
        print_usage()
        print("\nNo arguments provided. Starting webcam demo...")
        print("(Use --help for more options)\n")
        import time
        time.sleep(2)
        demo_webcam()

if __name__ == "__main__":
    main()
