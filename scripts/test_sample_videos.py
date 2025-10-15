#!/usr/bin/env python3
"""
Test tracking on sample videos from dataset
"""

import subprocess
import sys
from pathlib import Path

# Sample videos with descriptions
SAMPLE_VIDEOS = {
    "2people_meet": {
        "path": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01p02_meettogether_a1.mp4",
        "desc": "2 people meeting (Best for testing ID switching)"
    },
    "2people_follow": {
        "path": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01p02_followtogether_a1.mp4",
        "desc": "2 people following (Test persistent tracking)"
    },
    "2people_overtake": {
        "path": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01p02_overtake_a1.mp4",
        "desc": "2 people overtaking (Test occlusion handling)"
    },
    "1person_walk": {
        "path": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01_walk_a1.mp4",
        "desc": "Single person walking (Simple test)"
    },
    "ntu_action": {
        "path": r"D:\SPHAR-Dataset\videos\NTU\A001\S001C001P001R001A001_rgb.avi",
        "desc": "NTU action dataset (Always has person)"
    }
}

def test_video(video_key, conf=0.3, display=True):
    """Test tracking on a specific video"""
    if video_key not in SAMPLE_VIDEOS:
        print(f"❌ Unknown video: {video_key}")
        print(f"Available: {list(SAMPLE_VIDEOS.keys())}")
        return
    
    video_info = SAMPLE_VIDEOS[video_key]
    video_path = Path(video_info["path"])
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("="*70)
    print(f"🎬 Testing: {video_info['desc']}")
    print(f"📹 Video: {video_path.name}")
    print("="*70)
    
    # Output path
    output_dir = Path(r"D:\SPHAR-Dataset\output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tracked_{video_key}.mp4"
    
    # Build command
    cmd = [
        sys.executable,
        "test_yolo_deepsort.py",
        "--source", str(video_path),
        "--output", str(output_path),
        "--conf", str(conf)
    ]
    
    if not display:
        cmd.append("--no-display")
    
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    # Run
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Output saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tracking on sample videos')
    parser.add_argument('video', nargs='?', default='2people_meet',
                       choices=list(SAMPLE_VIDEOS.keys()),
                       help='Sample video to test')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video')
    parser.add_argument('--list', action='store_true',
                       help='List available videos')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📹 Available sample videos:")
        print("="*70)
        for key, info in SAMPLE_VIDEOS.items():
            print(f"  {key:20s} - {info['desc']}")
            print(f"  {'':20s}   {Path(info['path']).name}")
        print("="*70)
        return
    
    test_video(args.video, args.conf, not args.no_display)

if __name__ == "__main__":
    main()
