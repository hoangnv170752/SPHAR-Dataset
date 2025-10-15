#!/usr/bin/env python3
"""
Helper script to test IITB-Corridor surveillance videos
"""

import subprocess
import sys
from pathlib import Path

def list_iitb_videos():
    """List all available IITB-Corridor videos"""
    iitb_dir = Path(r'D:\SPHAR-Dataset\videos\IITB-Corridor')
    
    if not iitb_dir.exists():
        print("❌ IITB-Corridor directory not found")
        return []
    
    videos = []
    for video_dir in sorted(iitb_dir.iterdir()):
        if video_dir.is_dir():
            video_file = video_dir / f"{video_dir.name}.avi"
            if video_file.exists():
                videos.append({
                    'id': video_dir.name,
                    'path': video_file,
                    'size': video_file.stat().st_size / (1024 * 1024)  # MB
                })
    
    return videos

def print_video_list(videos):
    """Print formatted list of videos"""
    print("\n" + "="*80)
    print("📹 IITB-CORRIDOR SURVEILLANCE VIDEOS")
    print("="*80)
    print(f"Total videos: {len(videos)}\n")
    
    # Print in groups of 10
    for i in range(0, len(videos), 10):
        group = videos[i:i+10]
        ids = [v['id'] for v in group]
        print(f"  {', '.join(ids)}")
    
    print("\n" + "="*80)
    print("\n💡 Usage:")
    print("  python test_iitb_corridor.py <video_id>")
    print("  python test_iitb_corridor.py 000209")
    print("  python test_iitb_corridor.py 000220 --output tracked.mp4")
    print("\nOr use test_yolo_deepsort.py directly:")
    print("  python test_yolo_deepsort.py --source 000209")
    print("  python test_yolo_deepsort.py --source iitb:000220")

def test_video(video_id, output=None, conf=0.3, display=True):
    """Test specific IITB-Corridor video"""
    videos = list_iitb_videos()
    
    # Find video
    video_info = None
    for v in videos:
        if v['id'] == video_id:
            video_info = v
            break
    
    if not video_info:
        print(f"❌ Video ID '{video_id}' not found")
        print(f"\nAvailable IDs: {videos[0]['id']} to {videos[-1]['id']}")
        return
    
    print("="*80)
    print(f"🎬 Testing IITB-Corridor Video: {video_id}")
    print("="*80)
    print(f"📹 Video: {video_info['path'].name}")
    print(f"💾 Size: {video_info['size']:.1f} MB")
    print("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "test_yolo_deepsort.py",
        "--source", str(video_info['path']),
        "--conf", str(conf)
    ]
    
    if output:
        cmd.extend(["--output", output])
    
    if not display:
        cmd.append("--no-display")
    
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    # Run
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")

def test_random_sample(count=3, conf=0.3):
    """Test random sample of videos"""
    import random
    
    videos = list_iitb_videos()
    
    if count > len(videos):
        count = len(videos)
    
    sample = random.sample(videos, count)
    
    print("="*80)
    print(f"🎲 Testing {count} random IITB-Corridor videos")
    print("="*80)
    
    for i, video in enumerate(sample, 1):
        print(f"\n[{i}/{count}] Testing video: {video['id']}")
        test_video(video['id'], conf=conf, display=False)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test IITB-Corridor surveillance videos')
    parser.add_argument('video_id', nargs='?', 
                       help='Video ID (e.g., 000209)')
    parser.add_argument('--list', action='store_true',
                       help='List all available videos')
    parser.add_argument('--output', '-o',
                       help='Output video path')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video')
    parser.add_argument('--random', type=int, metavar='N',
                       help='Test N random videos')
    
    args = parser.parse_args()
    
    videos = list_iitb_videos()
    
    if not videos:
        print("❌ No IITB-Corridor videos found")
        return
    
    if args.list:
        print_video_list(videos)
    elif args.random:
        test_random_sample(args.random, args.conf)
    elif args.video_id:
        test_video(args.video_id, args.output, args.conf, not args.no_display)
    else:
        print_video_list(videos)

if __name__ == "__main__":
    main()
