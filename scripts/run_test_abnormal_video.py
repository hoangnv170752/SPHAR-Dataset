#!/usr/bin/env python3
"""
Script test YOLO + DeepSORT với video abnormal
"""

import argparse
import random
from pathlib import Path
import subprocess
import sys

def find_abnormal_videos(dataset_path):
    """Tìm videos abnormal"""
    dataset_path = Path(dataset_path)
    videos = []
    
    train_path = dataset_path / 'videos' / 'train'
    
    # Tìm abnormal videos
    for category in ['abnormal_physical', 'abnormal_biological', 'abnormal']:
        cat_dir = train_path / category
        if cat_dir.exists():
            for video in cat_dir.glob('*.mp4'):
                videos.append({'path': video, 'category': category})
    
    return videos

def main():
    parser = argparse.ArgumentParser(description='Test YOLO + DeepSORT')
    parser.add_argument('--dataset', '-d', 
                       default='/home/nguyenhoang/Downloads/abnormal_detection_dataset')
    parser.add_argument('--video', '-v', help='Video path cụ thể')
    
    args = parser.parse_args()
    
    if args.video:
        video_path = args.video
    else:
        # Tìm video abnormal ngẫu nhiên
        videos = find_abnormal_videos(args.dataset)
        if not videos:
            print("❌ Không tìm thấy video abnormal")
            return
        
        video_info = random.choice(videos)
        video_path = str(video_info['path'])
        print(f"🎯 Chọn video: {video_info['category']} - {Path(video_path).name}")
    
    # Chạy test
    cmd = [
        sys.executable,
        "scripts/test_yolo_deepsort_single_video.py",
        "--video", video_path,
        "--output", "abnormal_test_output"
    ]
    
    print(f"🚀 Chạy test: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
