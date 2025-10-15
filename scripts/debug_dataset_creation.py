#!/usr/bin/env python3
"""
Debug script to check dataset creation issues and test with sample data.
"""

import os
import cv2
import json
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np

def check_videos_directory(videos_dir):
    """Check videos directory structure and content"""
    videos_path = Path(videos_dir)
    print(f"Checking videos directory: {videos_path}")
    
    if not videos_path.exists():
        print(f"❌ Videos directory does not exist: {videos_path}")
        return False
    
    print(f"✅ Videos directory exists")
    
    # Check subdirectories
    categories = []
    total_videos = 0
    
    for category_dir in videos_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            
            # Count video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(category_dir.glob(ext))
            
            categories.append({
                'name': category_name,
                'path': category_dir,
                'video_count': len(video_files),
                'videos': [v.name for v in video_files[:3]]  # Show first 3
            })
            
            total_videos += len(video_files)
            
            print(f"  📁 {category_name}: {len(video_files)} videos")
            if video_files:
                print(f"    Sample files: {', '.join([v.name for v in video_files[:3]])}")
    
    print(f"\\n📊 Summary:")
    print(f"  Total categories: {len(categories)}")
    print(f"  Total videos: {total_videos}")
    
    if total_videos == 0:
        print("\\n❌ No video files found!")
        print("Possible solutions:")
        print("1. Check if videos are in the correct directories")
        print("2. Ensure video files have supported extensions (.mp4, .avi, .mov, .mkv)")
        print("3. Verify file permissions")
        return False
    
    return True, categories

def test_yolo_model_loading():
    """Test YOLO model loading"""
    print("\\n🔍 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Try to find local model first
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / 'models' / 'yolo11n.pt'
        
        if model_path.exists():
            print(f"✅ Found local model: {model_path}")
            model = YOLO(str(model_path))
            print("✅ Local model loaded successfully")
            return True, str(model_path)
        else:
            print("⚠️ Local model not found, trying to download...")
            try:
                model = YOLO('yolo11n.pt')
                print("✅ Model downloaded and loaded successfully")
                return True, 'yolo11n.pt'
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return False, None
                
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return False, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False, None

def create_sample_video(output_path, duration=5, fps=30):
    """Create a sample video for testing"""
    print(f"\\n🎬 Creating sample video: {output_path}")
    
    try:
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            # Create a simple animated frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some moving elements
            t = frame_num / fps
            
            # Moving circle (represents a person)
            center_x = int(width * 0.3 + 100 * np.sin(t))
            center_y = int(height * 0.5 + 50 * np.cos(t))
            cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
            
            # Add some text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Sample video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample video: {e}")
        return False

def create_sample_dataset(base_dir):
    """Create sample videos for testing"""
    print("\\n🏗️ Creating sample dataset for testing...")
    
    base_path = Path(base_dir)
    sample_videos_dir = base_path / 'sample_videos'
    
    # Create sample categories
    categories = ['neutral', 'walking', 'sitting']
    
    success_count = 0
    
    for category in categories:
        category_dir = sample_videos_dir / category
        
        # Create 2 sample videos per category
        for i in range(2):
            video_path = category_dir / f'sample_{category}_{i+1}.mp4'
            if create_sample_video(video_path, duration=3, fps=10):
                success_count += 1
    
    if success_count > 0:
        print(f"✅ Created {success_count} sample videos in {sample_videos_dir}")
        return str(sample_videos_dir)
    else:
        print("❌ Failed to create sample videos")
        return None

def test_dataset_creation_minimal(videos_dir, output_dir):
    """Test dataset creation with minimal processing"""
    print(f"\\n🧪 Testing minimal dataset creation...")
    print(f"Source: {videos_dir}")
    print(f"Output: {output_dir}")
    
    try:
        # Import the dataset creator
        import sys
        sys.path.append(str(Path(__file__).parent))
        
        from create_human_detection_dataset import HumanDetectionDatasetCreator
        
        # Create with minimal settings
        creator = HumanDetectionDatasetCreator(
            source_dir=videos_dir,
            output_dir=output_dir,
            frame_interval=60,  # Extract fewer frames
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        print("✅ Dataset creator initialized")
        
        # Test video collection
        print("Testing video collection...")
        all_frames_data = creator.process_all_videos()
        
        if not all_frames_data:
            print("❌ No frames extracted")
            return False
        
        print(f"✅ Extracted {len(all_frames_data)} frames")
        
        # Test splitting
        print("Testing data splitting...")
        splits = creator.split_frames_data(all_frames_data)
        
        print(f"✅ Data split completed:")
        for split_name, frames in splits.items():
            print(f"  {split_name}: {len(frames)} frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug dataset creation issues')
    parser.add_argument('--videos-dir', default=r'D:\\SPHAR-Dataset\\videos',
                       help='Videos directory to check')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample videos for testing')
    parser.add_argument('--test-creation', action='store_true',
                       help='Test dataset creation process')
    parser.add_argument('--output-dir', default=r'D:\\SPHAR-Dataset\\debug_output',
                       help='Output directory for testing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HUMAN DETECTION DATASET DEBUG")
    print("="*60)
    
    # Step 1: Check videos directory
    videos_exist, categories = check_videos_directory(args.videos_dir)
    
    # Step 2: Test YOLO model loading
    model_ok, model_path = test_yolo_model_loading()
    
    # Step 3: Create sample data if requested or if no videos found
    sample_videos_dir = None
    if args.create_samples or not videos_exist:
        sample_videos_dir = create_sample_dataset(Path(args.videos_dir).parent)
    
    # Step 4: Test dataset creation if requested
    if args.test_creation:
        test_dir = sample_videos_dir if sample_videos_dir else args.videos_dir
        if test_dir:
            test_dataset_creation_minimal(test_dir, args.output_dir)
    
    # Summary
    print("\\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    print(f"Videos directory OK: {'✅' if videos_exist else '❌'}")
    print(f"YOLO model OK: {'✅' if model_ok else '❌'}")
    if sample_videos_dir:
        print(f"Sample videos created: ✅ {sample_videos_dir}")
    
    if videos_exist and model_ok:
        print("\\n✅ Everything looks good! Try running the pipeline again.")
    else:
        print("\\n❌ Issues found. Please fix the problems above.")
        
        if not videos_exist:
            print("\\nTo fix video issues:")
            print("1. Add video files to the category directories")
            print("2. Or run with --create-samples to create test videos")
        
        if not model_ok:
            print("\\nTo fix model issues:")
            print("1. Check internet connection")
            print("2. Or download yolo11n.pt manually to models/ directory")

if __name__ == "__main__":
    main()
