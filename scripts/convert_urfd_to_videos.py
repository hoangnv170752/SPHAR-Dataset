#!/usr/bin/env python3
"""
Convert URFD image sequences to video files for action recognition training
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

def analyze_urfd_sequences():
    """Analyze URFD dataset and find fall sequences"""
    urfd_dir = Path(r'D:\SPHAR-Dataset\videos\URFD')
    
    print("="*80)
    print("📊 ANALYZING URFD DATASET")
    print("="*80)
    
    # Collect all fall images
    fall_sequences = defaultdict(list)
    
    for split in ['train', 'valid', 'test']:
        images_dir = urfd_dir / split / 'images'
        if images_dir.exists():
            fall_images = list(images_dir.glob('fall-*.jpg'))
            
            print(f"📁 {split}: {len(fall_images)} fall images")
            
            # Group by sequence ID
            for img_path in fall_images:
                try:
                    # Parse filename: fall-01-cam0-rgb-012_png.rf.xxx.jpg
                    filename = img_path.stem
                    parts = filename.split('-')
                    
                    if len(parts) >= 2 and parts[0] == 'fall':
                        seq_id = parts[1]  # Get '01', '02', etc.
                        
                        # Extract frame number from end
                        frame_part = parts[-1].split('_')[0]
                        if frame_part.isdigit():
                            frame_num = int(frame_part)
                            fall_sequences[seq_id].append({
                                'path': img_path,
                                'frame_num': frame_num,
                                'split': split
                            })
                except Exception as e:
                    print(f"⚠️ Error parsing {img_path.name}: {e}")
                    continue
    
    print(f"\n📊 Found {len(fall_sequences)} fall sequences:")
    
    valid_sequences = {}
    for seq_id, frames in fall_sequences.items():
        if len(frames) >= 8:  # Need at least 8 frames
            # Sort by frame number
            frames.sort(key=lambda x: x['frame_num'])
            valid_sequences[seq_id] = frames
            print(f"   Sequence {seq_id}: {len(frames)} frames (frames {frames[0]['frame_num']}-{frames[-1]['frame_num']})")
    
    print(f"\n✅ Valid sequences: {len(valid_sequences)}")
    return valid_sequences

def create_video_from_sequence(sequence_frames, output_path, fps=10):
    """Create video from image sequence"""
    if not sequence_frames:
        return False
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(str(sequence_frames[0]['path']))
    if first_frame is None:
        return False
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame_info in sequence_frames:
        frame = cv2.imread(str(frame_info['path']))
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    return True

def convert_urfd_to_videos(output_dir, fps=10, min_frames=8):
    """Convert URFD sequences to video files"""
    
    # Analyze sequences
    sequences = analyze_urfd_sequences()
    
    if not sequences:
        print("❌ No valid sequences found")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*80)
    print(f"🎬 CONVERTING TO VIDEOS")
    print(f"="*80)
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 FPS: {fps}")
    print(f"📊 Min frames: {min_frames}")
    
    converted_videos = []
    
    # Convert each sequence
    for seq_id, frames in tqdm(sequences.items(), desc="Converting sequences"):
        if len(frames) < min_frames:
            continue
        
        # Create output filename
        output_filename = f"urfd_fall_sequence_{seq_id}.mp4"
        output_path = output_dir / output_filename
        
        # Convert to video
        success = create_video_from_sequence(frames, output_path, fps)
        
        if success:
            converted_videos.append(output_path)
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {output_filename}: {len(frames)} frames, {file_size:.1f} MB")
        else:
            print(f"❌ Failed to convert sequence {seq_id}")
    
    print(f"\n" + "="*80)
    print(f"📊 CONVERSION SUMMARY")
    print(f"="*80)
    print(f"✅ Successfully converted: {len(converted_videos)} videos")
    print(f"📁 Total sequences found: {len(sequences)}")
    print(f"💾 Output directory: {output_dir}")
    
    # Calculate total size
    total_size = sum(video.stat().st_size for video in converted_videos) / (1024 * 1024)
    print(f"💿 Total size: {total_size:.1f} MB")
    
    return converted_videos

def move_videos_to_falling_folder(converted_videos):
    """Move converted videos to falling folder for training"""
    falling_dir = Path(r'D:\SPHAR-Dataset\videos\falling')
    
    if not falling_dir.exists():
        print(f"❌ Falling directory not found: {falling_dir}")
        return
    
    print(f"\n" + "="*80)
    print(f"📦 MOVING TO FALLING FOLDER")
    print(f"="*80)
    
    moved_count = 0
    
    for video_path in converted_videos:
        target_path = falling_dir / video_path.name
        
        try:
            # Move video to falling folder
            video_path.rename(target_path)
            moved_count += 1
            print(f"📁 Moved: {video_path.name} → falling/")
        except Exception as e:
            print(f"❌ Error moving {video_path.name}: {e}")
    
    print(f"\n✅ Moved {moved_count} videos to falling folder")
    print(f"📁 Location: {falling_dir}")
    
    # List falling folder contents
    falling_videos = list(falling_dir.glob('*.mp4')) + list(falling_dir.glob('*.avi'))
    print(f"📊 Total videos in falling folder: {len(falling_videos)}")

def create_urfd_summary():
    """Create summary of URFD conversion"""
    falling_dir = Path(r'D:\SPHAR-Dataset\videos\falling')
    urfd_videos = list(falling_dir.glob('urfd_fall_sequence_*.mp4'))
    
    print(f"\n" + "="*80)
    print(f"📋 URFD INTEGRATION SUMMARY")
    print(f"="*80)
    
    print(f"📊 URFD videos in falling folder: {len(urfd_videos)}")
    
    if urfd_videos:
        print(f"\n🎬 URFD Fall Videos:")
        for video in sorted(urfd_videos):
            file_size = video.stat().st_size / (1024 * 1024)
            print(f"   📹 {video.name} ({file_size:.1f} MB)")
    
    # Check total falling videos
    all_falling_videos = list(falling_dir.glob('*.mp4')) + list(falling_dir.glob('*.avi'))
    original_videos = [v for v in all_falling_videos if not v.name.startswith('urfd_')]
    
    print(f"\n📈 Falling Dataset Summary:")
    print(f"   Original videos: {len(original_videos)}")
    print(f"   URFD videos: {len(urfd_videos)}")
    print(f"   Total: {len(all_falling_videos)}")
    
    improvement = (len(urfd_videos) / len(original_videos)) * 100 if original_videos else 0
    print(f"   📊 Dataset increase: +{improvement:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Convert URFD image sequences to videos')
    parser.add_argument('--output-dir', default=r'D:\SPHAR-Dataset\temp_urfd_videos',
                       help='Temporary output directory for converted videos')
    parser.add_argument('--fps', type=int, default=10,
                       help='FPS for output videos')
    parser.add_argument('--min-frames', type=int, default=8,
                       help='Minimum frames required for a valid sequence')
    parser.add_argument('--no-move', action='store_true',
                       help='Do not move videos to falling folder')
    
    args = parser.parse_args()
    
    print("🎬 URFD TO VIDEO CONVERTER")
    print("="*80)
    print("📋 This script converts URFD image sequences to video files")
    print("🎯 Target: Add fall detection videos to training dataset")
    
    # Step 1: Convert sequences to videos
    converted_videos = convert_urfd_to_videos(args.output_dir, args.fps, args.min_frames)
    
    if not converted_videos:
        print("❌ No videos were converted")
        return
    
    # Step 2: Move to falling folder (unless disabled)
    if not args.no_move:
        move_videos_to_falling_folder(converted_videos)
        
        # Clean up temp directory
        temp_dir = Path(args.output_dir)
        if temp_dir.exists() and not list(temp_dir.iterdir()):
            temp_dir.rmdir()
            print(f"🧹 Cleaned up temporary directory")
    
    # Step 3: Create summary
    create_urfd_summary()
    
    print(f"\n" + "="*80)
    print(f"✅ URFD CONVERSION COMPLETED")
    print(f"="*80)
    print(f"🎯 Next steps:")
    print(f"   1. Run dataset organization:")
    print(f"      python run_action_training.py --organize-only")
    print(f"   ")
    print(f"   2. Check fall class data increase:")
    print(f"      Check dataset_info.json for updated fall video count")
    print(f"   ")
    print(f"   3. Start training:")
    print(f"      python run_action_training.py --train-only --epochs 50")

if __name__ == "__main__":
    main()
