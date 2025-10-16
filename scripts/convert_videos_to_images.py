#!/usr/bin/env python3
"""
Convert videos to image sequences for faster training
Solves 'moov atom not found' errors and improves GPU utilization
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoToImageConverter:
    """Convert videos to image sequences for faster training"""
    
    def __init__(self, sequence_length=8, target_fps=5):
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        
    def extract_frames_from_video(self, video_path, output_dir, max_frames=None):
        """Extract frames from video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path}")
                return []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                logger.warning(f"Invalid video properties: {video_path}")
                cap.release()
                return []
            
            # Calculate frame sampling
            if max_frames and total_frames > max_frames:
                # Sample frames uniformly
                frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            else:
                # Take all frames or sample at target FPS
                step = max(1, int(fps / self.target_fps))
                frame_indices = list(range(0, total_frames, step))
            
            # Extract frames
            extracted_frames = []
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize to standard size for faster processing
                    frame = cv2.resize(frame, (224, 224))
                    
                    # Save frame
                    frame_filename = f"frame_{i:04d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    extracted_frames.append(frame_path)
                
                if len(extracted_frames) >= self.sequence_length * 2:  # Extract more for variety
                    break
            
            cap.release()
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return []
    
    def process_urfd_sequence(self, urfd_path, output_dir):
        """Process URFD pseudo video files"""
        try:
            # Read URFD file to get image paths
            with open(urfd_path, 'r') as f:
                lines = f.readlines()
            
            # Extract image paths from URFD file
            image_paths = []
            for line in lines[2:]:  # Skip header lines
                line = line.strip()
                if line and Path(line).exists():
                    image_paths.append(Path(line))
            
            if not image_paths:
                return []
            
            # Sort by frame number
            try:
                image_paths.sort(key=lambda x: int(x.stem.split('-')[-1].split('_')[0]))
            except:
                pass  # If sorting fails, use as is
            
            # Copy and resize images
            extracted_frames = []
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img_path in enumerate(image_paths[:self.sequence_length * 2]):
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    # Resize to standard size
                    frame = cv2.resize(frame, (224, 224))
                    
                    # Save frame
                    frame_filename = f"frame_{i:04d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    extracted_frames.append(frame_path)
            
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Error processing URFD {urfd_path}: {e}")
            return []
    
    def convert_dataset(self, input_dir, output_dir):
        """Convert entire dataset to image sequences"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        logger.info(f"Converting dataset from {input_dir} to {output_dir}")
        
        # Track conversion statistics
        conversion_stats = {
            'total_videos': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_frames': 0
        }
        
        # Process each split and action
        for split in ['train', 'val', 'test']:
            split_dir = input_dir / split
            if not split_dir.exists():
                continue
                
            for action_dir in split_dir.iterdir():
                if not action_dir.is_dir():
                    continue
                
                action_name = action_dir.name
                output_action_dir = output_dir / split / action_name
                output_action_dir.mkdir(parents=True, exist_ok=True)
                
                # Get all video files
                video_files = []
                for ext in ['*.mp4', '*.avi', '*.urfd']:
                    video_files.extend(list(action_dir.glob(ext)))
                
                logger.info(f"Processing {len(video_files)} videos in {split}/{action_name}")
                
                # Process videos with progress bar
                for video_idx, video_path in enumerate(tqdm(video_files, desc=f"{split}/{action_name}")):
                    conversion_stats['total_videos'] += 1
                    
                    # Create output directory for this video
                    video_output_dir = output_action_dir / f"video_{video_idx:04d}"
                    
                    # Extract frames
                    if video_path.suffix == '.urfd':
                        extracted_frames = self.process_urfd_sequence(video_path, video_output_dir)
                    else:
                        extracted_frames = self.extract_frames_from_video(
                            video_path, video_output_dir, max_frames=self.sequence_length * 2
                        )
                    
                    if len(extracted_frames) >= self.sequence_length:
                        conversion_stats['successful_conversions'] += 1
                        conversion_stats['total_frames'] += len(extracted_frames)
                    else:
                        conversion_stats['failed_conversions'] += 1
                        # Remove empty directory
                        if video_output_dir.exists():
                            shutil.rmtree(video_output_dir)
        
        # Save conversion statistics
        stats_file = output_dir / 'conversion_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(conversion_stats, f, indent=2)
        
        logger.info("Conversion completed!")
        logger.info(f"Total videos: {conversion_stats['total_videos']}")
        logger.info(f"Successful: {conversion_stats['successful_conversions']}")
        logger.info(f"Failed: {conversion_stats['failed_conversions']}")
        logger.info(f"Total frames: {conversion_stats['total_frames']}")
        
        return conversion_stats

def main():
    parser = argparse.ArgumentParser(description='Convert videos to image sequences')
    parser.add_argument('--input-dir', default=r'D:\SPHAR-Dataset\action_recognition_optimized',
                       help='Input directory with organized videos')
    parser.add_argument('--output-dir', default=r'D:\SPHAR-Dataset\action_recognition_images',
                       help='Output directory for image sequences')
    parser.add_argument('--sequence-length', type=int, default=8,
                       help='Target sequence length')
    parser.add_argument('--target-fps', type=int, default=5,
                       help='Target FPS for frame extraction')
    
    args = parser.parse_args()
    
    print("🎬 VIDEO TO IMAGE CONVERTER")
    print("="*80)
    print("🎯 Benefits:")
    print("   ✅ Fixes 'moov atom not found' errors")
    print("   ✅ Faster GPU loading (no video decoding)")
    print("   ✅ Better memory utilization")
    print("   ✅ Consistent frame sampling")
    print("   ✅ Enables higher epochs with same time")
    
    print(f"\n📊 Configuration:")
    print(f"   Input: {args.input_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Sequence length: {args.sequence_length}")
    print(f"   Target FPS: {args.target_fps}")
    
    # Create converter
    converter = VideoToImageConverter(args.sequence_length, args.target_fps)
    
    # Convert dataset
    stats = converter.convert_dataset(args.input_dir, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETED")
    print("="*80)
    print(f"📊 Results:")
    print(f"   Total videos processed: {stats['total_videos']}")
    print(f"   Successful conversions: {stats['successful_conversions']}")
    print(f"   Failed conversions: {stats['failed_conversions']}")
    print(f"   Total frames extracted: {stats['total_frames']}")
    print(f"   Success rate: {stats['successful_conversions']/stats['total_videos']*100:.1f}%")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Update training script to use image sequences")
    print(f"   2. Train with higher epochs (20-30) in same time")
    print(f"   3. Better GPU utilization and faster convergence")

if __name__ == "__main__":
    main()
