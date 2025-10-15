#!/usr/bin/env python3
"""
Test URFD dataset processing for fall detection
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_urfd_dataset():
    """Analyze URFD dataset structure"""
    urfd_dir = Path(r'D:\SPHAR-Dataset\videos\URFD')
    
    if not urfd_dir.exists():
        print("❌ URFD directory not found")
        return
    
    print("="*80)
    print("📊 URFD DATASET ANALYSIS")
    print("="*80)
    
    # Check splits
    for split in ['train', 'valid', 'test']:
        images_dir = urfd_dir / split / 'images'
        labels_dir = urfd_dir / split / 'labels'
        
        if images_dir.exists():
            images = list(images_dir.glob('*.jpg'))
            print(f"\n📁 {split.upper()} split:")
            print(f"   Images: {len(images)}")
            
            if labels_dir.exists():
                labels = list(labels_dir.glob('*.txt'))
                print(f"   Labels: {len(labels)}")
            
            # Analyze image types
            fall_images = [img for img in images if 'fall-' in img.name]
            adl_images = [img for img in images if 'adl-' in img.name]
            
            print(f"   Fall images: {len(fall_images)}")
            print(f"   ADL images: {len(adl_images)}")
            
            # Analyze fall sequences
            if fall_images:
                fall_sequences = defaultdict(list)
                for img in fall_images:
                    # Extract sequence ID (e.g., fall-01-cam0-rgb-012)
                    parts = img.stem.split('-')
                    if len(parts) >= 2:
                        seq_id = parts[1]  # Get '01', '02', etc.
                        fall_sequences[seq_id].append(img)
                
                print(f"   Fall sequences: {len(fall_sequences)}")
                
                # Show sequence details
                for seq_id, seq_images in sorted(fall_sequences.items()):
                    print(f"     Sequence {seq_id}: {len(seq_images)} frames")

def test_urfd_sequence_loading():
    """Test loading URFD image sequences"""
    urfd_dir = Path(r'D:\SPHAR-Dataset\videos\URFD')
    
    print("\n" + "="*80)
    print("🧪 TESTING URFD SEQUENCE LOADING")
    print("="*80)
    
    # Find fall sequences
    fall_sequences = defaultdict(list)
    
    for split in ['train', 'valid', 'test']:
        images_dir = urfd_dir / split / 'images'
        if images_dir.exists():
            fall_images = list(images_dir.glob('fall-*.jpg'))
            
            for img in fall_images:
                parts = img.stem.split('-')
                if len(parts) >= 2:
                    seq_id = parts[1]
                    fall_sequences[seq_id].append(img)
    
    # Test loading first sequence
    if fall_sequences:
        first_seq_id = sorted(fall_sequences.keys())[0]
        first_seq_images = fall_sequences[first_seq_id]
        
        print(f"\n📹 Testing sequence: {first_seq_id}")
        print(f"📊 Images in sequence: {len(first_seq_images)}")
        
        # Sort by frame number
        first_seq_images.sort(key=lambda x: int(x.stem.split('-')[-1].split('_')[0]))
        
        print("\n🖼️ Image sequence:")
        for i, img_path in enumerate(first_seq_images[:10]):  # Show first 10
            frame_num = img_path.stem.split('-')[-1].split('_')[0]
            print(f"   {i+1:2d}. Frame {frame_num}: {img_path.name}")
        
        if len(first_seq_images) > 10:
            print(f"   ... and {len(first_seq_images) - 10} more frames")
        
        # Test loading images
        print(f"\n🔄 Loading images...")
        loaded_frames = []
        
        for img_path in first_seq_images[:16]:  # Load first 16 frames
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.resize(frame, (224, 224))
                loaded_frames.append(frame)
        
        print(f"✅ Successfully loaded {len(loaded_frames)} frames")
        print(f"📐 Frame shape: {loaded_frames[0].shape if loaded_frames else 'None'}")
        
        # Create a sample video clip
        if loaded_frames:
            print(f"\n🎬 Creating sample video clip...")
            
            # Create output directory
            output_dir = Path(r'D:\SPHAR-Dataset\output')
            output_dir.mkdir(exist_ok=True)
            
            # Save as video
            output_path = output_dir / f'urfd_fall_sequence_{first_seq_id}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, 5.0, (224, 224))
            
            for frame in loaded_frames:
                writer.write(frame)
            
            writer.release()
            
            print(f"💾 Sample video saved: {output_path}")
    
    else:
        print("❌ No fall sequences found")

def create_urfd_pseudo_videos():
    """Create pseudo video files for URFD sequences"""
    urfd_dir = Path(r'D:\SPHAR-Dataset\videos\URFD')
    
    print("\n" + "="*80)
    print("🎯 CREATING URFD PSEUDO VIDEOS")
    print("="*80)
    
    # Find all fall sequences
    fall_sequences = defaultdict(list)
    
    for split in ['train', 'valid', 'test']:
        images_dir = urfd_dir / split / 'images'
        if images_dir.exists():
            fall_images = list(images_dir.glob('fall-*.jpg'))
            
            for img in fall_images:
                parts = img.stem.split('-')
                if len(parts) >= 2:
                    seq_id = parts[1]
                    fall_sequences[seq_id].append(img)
    
    print(f"📊 Found {len(fall_sequences)} fall sequences")
    
    # Create pseudo video files
    pseudo_videos = []
    for seq_id, images in fall_sequences.items():
        if len(images) >= 8:  # Need at least 8 images
            pseudo_video = urfd_dir / f"fall_sequence_{seq_id}.urfd"
            pseudo_videos.append(pseudo_video)
            
            # Create the pseudo file (just a marker)
            with open(pseudo_video, 'w') as f:
                f.write(f"URFD fall sequence {seq_id}\n")
                f.write(f"Images: {len(images)}\n")
                for img in sorted(images, key=lambda x: int(x.stem.split('-')[-1].split('_')[0])):
                    f.write(f"{img}\n")
    
    print(f"✅ Created {len(pseudo_videos)} pseudo video files")
    
    for pv in pseudo_videos[:5]:  # Show first 5
        print(f"   📄 {pv.name}")
    
    if len(pseudo_videos) > 5:
        print(f"   ... and {len(pseudo_videos) - 5} more")
    
    return pseudo_videos

def main():
    print("🎬 URFD Dataset Processing Test")
    
    # Step 1: Analyze dataset
    analyze_urfd_dataset()
    
    # Step 2: Test sequence loading
    test_urfd_sequence_loading()
    
    # Step 3: Create pseudo videos
    pseudo_videos = create_urfd_pseudo_videos()
    
    print("\n" + "="*80)
    print("✅ URFD PROCESSING TEST COMPLETED")
    print("="*80)
    print(f"📊 Summary:")
    print(f"   - URFD contains image sequences for fall detection")
    print(f"   - Created {len(pseudo_videos)} pseudo video files")
    print(f"   - Ready for action recognition training")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Run dataset organization:")
    print(f"      python run_action_training.py --organize-only")
    print(f"   ")
    print(f"   2. Check organized dataset:")
    print(f"      Check D:\\SPHAR-Dataset\\action_recognition\\dataset_info.json")

if __name__ == "__main__":
    main()
