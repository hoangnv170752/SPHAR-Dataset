#!/usr/bin/env python3
"""
Script to create a human detection dataset from SPHAR videos.
This script will:
1. Extract frames from videos
2. Use YOLOv8 to detect humans in frames
3. Classify frames as 'with_human' or 'without_human'
4. Create YOLO format dataset for human detection training

Author: Generated for human detection training
"""

import os
import cv2
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

class HumanDetectionDatasetCreator:
    def __init__(self, source_dir, output_dir, frame_interval=30, max_videos_per_category=50, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, focus_on_human_behavior=True):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval  # Extract 1 frame every N frames
        self.max_videos_per_category = max_videos_per_category if max_videos_per_category > 0 else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.focus_on_human_behavior = focus_on_human_behavior
        
        # Define categories with high probability of humans (indoor activities)
        self.human_focused_categories = {
            'NTU',  # Always has humans
            'neutral', 'sitting', 'walking', 'running',  # Human activities
            'hitting', 'kicking', 'stealing', 'murdering'  # Human interactions
        }
        
        # Categories that often have no humans or are outdoor/vehicle focused
        self.non_human_categories = {
            'carcrash', 'igniting', 'vandalizing'  # Often no humans visible
        }
        
        # Mixed categories (need detection to separate)
        self.mixed_categories = {
            'falling', 'luggage', 'panicking'
        }
        
        # Load YOLO model for human detection
        self.yolo_model = None
        self._load_yolo_model()
        
        # Statistics tracking
        self.stats = {
            'total_frames_extracted': 0,
            'frames_with_human': 0,
            'frames_without_human': 0,
            'videos_processed': 0,
            'categories': defaultdict(lambda: {'with_human': 0, 'without_human': 0})
        }
        
    def _load_yolo_model(self):
        """Load YOLOv11 model for human detection"""
        try:
            print("Loading YOLOv11 model for human detection...")
            
            # Try to find model in models directory first
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / 'models' / 'yolo11n.pt'
            
            if model_path.exists():
                print(f"Using local model: {model_path}")
                self.yolo_model = YOLO(str(model_path))
            else:
                print("Local model not found, downloading yolo11n.pt...")
                self.yolo_model = YOLO('yolo11n.pt')  # This will download if needed
            
            print("YOLOv11 model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            print("Please ensure you have the model file or internet connection for download")
            raise
            
    def create_output_structure(self):
        """Create the output directory structure for YOLO format"""
        print("Creating output directory structure...")
        
        # YOLO dataset structure
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        # Additional directories for organization
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
    def detect_humans_in_frame(self, frame):
        """
        Detect humans in a frame using YOLOv8
        Returns: (has_human, human_boxes)
        """
        try:
            results = self.yolo_model(frame, verbose=False)
            
            human_boxes = []
            has_human = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Class 0 is 'person' in COCO dataset
                    person_mask = boxes.cls == 0
                    if person_mask.any():
                        has_human = True
                        person_boxes = boxes[person_mask]
                        
                        # Convert to normalized coordinates
                        for box in person_boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            h, w = frame.shape[:2]
                            
                            # Convert to YOLO format (center_x, center_y, width, height)
                            center_x = (x1 + x2) / 2 / w
                            center_y = (y1 + y2) / 2 / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            human_boxes.append([center_x, center_y, width, height])
            
            return has_human, human_boxes
            
        except Exception as e:
            print(f"Error in human detection: {e}")
            return False, []
    
    def extract_frames_from_video(self, video_path, category):
        """Extract frames from video and classify them"""
        print(f"Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust frame interval based on category
        if category == 'NTU':
            # NTU always has humans, extract more frames
            frame_interval = self.frame_interval // 2  # More frequent sampling
        elif category in self.non_human_categories:
            # Background categories - extract more frames for negative samples
            frame_interval = self.frame_interval // 3  # Even more frequent for backgrounds
        elif category in self.human_focused_categories:
            # Human activities - normal sampling
            frame_interval = self.frame_interval
        else:
            # Mixed categories - normal sampling
            frame_interval = self.frame_interval
        
        frames_data = []
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=total_frames//frame_interval, desc=f"Extracting frames from {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    # Detect humans in frame
                    has_human, human_boxes = self.detect_humans_in_frame(frame)
                    
                    # Create frame filename
                    video_stem = video_path.stem
                    frame_filename = f"{category}_{video_stem}_frame_{frame_count:06d}.jpg"
                    
                    # Store frame data
                    frame_data = {
                        'frame': frame.copy(),
                        'filename': frame_filename,
                        'has_human': has_human,
                        'human_boxes': human_boxes,
                        'category': category,
                        'video_name': video_path.name,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0
                    }
                    
                    frames_data.append(frame_data)
                    extracted_count += 1
                    
                    # Update statistics
                    if has_human:
                        self.stats['frames_with_human'] += 1
                        self.stats['categories'][category]['with_human'] += 1
                    else:
                        self.stats['frames_without_human'] += 1
                        self.stats['categories'][category]['without_human'] += 1
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        self.stats['total_frames_extracted'] += extracted_count
        self.stats['videos_processed'] += 1
        
        print(f"Extracted {extracted_count} frames from {video_path.name}")
        return frames_data
    
    def process_all_videos(self, max_videos_per_category=None):
        """Process all videos with focus on human behavior"""
        print("Processing videos with human behavior focus...")
        
        all_frames_data = []
        
        # Prioritize categories based on focus
        if self.focus_on_human_behavior:
            # Process human-focused categories first with higher limits
            priority_categories = list(self.human_focused_categories)
            secondary_categories = list(self.mixed_categories)
            background_categories = list(self.non_human_categories)
        else:
            # Process all categories equally
            priority_categories = []
            secondary_categories = []
            background_categories = []
            for category_dir in self.source_dir.iterdir():
                if category_dir.is_dir():
                    background_categories.append(category_dir.name)
        
        # Define limits per category type
        if self.focus_on_human_behavior:
            # For human behavior focus: NTU unlimited, human activities high, background for negative samples
            ntu_limit = None  # Unlimited for NTU (always has humans)
            priority_limit = max_videos_per_category if max_videos_per_category else 100
            secondary_limit = max_videos_per_category // 2 if max_videos_per_category else 50
            background_limit = max_videos_per_category if max_videos_per_category else 100  # More background for negative samples
        else:
            # Equal processing
            ntu_limit = max_videos_per_category
            priority_limit = max_videos_per_category if max_videos_per_category else 100
            secondary_limit = max_videos_per_category // 2 if max_videos_per_category else 50
            background_limit = max_videos_per_category // 4 if max_videos_per_category else 20
        
        category_processing_plan = [
            (priority_categories, priority_limit, "High Priority (Human Behavior)"),
            (secondary_categories, secondary_limit, "Medium Priority (Mixed)"),
            (background_categories, background_limit, "Background (Non-Human)")
        ]
        
        total_videos = 0
        for categories, limit, desc in category_processing_plan:
            for category_name in categories:
                category_dir = self.source_dir / category_name
                if not category_dir.exists():
                    continue
                    
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                video_files = []
                
                if category_name == 'NTU':
                    # Count videos in NTU subdirectories
                    for action_dir in category_dir.iterdir():
                        if action_dir.is_dir() and action_dir.name.startswith('A'):
                            for ext in video_extensions:
                                video_files.extend(action_dir.glob(ext))
                    # Use special NTU limit
                    actual_limit = min(len(video_files), ntu_limit) if ntu_limit else len(video_files)
                else:
                    # Count videos in regular categories
                    for ext in video_extensions:
                        video_files.extend(category_dir.glob(ext))
                    actual_limit = min(len(video_files), limit) if limit else len(video_files)
                total_videos += actual_limit
                print(f"{desc} - {category_name}: {actual_limit}/{len(video_files)} videos")
        
        print(f"\\nTotal videos to process: {total_videos}")
        
        # Process categories according to plan
        processed_videos = 0
        
        for categories, limit, desc in category_processing_plan:
            if not categories:
                continue
                
            print(f"\\n=== {desc} ===")
            
            for category_name in categories:
                category_dir = self.source_dir / category_name
                if not category_dir.exists():
                    print(f"Skipping {category_name} (directory not found)")
                    continue
                
                # Find video files - handle NTU special structure
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                video_files = []
                
                if category_name == 'NTU':
                    # NTU has subdirectories A001, A002, etc.
                    print(f"  Scanning NTU subdirectories...")
                    for action_dir in category_dir.iterdir():
                        if action_dir.is_dir() and action_dir.name.startswith('A'):
                            for ext in video_extensions:
                                action_videos = list(action_dir.glob(ext))
                                video_files.extend(action_videos)
                                if action_videos:
                                    print(f"    {action_dir.name}: {len(action_videos)} videos")
                else:
                    # Regular categories - videos directly in category folder
                    for ext in video_extensions:
                        video_files.extend(category_dir.glob(ext))
                
                if not video_files:
                    print(f"Skipping {category_name} (no videos)")
                    continue
                
                # Apply limit - special handling for NTU
                if category_name == 'NTU':
                    if ntu_limit and len(video_files) > ntu_limit:
                        video_files = video_files[:ntu_limit]
                    print(f"NTU: Using {len(video_files)} videos (unlimited for human behavior)")
                else:
                    if limit and len(video_files) > limit:
                        video_files = video_files[:limit]
                
                # Show frame extraction strategy
                if category_name == 'NTU':
                    strategy = f"interval={self.frame_interval//2} (high sampling - always has humans)"
                elif category_name in self.non_human_categories:
                    strategy = f"interval={self.frame_interval//3} (very high sampling - background frames)"
                elif category_name in self.human_focused_categories:
                    strategy = f"interval={self.frame_interval} (normal sampling - human activities)"
                else:
                    strategy = f"interval={self.frame_interval} (normal sampling - mixed)"
                
                print(f"Processing {category_name}: {len(video_files)} videos ({strategy})")
                
                # Process videos with progress tracking
                category_frames = []
                with tqdm(video_files, desc=f"{category_name}", unit="video") as pbar:
                    for video_path in pbar:
                        try:
                            frames_data = self.extract_frames_from_video(video_path, category_name)
                            category_frames.extend(frames_data)
                            processed_videos += 1
                            
                            pbar.set_postfix({
                                'frames': len(category_frames),
                                'with_human': sum(1 for f in category_frames if f['has_human']),
                                'progress': f"{processed_videos}/{total_videos}"
                            })
                            
                        except Exception as e:
                            print(f"Error processing {video_path}: {e}")
                            continue
                
                all_frames_data.extend(category_frames)
                
                # Print category summary
                with_human = sum(1 for f in category_frames if f['has_human'])
                without_human = len(category_frames) - with_human
                print(f"  {category_name} summary: {len(category_frames)} frames "
                      f"({with_human} with human, {without_human} without human)")
        
        print(f"\\nCompleted processing {processed_videos} videos")
        print(f"Total frames extracted: {len(all_frames_data)}")
        
        return all_frames_data
    
    def split_frames_data(self, frames_data):
        """Split frames data into train/val/test sets"""
        print("Splitting frames data...")
        
        # Separate frames with and without humans
        frames_with_human = [f for f in frames_data if f['has_human']]
        frames_without_human = [f for f in frames_data if not f['has_human']]
        
        print(f"Frames with human: {len(frames_with_human)}")
        print(f"Frames without human: {len(frames_without_human)}")
        
        # Shuffle both groups
        random.shuffle(frames_with_human)
        random.shuffle(frames_without_human)
        
        # Split each group
        def split_list(data_list, train_r, val_r, test_r):
            n_total = len(data_list)
            n_train = int(n_total * train_r)
            n_val = int(n_total * val_r)
            
            train_data = data_list[:n_train]
            val_data = data_list[n_train:n_train + n_val]
            test_data = data_list[n_train + n_val:]
            
            return train_data, val_data, test_data
        
        # Split frames with humans
        human_train, human_val, human_test = split_list(
            frames_with_human, self.train_ratio, self.val_ratio, self.test_ratio
        )
        
        # Split frames without humans
        no_human_train, no_human_val, no_human_test = split_list(
            frames_without_human, self.train_ratio, self.val_ratio, self.test_ratio
        )
        
        # Combine splits
        splits = {
            'train': human_train + no_human_train,
            'val': human_val + no_human_val,
            'test': human_test + no_human_test
        }
        
        # Shuffle combined splits
        for split_data in splits.values():
            random.shuffle(split_data)
        
        return splits
    
    def save_frames_and_labels(self, splits):
        """Save frames as images and create YOLO format labels"""
        print("Saving frames and creating labels...")
        
        annotations = {}
        
        for split_name, frames_data in splits.items():
            print(f"Processing {split_name} split ({len(frames_data)} frames)...")
            
            annotations[split_name] = []
            
            for frame_data in tqdm(frames_data, desc=f"Saving {split_name} frames"):
                # Save image
                img_path = self.output_dir / 'images' / split_name / frame_data['filename']
                cv2.imwrite(str(img_path), frame_data['frame'])
                
                # Create YOLO label file
                label_filename = frame_data['filename'].replace('.jpg', '.txt')
                label_path = self.output_dir / 'labels' / split_name / label_filename
                
                # Write YOLO format labels (class_id center_x center_y width height)
                with open(label_path, 'w') as f:
                    if frame_data['has_human']:
                        for box in frame_data['human_boxes']:
                            # Class 0 for human
                            f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\\n")
                    # If no humans, create empty label file (required for YOLO)
                
                # Add to annotations
                annotations[split_name].append({
                    'filename': frame_data['filename'],
                    'has_human': frame_data['has_human'],
                    'num_humans': len(frame_data['human_boxes']),
                    'category': frame_data['category'],
                    'video_name': frame_data['video_name'],
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'split': split_name
                })
        
        return annotations
    
    def create_yolo_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['person']  # Class names
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"YOLO config saved to: {config_path}")
        return config_path
    
    def save_annotations_and_metadata(self, annotations):
        """Save annotations and metadata"""
        print("Saving annotations and metadata...")
        
        # Save annotations as JSON
        annotations_path = self.output_dir / 'annotations' / 'dataset_annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save statistics
        stats_path = self.output_dir / 'metadata' / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create CSV files for each split
        for split_name, frames_data in annotations.items():
            csv_path = self.output_dir / 'annotations' / f'{split_name}_annotations.csv'
            with open(csv_path, 'w') as f:
                f.write('filename,has_human,num_humans,category,video_name,frame_number,timestamp,split\\n')
                for frame in frames_data:
                    f.write(f"{frame['filename']},{frame['has_human']},{frame['num_humans']},"
                           f"{frame['category']},{frame['video_name']},{frame['frame_number']},"
                           f"{frame['timestamp']:.2f},{frame['split']}\\n")
    
    def create_dataset_info(self, annotations):
        """Create dataset information file"""
        print("Creating dataset information...")
        
        # Calculate split statistics
        split_stats = {}
        for split_name, frames_data in annotations.items():
            with_human = sum(1 for f in frames_data if f['has_human'])
            without_human = len(frames_data) - with_human
            
            split_stats[split_name] = {
                'total_frames': len(frames_data),
                'frames_with_human': with_human,
                'frames_without_human': without_human,
                'human_ratio': with_human / len(frames_data) if frames_data else 0
            }
        
        dataset_info = {
            'dataset_name': 'SPHAR Human Detection Dataset',
            'description': 'Dataset for human detection in surveillance videos',
            'created_from': 'SPHAR Dataset',
            'task': 'Human Detection (YOLO format)',
            'classes': {
                'person': {
                    'id': 0,
                    'description': 'Human/Person detection'
                }
            },
            'splits': split_stats,
            'total_frames': sum(len(frames) for frames in annotations.values()),
            'frame_extraction': {
                'interval': self.frame_interval,
                'method': 'uniform_sampling'
            },
            'statistics': self.stats
        }
        
        # Save dataset info
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create README
        self._create_readme(dataset_info)
        
        return dataset_info
    
    def _create_readme(self, dataset_info):
        """Create README file for the dataset"""
        readme_content = f"""# SPHAR Human Detection Dataset

## Overview
This dataset is created for training YOLO models to detect humans in surveillance videos.
It contains frames extracted from the SPHAR dataset, labeled for human detection.

## Dataset Statistics
- **Total Frames**: {dataset_info['total_frames']}
- **Frame Extraction Interval**: Every {self.frame_interval} frames
- **Task**: Human Detection (Object Detection)

### Split Distribution
"""
        
        for split_name, stats in dataset_info['splits'].items():
            readme_content += f"""
#### {split_name.title()} Split
- Total Frames: {stats['total_frames']}
- Frames with Human: {stats['frames_with_human']}
- Frames without Human: {stats['frames_without_human']}
- Human Ratio: {stats['human_ratio']:.2f}
"""
        
        readme_content += f"""
## Directory Structure
```
human_detection_dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   ├── val/            # Validation labels
│   └── test/           # Test labels
├── annotations/
│   ├── dataset_annotations.json
│   ├── train_annotations.csv
│   ├── val_annotations.csv
│   └── test_annotations.csv
├── metadata/
│   └── dataset_stats.json
├── dataset.yaml        # YOLO configuration
├── dataset_info.json
└── README.md
```

## YOLO Format
- **Images**: JPG format in images/ subdirectories
- **Labels**: TXT format in labels/ subdirectories
- **Label Format**: `class_id center_x center_y width height` (normalized coordinates)
- **Classes**: 0 = person

## Usage
This dataset is ready for training YOLO models for human detection:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11s.pt')

# Train model
model.train(data='dataset.yaml', epochs=100, imgsz=640)
```

## Citation
Please cite the original SPHAR dataset when using this derived dataset.
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def create_dataset(self):
        """Main method to create the human detection dataset"""
        print("Starting human detection dataset creation...")
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Frame extraction interval: {self.frame_interval}")
        
        # Create output structure
        self.create_output_structure()
        
        # Process all videos and extract frames
        all_frames_data = self.process_all_videos(max_videos_per_category=self.max_videos_per_category)
        
        if not all_frames_data:
            print("No frames extracted! Please check the source directory and video files.")
            return
        
        print(f"Total frames extracted: {len(all_frames_data)}")
        
        # Split data
        splits = self.split_frames_data(all_frames_data)
        
        # Save frames and create labels
        annotations = self.save_frames_and_labels(splits)
        
        # Create YOLO config
        self.create_yolo_config()
        
        # Save annotations and metadata
        self.save_annotations_and_metadata(annotations)
        
        # Create dataset info
        dataset_info = self.create_dataset_info(annotations)
        
        print("\\n" + "="*60)
        print("Human Detection Dataset Creation Completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total frames processed: {dataset_info['total_frames']}")
        print(f"Frames with humans: {self.stats['frames_with_human']}")
        print(f"Frames without humans: {self.stats['frames_without_human']}")
        print(f"Videos processed: {self.stats['videos_processed']}")
        print("="*60)
        
        return dataset_info

def main():
    parser = argparse.ArgumentParser(description='Create human detection dataset from SPHAR videos')
    parser.add_argument('--source', '-s', 
                       default=r'D:\\SPHAR-Dataset\\videos',
                       help='Source directory containing SPHAR videos')
    parser.add_argument('--output', '-o',
                       default=r'D:\\SPHAR-Dataset\\train\\human_detection_dataset',
                       help='Output directory for the human detection dataset')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Extract 1 frame every N frames (default: 30)')
    parser.add_argument('--max-videos-per-category', type=int, default=50,
                       help='Maximum videos to process per category (default: 50, use 0 for unlimited)')
    parser.add_argument('--focus-human-behavior', action='store_true', default=True,
                       help='Focus on human behavior categories (default: True)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of frames for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Ratio of frames for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Ratio of frames for testing (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("Error: Train, validation, and test ratios must sum to 1.0")
        return
    
    # Create dataset
    creator = HumanDetectionDatasetCreator(
        source_dir=args.source,
        output_dir=args.output,
        frame_interval=args.frame_interval,
        max_videos_per_category=args.max_videos_per_category,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        focus_on_human_behavior=args.focus_human_behavior
    )
    
    try:
        dataset_info = creator.create_dataset()
        
        if dataset_info:
            print(f"\\nDataset successfully created at: {args.output}")
            print("You can now use this dataset for training human detection models!")
            print("\\nNext steps:")
            print("1. Review the dataset statistics in dataset_info.json")
            print("2. Train YOLO model using: python -m ultralytics.yolo train data=dataset.yaml")
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
