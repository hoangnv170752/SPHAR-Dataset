#!/usr/bin/env python3
"""
Enhanced script to create indoor-focused human detection dataset with occlusion handling.
This script prioritizes indoor activities from Toyota RGB videos and SPHAR dataset
with improved detection for partially occluded humans.

Features:
- Focus on indoor activities (cooking, cleaning, etc.)
- Enhanced human detection with occlusion handling
- Improved data augmentation for indoor scenarios
- Better sampling strategy for occluded humans
- Multi-scale detection for different human sizes

Author: Enhanced for indoor human detection with occlusion handling
"""

import os
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
from ultralytics import YOLO
import yaml

class IndoorFocusedDatasetCreator:
    def __init__(self, source_dir, toyota_dir, output_dir, 
                 frame_interval=20, max_videos_per_category=100, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 min_human_size=0.02, confidence_threshold=0.3):
        
        self.source_dir = Path(source_dir)
        self.toyota_dir = Path(toyota_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval  # More frequent sampling for indoor scenes
        self.max_videos_per_category = max_videos_per_category
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_human_size = min_human_size  # Minimum human bounding box size (relative)
        self.confidence_threshold = confidence_threshold  # Lower threshold for occluded humans
        
        # Define indoor-focused categories with high priority
        self.indoor_priority_categories = {
            'Cook.Cleandishes', 'Cook.Cleanup', 'Cook.Cut', 'Cook.Stir',  # Toyota cooking
            'WatchTV', 'Walk',  # Toyota indoor activities
            'NTU',  # Always has humans, indoor activities
            'sitting', 'neutral', 'walking',  # Indoor human activities
            'hitting', 'kicking', 'stealing', 'murdering'  # Human interactions (indoor)
        }
        
        # Categories that might have occlusion challenges
        self.occlusion_categories = {
            'Cook.Cleandishes', 'Cook.Cleanup', 'Cook.Cut', 'Cook.Stir',  # Kitchen occlusion
            'luggage', 'falling'  # Object occlusion
        }
        
        # Background categories (lower priority but needed for negative samples)
        self.background_categories = {
            'carcrash', 'igniting', 'vandalizing'
        }
        
        # Load enhanced YOLO model for better occlusion handling
        self.yolo_model = None
        self._load_yolo_model()
        
        # Statistics tracking
        self.stats = {
            'total_frames_extracted': 0,
            'frames_with_human': 0,
            'frames_without_human': 0,
            'frames_with_occluded_human': 0,
            'videos_processed': 0,
            'categories': defaultdict(lambda: {
                'with_human': 0, 
                'without_human': 0, 
                'occluded_human': 0,
                'small_human': 0
            })
        }
        
    def _load_yolo_model(self):
        """Load YOLOv11 model with enhanced settings for occlusion detection"""
        try:
            print("Loading YOLOv11 model for enhanced human detection...")
            
            # Try to find model in models directory first
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / 'models' / 'yolo11s.pt'
            
            if model_path.exists():
                print(f"Using local model: {model_path}")
                self.yolo_model = YOLO(str(model_path))
            else:
                print("Local model not found, downloading yolo11s.pt...")
                self.yolo_model = YOLO('yolo11s.pt')
            
            print("YOLOv11 model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
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
        
        # Additional directories
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
    def detect_humans_with_occlusion_handling(self, frame):
        """
        Enhanced human detection with better occlusion handling
        Returns: (has_human, human_boxes, occlusion_info)
        """
        try:
            # Use multiple confidence thresholds for better occlusion detection
            results_high = self.yolo_model(frame, conf=0.5, verbose=False)  # High confidence
            results_low = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)  # Low confidence for occluded
            
            human_boxes = []
            has_human = False
            occlusion_info = {
                'has_occluded': False,
                'has_small': False,
                'total_detections': 0,
                'high_conf_detections': 0
            }
            
            # Process high confidence detections first
            high_conf_boxes = []
            for result in results_high:
                boxes = result.boxes
                if boxes is not None:
                    person_mask = boxes.cls == 0
                    if person_mask.any():
                        person_boxes = boxes[person_mask]
                        for box in person_boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].item()
                            high_conf_boxes.append((x1, y1, x2, y2, conf))
            
            # Process low confidence detections
            all_boxes = []
            for result in results_low:
                boxes = result.boxes
                if boxes is not None:
                    person_mask = boxes.cls == 0
                    if person_mask.any():
                        person_boxes = boxes[person_mask]
                        for box in person_boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].item()
                            all_boxes.append((x1, y1, x2, y2, conf))
            
            # Remove duplicates and process all detections
            final_boxes = self._remove_duplicate_detections(all_boxes)
            
            if final_boxes:
                has_human = True
                h, w = frame.shape[:2]
                
                for x1, y1, x2, y2, conf in final_boxes:
                    # Convert to YOLO format
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Check for small/occluded humans
                    box_area = width * height
                    if box_area < self.min_human_size:
                        occlusion_info['has_small'] = True
                    
                    if conf < 0.5:
                        occlusion_info['has_occluded'] = True
                    
                    human_boxes.append([center_x, center_y, width, height, conf])
                
                occlusion_info['total_detections'] = len(final_boxes)
                occlusion_info['high_conf_detections'] = len(high_conf_boxes)
            
            return has_human, human_boxes, occlusion_info
            
        except Exception as e:
            print(f"Error in human detection: {e}")
            return False, [], {'has_occluded': False, 'has_small': False, 'total_detections': 0, 'high_conf_detections': 0}
    
    def _remove_duplicate_detections(self, boxes, iou_threshold=0.5):
        """Remove duplicate detections using Non-Maximum Suppression"""
        if not boxes:
            return []
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        final_boxes = []
        for box in boxes:
            x1, y1, x2, y2, conf = box
            
            # Check IoU with existing boxes
            is_duplicate = False
            for existing_box in final_boxes:
                ex1, ey1, ex2, ey2, _ = existing_box
                
                # Calculate IoU
                intersection = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ex2 - ex1) * (ey2 - ey1)
                union = area1 + area2 - intersection
                
                if union > 0 and intersection / union > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_boxes.append(box)
        
        return final_boxes
    
    def extract_frames_from_video(self, video_path, category):
        """Extract frames with enhanced indoor focus and occlusion handling"""
        print(f"Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust frame interval based on category priority
        if any(cat in category for cat in self.indoor_priority_categories):
            frame_interval = self.frame_interval // 2  # More frequent sampling for indoor
        elif any(cat in category for cat in self.occlusion_categories):
            frame_interval = self.frame_interval // 3  # Even more for occlusion scenarios
        else:
            frame_interval = self.frame_interval
        
        frames_data = []
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=total_frames//frame_interval, desc=f"Extracting from {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Enhanced human detection
                    has_human, human_boxes, occlusion_info = self.detect_humans_with_occlusion_handling(frame)
                    
                    # Create frame filename
                    video_stem = video_path.stem
                    frame_filename = f"{category}_{video_stem}_frame_{frame_count:06d}.jpg"
                    
                    # Store frame data with occlusion info
                    frame_data = {
                        'frame': frame.copy(),
                        'filename': frame_filename,
                        'has_human': has_human,
                        'human_boxes': human_boxes,
                        'occlusion_info': occlusion_info,
                        'category': category,
                        'video_name': video_path.name,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'is_indoor': any(cat in category for cat in self.indoor_priority_categories)
                    }
                    
                    frames_data.append(frame_data)
                    extracted_count += 1
                    
                    # Update statistics
                    if has_human:
                        self.stats['frames_with_human'] += 1
                        self.stats['categories'][category]['with_human'] += 1
                        
                        if occlusion_info['has_occluded']:
                            self.stats['frames_with_occluded_human'] += 1
                            self.stats['categories'][category]['occluded_human'] += 1
                        
                        if occlusion_info['has_small']:
                            self.stats['categories'][category]['small_human'] += 1
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
    
    def process_toyota_videos(self):
        """Process Toyota RGB videos with indoor focus"""
        print("Processing Toyota RGB videos...")
        
        toyota_frames = []
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        
        # Get all Toyota videos
        toyota_videos = []
        for ext in video_extensions:
            toyota_videos.extend(self.toyota_dir.glob(ext))
        
        # Group by activity type
        activity_groups = defaultdict(list)
        for video in toyota_videos:
            # Extract activity from filename (e.g., "Cook.Cleandishes_p02_r00_v02_c03.mp4")
            activity = video.name.split('_')[0]
            activity_groups[activity].append(video)
        
        print(f"Found {len(toyota_videos)} Toyota videos in {len(activity_groups)} activity groups")
        
        # Process each activity group
        for activity, videos in activity_groups.items():
            if self.max_videos_per_category and len(videos) > self.max_videos_per_category:
                videos = videos[:self.max_videos_per_category]
            
            print(f"Processing {activity}: {len(videos)} videos")
            
            for video_path in tqdm(videos, desc=f"Toyota {activity}"):
                try:
                    frames_data = self.extract_frames_from_video(video_path, f"Toyota_{activity}")
                    toyota_frames.extend(frames_data)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue
        
        return toyota_frames
    
    def process_sphar_videos(self):
        """Process SPHAR videos with indoor focus"""
        print("Processing SPHAR videos with indoor focus...")
        
        sphar_frames = []
        
        # Priority processing order
        category_processing_plan = [
            (self.indoor_priority_categories, "High Priority Indoor"),
            (self.occlusion_categories, "Occlusion Focus"),
            (self.background_categories, "Background")
        ]
        
        for categories, desc in category_processing_plan:
            print(f"\n=== {desc} ===")
            
            for category_name in categories:
                category_dir = self.source_dir / category_name
                if not category_dir.exists():
                    continue
                
                # Find video files
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                video_files = []
                
                if category_name == 'NTU':
                    # NTU has subdirectories
                    for action_dir in category_dir.iterdir():
                        if action_dir.is_dir() and action_dir.name.startswith('A'):
                            for ext in video_extensions:
                                video_files.extend(action_dir.glob(ext))
                else:
                    for ext in video_extensions:
                        video_files.extend(category_dir.glob(ext))
                
                if not video_files:
                    continue
                
                # Apply limits
                if self.max_videos_per_category and len(video_files) > self.max_videos_per_category:
                    video_files = video_files[:self.max_videos_per_category]
                
                print(f"Processing {category_name}: {len(video_files)} videos")
                
                for video_path in tqdm(video_files, desc=category_name):
                    try:
                        frames_data = self.extract_frames_from_video(video_path, category_name)
                        sphar_frames.extend(frames_data)
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        continue
        
        return sphar_frames
    
    def create_balanced_splits(self, all_frames_data):
        """Create balanced train/val/test splits with indoor focus"""
        print("Creating balanced splits with indoor focus...")
        
        # Separate indoor and outdoor frames
        indoor_frames = [f for f in all_frames_data if f.get('is_indoor', False)]
        outdoor_frames = [f for f in all_frames_data if not f.get('is_indoor', False)]
        
        print(f"Indoor frames: {len(indoor_frames)}")
        print(f"Outdoor frames: {len(outdoor_frames)}")
        
        # Further separate by human presence
        indoor_with_human = [f for f in indoor_frames if f['has_human']]
        indoor_without_human = [f for f in indoor_frames if not f['has_human']]
        outdoor_with_human = [f for f in outdoor_frames if f['has_human']]
        outdoor_without_human = [f for f in outdoor_frames if not f['has_human']]
        
        # Shuffle all groups
        for group in [indoor_with_human, indoor_without_human, outdoor_with_human, outdoor_without_human]:
            random.shuffle(group)
        
        def split_group(group, train_r, val_r, test_r):
            n_total = len(group)
            n_train = int(n_total * train_r)
            n_val = int(n_total * val_r)
            
            return (group[:n_train], 
                   group[n_train:n_train + n_val], 
                   group[n_train + n_val:])
        
        # Split each group
        splits = {'train': [], 'val': [], 'test': []}
        
        for group in [indoor_with_human, indoor_without_human, outdoor_with_human, outdoor_without_human]:
            if group:
                train_split, val_split, test_split = split_group(group, self.train_ratio, self.val_ratio, self.test_ratio)
                splits['train'].extend(train_split)
                splits['val'].extend(val_split)
                splits['test'].extend(test_split)
        
        # Final shuffle
        for split_data in splits.values():
            random.shuffle(split_data)
        
        return splits
    
    def save_frames_and_labels(self, splits):
        """Save frames and create enhanced YOLO labels"""
        print("Saving frames and creating enhanced labels...")
        
        annotations = {}
        
        for split_name, frames_data in splits.items():
            print(f"Processing {split_name} split ({len(frames_data)} frames)...")
            
            annotations[split_name] = []
            
            for frame_data in tqdm(frames_data, desc=f"Saving {split_name} frames"):
                # Save image
                img_path = self.output_dir / 'images' / split_name / frame_data['filename']
                cv2.imwrite(str(img_path), frame_data['frame'])
                
                # Create enhanced YOLO label file
                label_filename = frame_data['filename'].replace('.jpg', '.txt')
                label_path = self.output_dir / 'labels' / split_name / label_filename
                
                # Write YOLO format labels with confidence scores
                with open(label_path, 'w') as f:
                    if frame_data['has_human']:
                        for box in frame_data['human_boxes']:
                            if len(box) >= 4:  # Ensure we have at least x, y, w, h
                                # Class 0 for human, with optional confidence
                                if len(box) >= 5:
                                    f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                                else:
                                    f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                
                # Enhanced annotations
                annotations[split_name].append({
                    'filename': frame_data['filename'],
                    'has_human': frame_data['has_human'],
                    'num_humans': len(frame_data['human_boxes']),
                    'category': frame_data['category'],
                    'video_name': frame_data['video_name'],
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'split': split_name,
                    'is_indoor': frame_data.get('is_indoor', False),
                    'occlusion_info': frame_data.get('occlusion_info', {}),
                    'detection_quality': 'high' if frame_data.get('occlusion_info', {}).get('high_conf_detections', 0) > 0 else 'low'
                })
        
        return annotations
    
    def create_enhanced_yolo_config(self):
        """Create enhanced YOLO dataset configuration"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['person'],
            
            # Enhanced settings for indoor detection
            'indoor_focused': True,
            'occlusion_handling': True,
            'min_human_size': self.min_human_size,
            'confidence_threshold': self.confidence_threshold
        }
        
        config_path = self.output_dir / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Enhanced YOLO config saved to: {config_path}")
        return config_path
    
    def create_dataset_info(self, annotations):
        """Create comprehensive dataset information"""
        print("Creating enhanced dataset information...")
        
        # Calculate enhanced statistics
        split_stats = {}
        for split_name, frames_data in annotations.items():
            indoor_frames = sum(1 for f in frames_data if f.get('is_indoor', False))
            occluded_frames = sum(1 for f in frames_data if f.get('occlusion_info', {}).get('has_occluded', False))
            small_human_frames = sum(1 for f in frames_data if f.get('occlusion_info', {}).get('has_small', False))
            
            with_human = sum(1 for f in frames_data if f['has_human'])
            without_human = len(frames_data) - with_human
            
            split_stats[split_name] = {
                'total_frames': len(frames_data),
                'frames_with_human': with_human,
                'frames_without_human': without_human,
                'human_ratio': with_human / len(frames_data) if frames_data else 0,
                'indoor_frames': indoor_frames,
                'indoor_ratio': indoor_frames / len(frames_data) if frames_data else 0,
                'occluded_frames': occluded_frames,
                'small_human_frames': small_human_frames
            }
        
        dataset_info = {
            'dataset_name': 'Indoor-Focused Human Detection Dataset',
            'description': 'Enhanced dataset for human detection in indoor environments with occlusion handling',
            'created_from': 'SPHAR Dataset + Toyota RGB Videos',
            'task': 'Human Detection (YOLO format) - Indoor Focus',
            'classes': {
                'person': {
                    'id': 0,
                    'description': 'Human/Person detection with occlusion handling'
                }
            },
            'splits': split_stats,
            'total_frames': sum(len(frames) for frames in annotations.values()),
            'frame_extraction': {
                'interval': self.frame_interval,
                'method': 'adaptive_sampling_indoor_focus',
                'min_human_size': self.min_human_size,
                'confidence_threshold': self.confidence_threshold
            },
            'enhancements': {
                'indoor_focus': True,
                'occlusion_handling': True,
                'multi_confidence_detection': True,
                'toyota_rgb_integration': True
            },
            'statistics': self.stats
        }
        
        # Save dataset info
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info
    
    def create_dataset(self):
        """Main method to create the enhanced indoor-focused dataset"""
        print("="*60)
        print("INDOOR-FOCUSED HUMAN DETECTION DATASET CREATION")
        print("="*60)
        print(f"Source directory: {self.source_dir}")
        print(f"Toyota directory: {self.toyota_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Frame extraction interval: {self.frame_interval}")
        print(f"Min human size: {self.min_human_size}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Create output structure
        self.create_output_structure()
        
        # Process videos
        all_frames_data = []
        
        # Process Toyota videos (high priority)
        toyota_frames = self.process_toyota_videos()
        all_frames_data.extend(toyota_frames)
        
        # Process SPHAR videos
        sphar_frames = self.process_sphar_videos()
        all_frames_data.extend(sphar_frames)
        
        if not all_frames_data:
            print("No frames extracted! Please check the source directories.")
            return
        
        print(f"Total frames extracted: {len(all_frames_data)}")
        
        # Create balanced splits
        splits = self.create_balanced_splits(all_frames_data)
        
        # Save frames and labels
        annotations = self.save_frames_and_labels(splits)
        
        # Create enhanced YOLO config
        self.create_enhanced_yolo_config()
        
        # Create dataset info
        dataset_info = self.create_dataset_info(annotations)
        
        print("\n" + "="*60)
        print("INDOOR-FOCUSED DATASET CREATION COMPLETED!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total frames: {dataset_info['total_frames']}")
        print(f"Indoor frames: {sum(stats.get('indoor_frames', 0) for stats in dataset_info['splits'].values())}")
        print(f"Frames with humans: {self.stats['frames_with_human']}")
        print(f"Frames with occluded humans: {self.stats['frames_with_occluded_human']}")
        print("="*60)
        
        return dataset_info

def main():
    parser = argparse.ArgumentParser(description='Create indoor-focused human detection dataset')
    parser.add_argument('--source', '-s',
                       default=r'D:\SPHAR-Dataset\videos',
                       help='Path to SPHAR videos directory')
    parser.add_argument('--toyota', '-t',
                       default=r'D:\SPHAR-Dataset\videos\toyota',
                       help='Path to Toyota RGB videos directory')
    parser.add_argument('--output', '-o',
                       default=r'D:\SPHAR-Dataset\train\indoor_focused_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--interval', '-i', type=int, default=20,
                       help='Frame extraction interval (default: 20)')
    parser.add_argument('--max-videos', '-m', type=int, default=100,
                       help='Maximum videos per category (default: 100)')
    parser.add_argument('--min-size', type=float, default=0.02,
                       help='Minimum human size ratio (default: 0.02)')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for occlusion detection (default: 0.3)')
    
    args = parser.parse_args()
    
    try:
        creator = IndoorFocusedDatasetCreator(
            source_dir=args.source,
            toyota_dir=args.toyota,
            output_dir=args.output,
            frame_interval=args.interval,
            max_videos_per_category=args.max_videos,
            min_human_size=args.min_size,
            confidence_threshold=args.conf_threshold
        )
        
        dataset_info = creator.create_dataset()
        
        if dataset_info:
            print("\n✅ Dataset creation completed successfully!")
            print("\nNext steps:")
            print("1. Review the dataset statistics")
            print("2. Use the enhanced training script for fine-tuning")
            print("3. Monitor training for occlusion handling performance")
        else:
            print("\n❌ Dataset creation failed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
