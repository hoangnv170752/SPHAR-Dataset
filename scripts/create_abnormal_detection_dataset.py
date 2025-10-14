#!/usr/bin/env python3
"""
Script to create a new dataset for abnormal activity detection training
by combining NTU RGB+ videos with other videos from the SPHAR dataset.

This script will:
1. Categorize videos into 'normal' and 'abnormal' classes
2. Create a balanced dataset for training
3. Generate train/validation/test splits
4. Create annotation files and dataset structure

Author: Generated for abnormal activity detection training
"""

import os
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

class AbnormalDetectionDatasetCreator:
    def __init__(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Define which categories are considered normal
        self.normal_categories = {
            'neutral', 'sitting', 'walking', 'running', 'luggage'
        }
        
        # Define which categories are abnormal based on physical actions (violence, accidents)
        self.abnormal_physical_categories = {
            'hitting', 'kicking', 'murdering', 'stealing', 'vandalizing', 'carcrash'
        }
        
        # Define which categories are abnormal based on biological conditions (medical, falling)
        self.abnormal_biological_categories = {
            'falling', 'igniting', 'panicking'
        }
        
        # Define NTU actions for physical abnormal activities (aggressive/violent actions)
        self.ntu_abnormal_physical_actions = {
            # Aggressive Mutual Actions
            'A050',  # punch/slap
            'A051',  # kicking
            'A052',  # pushing
            'A106',  # hit with object
            'A110',  # shoot with gun
            # Potentially aggressive daily actions
            'A024',  # kicking something
        }
        
        # Define NTU actions for biological abnormal activities (medical conditions)
        self.ntu_abnormal_biological_actions = {
            # Medical Conditions
            'A041',  # sneeze/cough
            'A042',  # staggering
            'A043',  # falling down
            'A044',  # headache
            'A045',  # chest pain
            'A046',  # back pain
            'A047',  # neck pain
            'A048',  # nausea/vomiting
            'A049',  # fan self
        }
        
        # Define NTU actions that should be considered normal (daily activities, peaceful interactions)
        self.ntu_normal_actions = {
            # Daily Actions (A001-A040) - excluding aggressive and medical ones
            'A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010',
            'A011', 'A012', 'A013', 'A014', 'A015', 'A016', 'A017', 'A018', 'A019', 'A020',
            'A021', 'A022', 'A023', 'A025', 'A026', 'A027', 'A028', 'A029', 'A030',
            'A031', 'A032', 'A033', 'A034', 'A035', 'A036', 'A037', 'A038', 'A039', 'A040',
            # Normal biological functions
            'A103', 'A104', 'A105',  # yawn, stretch oneself, blow nose
            # Normal Mutual Actions (peaceful interactions)
            'A053', 'A054', 'A055', 'A056', 'A057', 'A058', 'A059', 'A060',  # pat on back, point finger, etc.
            'A107', 'A108', 'A109', 'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119', 'A120'
        }
        
        # Statistics tracking
        self.stats = {
            'normal': defaultdict(int),
            'abnormal_physical': defaultdict(int),
            'abnormal_biological': defaultdict(int),
            'total_normal': 0,
            'total_abnormal_physical': 0,
            'total_abnormal_biological': 0
        }
        
    def create_output_structure(self):
        """Create the output directory structure"""
        print("Creating output directory structure...")
        
        # Main directories for 3 classes
        (self.output_dir / 'videos' / 'train' / 'normal').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'train' / 'abnormal_physical').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'train' / 'abnormal_biological').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'val' / 'normal').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'val' / 'abnormal_physical').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'val' / 'abnormal_biological').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'test' / 'normal').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'test' / 'abnormal_physical').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'test' / 'abnormal_biological').mkdir(parents=True, exist_ok=True)
        
        # Annotations directory
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
        
    def collect_video_files(self):
        """Collect all video files from source directories"""
        print("Collecting video files...")
        
        normal_videos = []
        abnormal_physical_videos = []
        abnormal_biological_videos = []
        
        # Process each category
        for category_dir in self.source_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            
            if category_name == 'NTU':
                # Handle NTU RGB+ videos with action-based classification
                ntu_normal, ntu_physical, ntu_biological = self._process_ntu_videos(category_dir)
                normal_videos.extend(ntu_normal)
                abnormal_physical_videos.extend(ntu_physical)
                abnormal_biological_videos.extend(ntu_biological)
                self.stats['normal']['NTU_normal'] = len(ntu_normal)
                self.stats['abnormal_physical']['NTU_physical'] = len(ntu_physical)
                self.stats['abnormal_biological']['NTU_biological'] = len(ntu_biological)
                
            elif category_name in self.normal_categories:
                videos = self._get_videos_from_category(category_dir, 'normal')
                normal_videos.extend(videos)
                self.stats['normal'][category_name] = len(videos)
                
            elif category_name in self.abnormal_physical_categories:
                videos = self._get_videos_from_category(category_dir, 'abnormal_physical')
                abnormal_physical_videos.extend(videos)
                self.stats['abnormal_physical'][category_name] = len(videos)
                
            elif category_name in self.abnormal_biological_categories:
                videos = self._get_videos_from_category(category_dir, 'abnormal_biological')
                abnormal_biological_videos.extend(videos)
                self.stats['abnormal_biological'][category_name] = len(videos)
        
        self.stats['total_normal'] = len(normal_videos)
        self.stats['total_abnormal_physical'] = len(abnormal_physical_videos)
        self.stats['total_abnormal_biological'] = len(abnormal_biological_videos)
        
        print(f"Found {len(normal_videos)} normal videos, {len(abnormal_physical_videos)} abnormal physical videos, and {len(abnormal_biological_videos)} abnormal biological videos")
        
        return normal_videos, abnormal_physical_videos, abnormal_biological_videos
    
    def _process_ntu_videos(self, ntu_dir):
        """Process NTU RGB+ videos and classify them into 3 classes"""
        normal_videos = []
        abnormal_physical_videos = []
        abnormal_biological_videos = []
        
        for action_dir in ntu_dir.iterdir():
            if action_dir.is_dir():
                action_code = action_dir.name  # e.g., 'A001', 'A050', etc.
                
                # Determine classification based on action code
                if action_code in self.ntu_abnormal_physical_actions:
                    label = 'abnormal_physical'
                    target_list = abnormal_physical_videos
                elif action_code in self.ntu_abnormal_biological_actions:
                    label = 'abnormal_biological'
                    target_list = abnormal_biological_videos
                elif action_code in self.ntu_normal_actions:
                    label = 'normal'
                    target_list = normal_videos
                else:
                    # Default to normal for unlisted actions
                    label = 'normal'
                    target_list = normal_videos
                
                # Add all videos from this action directory
                for video_file in action_dir.glob('*.avi'):
                    target_list.append({
                        'path': video_file,
                        'label': label,
                        'category': 'NTU',
                        'action': action_code,
                        'filename': video_file.name
                    })
        
        return normal_videos, abnormal_physical_videos, abnormal_biological_videos
    
    def _get_videos_from_category(self, category_dir, label):
        """Get all video files from a category directory (non-NTU)"""
        videos = []
        
        # Handle flat structure for other categories
        for ext in ['*.mp4', '*.avi', '*.mov']:
            for video_file in category_dir.glob(ext):
                videos.append({
                    'path': video_file,
                    'label': label,
                    'category': category_dir.name,
                    'filename': video_file.name
                })
        
        return videos
    
    def split_dataset(self, videos):
        """Split videos into train/val/test sets"""
        print("Splitting dataset...")
        
        random.shuffle(videos)
        
        n_total = len(videos)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]
        
        return train_videos, val_videos, test_videos
    
    def copy_videos_and_create_annotations(self, normal_videos, abnormal_physical_videos, abnormal_biological_videos):
        """Copy videos to output structure and create annotation files"""
        print("Copying videos and creating annotations...")
        
        # Split all three types of videos
        normal_train, normal_val, normal_test = self.split_dataset(normal_videos)
        physical_train, physical_val, physical_test = self.split_dataset(abnormal_physical_videos)
        biological_train, biological_val, biological_test = self.split_dataset(abnormal_biological_videos)
        
        # Combine splits
        splits = {
            'train': normal_train + physical_train + biological_train,
            'val': normal_val + physical_val + biological_val,
            'test': normal_test + physical_test + biological_test
        }
        
        annotations = {}
        
        for split_name, videos in splits.items():
            print(f"Processing {split_name} split ({len(videos)} videos)...")
            
            annotations[split_name] = []
            
            for video_info in tqdm(videos, desc=f"Copying {split_name} videos"):
                # Determine output path
                label = video_info['label']
                new_filename = self._generate_new_filename(video_info, split_name)
                output_path = self.output_dir / 'videos' / split_name / label / new_filename
                
                # Copy video file
                try:
                    shutil.copy2(video_info['path'], output_path)
                    
                    # Determine label_id for 3-class classification
                    if label == 'normal':
                        label_id = 0
                    elif label == 'abnormal_physical':
                        label_id = 1
                    elif label == 'abnormal_biological':
                        label_id = 2
                    else:
                        label_id = 0  # default to normal
                    
                    # Add to annotations
                    annotations[split_name].append({
                        'filename': new_filename,
                        'label': label,
                        'label_id': label_id,
                        'original_category': video_info['category'],
                        'original_path': str(video_info['path']),
                        'split': split_name
                    })
                    
                except Exception as e:
                    print(f"Error copying {video_info['path']}: {e}")
        
        # Save annotations
        self._save_annotations(annotations)
        
        return annotations
    
    def _generate_new_filename(self, video_info, split):
        """Generate a new filename for the video"""
        category = video_info['category']
        label = video_info['label']
        original_name = Path(video_info['filename']).stem
        extension = Path(video_info['filename']).suffix
        
        # Create a unique identifier
        if category == 'NTU':
            action = video_info.get('action', 'unknown')
            new_name = f"{split}_{label}_{category}_{action}_{original_name}{extension}"
        else:
            new_name = f"{split}_{label}_{category}_{original_name}{extension}"
        
        return new_name
    
    def _save_annotations(self, annotations):
        """Save annotation files in multiple formats"""
        print("Saving annotation files...")
        
        # Save as JSON
        with open(self.output_dir / 'annotations' / 'dataset_annotations.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save as CSV for each split
        for split_name, videos in annotations.items():
            csv_path = self.output_dir / 'annotations' / f'{split_name}_annotations.csv'
            with open(csv_path, 'w') as f:
                f.write('filename,label,label_id,original_category,split\n')
                for video in videos:
                    f.write(f"{video['filename']},{video['label']},{video['label_id']},{video['original_category']},{video['split']}\n")
    
    def create_dataset_info(self, annotations):
        """Create dataset information file"""
        print("Creating dataset information...")
        
        # Calculate statistics for each split
        split_stats = {}
        for split_name, videos in annotations.items():
            normal_count = sum(1 for v in videos if v['label'] == 'normal')
            abnormal_physical_count = sum(1 for v in videos if v['label'] == 'abnormal_physical')
            abnormal_biological_count = sum(1 for v in videos if v['label'] == 'abnormal_biological')
            
            split_stats[split_name] = {
                'total': len(videos),
                'normal': normal_count,
                'abnormal_physical': abnormal_physical_count,
                'abnormal_biological': abnormal_biological_count,
                'physical_ratio': abnormal_physical_count / normal_count if normal_count > 0 else 0,
                'biological_ratio': abnormal_biological_count / normal_count if normal_count > 0 else 0
            }
        
        dataset_info = {
            'dataset_name': 'SPHAR 3-Class Activity Detection Dataset',
            'description': 'Dataset for 3-class classification: normal, abnormal physical, abnormal biological activities',
            'created_from': 'SPHAR Dataset + NTU RGB+',
            'classes': {
                'normal': {
                    'id': 0,
                    'description': 'Normal human activities',
                    'source_categories': list(self.normal_categories)
                },
                'abnormal_physical': {
                    'id': 1,
                    'description': 'Abnormal activities based on physical actions (violence, accidents)',
                    'source_categories': list(self.abnormal_physical_categories)
                },
                'abnormal_biological': {
                    'id': 2,
                    'description': 'Abnormal activities based on biological conditions (medical, falling)',
                    'source_categories': list(self.abnormal_biological_categories)
                }
            },
            'splits': split_stats,
            'total_videos': sum(len(videos) for videos in annotations.values()),
            'source_statistics': dict(self.stats)
        }
        
        # Save dataset info
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create README
        self._create_readme(dataset_info)
        
        return dataset_info
    
    def _create_readme(self, dataset_info):
        """Create a README file for the dataset"""
        readme_content = f"""# SPHAR 3-Class Activity Detection Dataset

## Overview
This dataset is created for training models to classify activities into 3 categories in surveillance videos.
It combines videos from the SPHAR dataset and NTU RGB+ dataset, categorized into normal, abnormal physical, and abnormal biological activities.

## Dataset Statistics
- **Total Videos**: {dataset_info['total_videos']}
- **Classes**: 3 (Normal, Abnormal Physical, Abnormal Biological)

### Split Distribution
"""
        
        for split_name, stats in dataset_info['splits'].items():
            readme_content += f"""
#### {split_name.title()} Split
- Total: {stats['total']} videos
- Normal: {stats['normal']} videos
- Abnormal Physical: {stats['abnormal_physical']} videos
- Abnormal Biological: {stats['abnormal_biological']} videos
- Physical Ratio: {stats['physical_ratio']:.2f}
- Biological Ratio: {stats['biological_ratio']:.2f}
"""
        
        readme_content += f"""
## Class Definitions

### Normal Activities (Label: 0)
Activities considered as normal human behavior:
{', '.join(dataset_info['classes']['normal']['source_categories'])}

### Abnormal Physical Activities (Label: 1)
Activities involving violence, accidents, or physical aggression:
{', '.join(dataset_info['classes']['abnormal_physical']['source_categories'])}

### Abnormal Biological Activities (Label: 2)
Activities involving medical conditions, falling, or biological distress:
{', '.join(dataset_info['classes']['abnormal_biological']['source_categories'])}

## Directory Structure
```
abnormal_detection_3class_dataset/
├── videos/
│   ├── train/
│   │   ├── normal/
│   │   ├── abnormal_physical/
│   │   └── abnormal_biological/
│   ├── val/
│   │   ├── normal/
│   │   ├── abnormal_physical/
│   │   └── abnormal_biological/
│   └── test/
│       ├── normal/
│       ├── abnormal_physical/
│       └── abnormal_biological/
├── annotations/
│   ├── dataset_annotations.json
│   ├── train_annotations.csv
│   ├── val_annotations.csv
│   └── test_annotations.csv
├── dataset_info.json
└── README.md
```

## Usage
The dataset is ready for training 3-class classification models for activity detection.
Each video is labeled as:
- Normal (0): Regular human activities
- Abnormal Physical (1): Violence, accidents, physical aggression
- Abnormal Biological (2): Medical conditions, falling, biological distress

## Citation
Please cite the original SPHAR dataset and NTU RGB+ dataset when using this derived dataset.
"""
        
        with open(self.output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def create_dataset(self):
        """Main method to create the dataset"""
        print("Starting dataset creation...")
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Create output structure
        self.create_output_structure()
        
        # Collect video files
        normal_videos, abnormal_physical_videos, abnormal_biological_videos = self.collect_video_files()
        
        if not normal_videos and not abnormal_physical_videos and not abnormal_biological_videos:
            print("No videos found! Please check the source directory.")
            return
        
        # Copy videos and create annotations
        annotations = self.copy_videos_and_create_annotations(normal_videos, abnormal_physical_videos, abnormal_biological_videos)
        
        # Create dataset info
        dataset_info = self.create_dataset_info(annotations)
        
        print("\n" + "="*50)
        print("Dataset creation completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total videos processed: {dataset_info['total_videos']}")
        print("="*50)
        
        return dataset_info

def main():
    parser = argparse.ArgumentParser(description='Create abnormal activity detection dataset')
    parser.add_argument('--source', '-s', 
                       default=r'D:\SPHAR-Dataset\videos',
                       help='Source directory containing SPHAR videos')
    parser.add_argument('--output', '-o',
                       default=r'D:\abnormal_detection_dataset',
                       help='Output directory for the new dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of videos for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Ratio of videos for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Ratio of videos for testing (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("Error: Train, validation, and test ratios must sum to 1.0")
        return
    
    # Create dataset
    creator = AbnormalDetectionDatasetCreator(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    dataset_info = creator.create_dataset()
    
    if dataset_info:
        print(f"\nDataset successfully created at: {args.output}")
        print("You can now use this dataset for training abnormal activity detection models!")

if __name__ == "__main__":
    main()
