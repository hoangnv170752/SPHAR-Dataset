#!/usr/bin/env python3
"""
Script to classify video frames as 'with_human' or 'without_human' using trained YOLO model.

This script will:
1. Load trained human detection model
2. Process video frames or directories of images
3. Classify frames based on human presence
4. Create organized dataset for further training

Author: Generated for human detection classification
"""

import os
import cv2
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

class HumanFrameClassifier:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model()
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'frames_with_human': 0,
            'frames_without_human': 0,
            'videos_processed': 0,
            'categories': defaultdict(lambda: {'with_human': 0, 'without_human': 0})
        }
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            print(f"Loading model from: {self.model_path}")
            model = YOLO(str(self.model_path))
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_human_in_frame(self, frame):
        """
        Detect human in frame and return classification
        Returns: (has_human, confidence, num_humans, boxes)
        """
        try:
            results = self.model(frame, verbose=False)
            
            has_human = False
            max_confidence = 0.0
            num_humans = 0
            human_boxes = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Filter by confidence threshold
                    high_conf_mask = boxes.conf >= self.confidence_threshold
                    if high_conf_mask.any():
                        high_conf_boxes = boxes[high_conf_mask]
                        has_human = True
                        num_humans = len(high_conf_boxes)
                        max_confidence = float(high_conf_boxes.conf.max())
                        
                        # Store box coordinates
                        for box in high_conf_boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            human_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            
            return has_human, max_confidence, num_humans, human_boxes
            
        except Exception as e:
            print(f"Error in human detection: {e}")
            return False, 0.0, 0, []
    
    def process_single_image(self, image_path):
        """Process a single image file"""
        try:
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Error: Cannot read image {image_path}")
                return None
            
            # Detect human
            has_human, confidence, num_humans, boxes = self.detect_human_in_frame(frame)
            
            result = {
                'filename': image_path.name,
                'path': str(image_path),
                'has_human': has_human,
                'confidence': confidence,
                'num_humans': num_humans,
                'boxes': boxes,
                'image_size': frame.shape[:2]  # (height, width)
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_video_frames(self, video_path, frame_interval=30, max_frames=None):
        """Extract and classify frames from video"""
        print(f"Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_count = 0
        extracted_count = 0
        
        # Calculate number of frames to process
        frames_to_process = total_frames // frame_interval
        if max_frames:
            frames_to_process = min(frames_to_process, max_frames)
        
        with tqdm(total=frames_to_process, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame at intervals
                if frame_count % frame_interval == 0:
                    # Detect human
                    has_human, confidence, num_humans, boxes = self.detect_human_in_frame(frame)
                    
                    # Create frame data
                    video_stem = video_path.stem
                    frame_filename = f"{video_stem}_frame_{frame_count:06d}.jpg"
                    
                    frame_data = {
                        'frame': frame.copy(),
                        'filename': frame_filename,
                        'has_human': has_human,
                        'confidence': confidence,
                        'num_humans': num_humans,
                        'boxes': boxes,
                        'video_name': video_path.name,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'image_size': frame.shape[:2]
                    }
                    
                    frames_data.append(frame_data)
                    extracted_count += 1
                    
                    pbar.update(1)
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        cap.release()
        print(f"Extracted and classified {extracted_count} frames from {video_path.name}")
        return frames_data
    
    def process_directory(self, input_dir, output_dir, frame_interval=30, max_frames_per_video=None):
        """Process all videos in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        print(f"Processing directory: {input_path}")
        print(f"Output directory: {output_path}")
        
        # Create output structure
        self._create_output_structure(output_path)
        
        all_results = []
        
        # Process each category directory
        for category_dir in input_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            print(f"\\nProcessing category: {category_name}")
            
            # Find video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(category_dir.glob(ext))
            
            if not video_files:
                print(f"No video files found in {category_name}")
                continue
            
            category_results = []
            
            # Process each video
            for video_path in video_files:
                frames_data = self.process_video_frames(
                    video_path, frame_interval, max_frames_per_video
                )
                
                # Add category info
                for frame_data in frames_data:
                    frame_data['category'] = category_name
                
                category_results.extend(frames_data)
                self.stats['videos_processed'] += 1
            
            # Update statistics
            with_human = sum(1 for f in category_results if f['has_human'])
            without_human = len(category_results) - with_human
            
            self.stats['categories'][category_name]['with_human'] = with_human
            self.stats['categories'][category_name]['without_human'] = without_human
            self.stats['frames_with_human'] += with_human
            self.stats['frames_without_human'] += without_human
            
            all_results.extend(category_results)
        
        self.stats['total_frames_processed'] = len(all_results)
        
        # Save results
        self._save_classified_frames(all_results, output_path)
        self._save_statistics(output_path)
        
        return all_results
    
    def _create_output_structure(self, output_path):
        """Create output directory structure"""
        # Create directories for classified frames
        (output_path / 'frames' / 'with_human').mkdir(parents=True, exist_ok=True)
        (output_path / 'frames' / 'without_human').mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        (output_path / 'metadata').mkdir(parents=True, exist_ok=True)
        (output_path / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def _save_classified_frames(self, results, output_path):
        """Save classified frames to organized directories"""
        print("\\nSaving classified frames...")
        
        annotations = {
            'with_human': [],
            'without_human': []
        }
        
        for frame_data in tqdm(results, desc="Saving frames"):
            # Determine output directory
            if frame_data['has_human']:
                class_dir = 'with_human'
            else:
                class_dir = 'without_human'
            
            # Save frame image
            frame_path = output_path / 'frames' / class_dir / frame_data['filename']
            cv2.imwrite(str(frame_path), frame_data['frame'])
            
            # Create annotation entry
            annotation = {
                'filename': frame_data['filename'],
                'has_human': frame_data['has_human'],
                'confidence': frame_data['confidence'],
                'num_humans': frame_data['num_humans'],
                'boxes': frame_data['boxes'],
                'category': frame_data.get('category', 'unknown'),
                'video_name': frame_data.get('video_name', 'unknown'),
                'frame_number': frame_data.get('frame_number', 0),
                'timestamp': frame_data.get('timestamp', 0),
                'image_size': frame_data['image_size']
            }
            
            annotations[class_dir].append(annotation)
        
        # Save annotations
        annotations_path = output_path / 'annotations' / 'classified_frames.json'
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save CSV files
        for class_name, class_annotations in annotations.items():
            csv_path = output_path / 'annotations' / f'{class_name}_frames.csv'
            with open(csv_path, 'w') as f:
                f.write('filename,has_human,confidence,num_humans,category,video_name,frame_number,timestamp\\n')
                for ann in class_annotations:
                    f.write(f"{ann['filename']},{ann['has_human']},{ann['confidence']:.3f},"
                           f"{ann['num_humans']},{ann['category']},{ann['video_name']},"
                           f"{ann['frame_number']},{ann['timestamp']:.2f}\\n")
        
        print(f"Saved {len(results)} classified frames")
        print(f"Frames with human: {len(annotations['with_human'])}")
        print(f"Frames without human: {len(annotations['without_human'])}")
    
    def _save_statistics(self, output_path):
        """Save classification statistics"""
        stats_path = output_path / 'metadata' / 'classification_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create summary report
        report_path = output_path / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("HUMAN DETECTION CLASSIFICATION REPORT\\n")
            f.write("="*50 + "\\n\\n")
            f.write(f"Model used: {self.model_path}\\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\\n\\n")
            f.write(f"Total frames processed: {self.stats['total_frames_processed']}\\n")
            f.write(f"Frames with human: {self.stats['frames_with_human']}\\n")
            f.write(f"Frames without human: {self.stats['frames_without_human']}\\n")
            f.write(f"Videos processed: {self.stats['videos_processed']}\\n\\n")
            
            f.write("CATEGORY BREAKDOWN:\\n")
            f.write("-" * 30 + "\\n")
            for category, counts in self.stats['categories'].items():
                total = counts['with_human'] + counts['without_human']
                human_ratio = counts['with_human'] / total if total > 0 else 0
                f.write(f"{category}:\\n")
                f.write(f"  Total: {total}\\n")
                f.write(f"  With human: {counts['with_human']} ({human_ratio:.1%})\\n")
                f.write(f"  Without human: {counts['without_human']}\\n\\n")
        
        print(f"Statistics saved to: {stats_path}")
        print(f"Report saved to: {report_path}")
    
    def process_images_directory(self, images_dir, output_dir):
        """Process directory containing individual images"""
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        
        print(f"Processing images directory: {images_path}")
        
        # Create output structure
        self._create_output_structure(output_path)
        
        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_path.glob(ext))
        
        if not image_files:
            print("No image files found!")
            return []
        
        print(f"Found {len(image_files)} images")
        
        results = []
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(image_path)
            if result:
                # Add frame data for consistency
                result['frame'] = cv2.imread(str(image_path))
                result['category'] = 'images'
                result['video_name'] = 'N/A'
                result['frame_number'] = 0
                result['timestamp'] = 0
                results.append(result)
        
        # Update statistics
        with_human = sum(1 for r in results if r['has_human'])
        without_human = len(results) - with_human
        
        self.stats['total_frames_processed'] = len(results)
        self.stats['frames_with_human'] = with_human
        self.stats['frames_without_human'] = without_human
        self.stats['categories']['images']['with_human'] = with_human
        self.stats['categories']['images']['without_human'] = without_human
        
        # Save results
        self._save_classified_frames(results, output_path)
        self._save_statistics(output_path)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Classify frames based on human presence')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory (videos or images)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for classified frames')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for human detection (default: 0.5)')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Extract 1 frame every N frames from videos (default: 30)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to extract per video (default: unlimited)')
    parser.add_argument('--mode', choices=['videos', 'images'], default='videos',
                       help='Input mode: videos or images (default: videos)')
    
    args = parser.parse_args()
    
    try:
        # Create classifier
        classifier = HumanFrameClassifier(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # Process input
        if args.mode == 'videos':
            results = classifier.process_directory(
                input_dir=args.input,
                output_dir=args.output,
                frame_interval=args.frame_interval,
                max_frames_per_video=args.max_frames
            )
        else:
            results = classifier.process_images_directory(
                images_dir=args.input,
                output_dir=args.output
            )
        
        print("\\n" + "="*60)
        print("CLASSIFICATION COMPLETED!")
        print(f"Total frames processed: {len(results)}")
        print(f"Frames with human: {classifier.stats['frames_with_human']}")
        print(f"Frames without human: {classifier.stats['frames_without_human']}")
        print(f"Output saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
