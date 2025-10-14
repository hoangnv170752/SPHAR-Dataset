"""
YOLO v11 Human Detection Fine-tuning Script
Fine-tune YOLO v11 for human detection using the abnormal detection dataset
"""

import os
import sys
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import shutil

class YOLOHumanDetectionTrainer:
    def __init__(self, dataset_path, model_path, output_dir):
        """
        Initialize the YOLO trainer
        
        Args:
            dataset_path (str): Path to the abnormal detection dataset
            model_path (str): Path to the pre-trained YOLO model
            output_dir (str): Directory to save training outputs
        """
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.runs_dir = self.output_dir / 'runs'
        self.datasets_dir = self.output_dir / 'datasets'
        self.models_dir = self.output_dir / 'models'
        
        for dir_path in [self.runs_dir, self.datasets_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Dataset path: {self.dataset_path}")
        print(f"Model path: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        
    def create_yolo_dataset_structure(self):
        """
        Create YOLO-compatible dataset structure from video dataset
        """
        print("Creating YOLO dataset structure...")
        
        yolo_dataset_path = self.datasets_dir / 'human_detection_yolo'
        yolo_dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (yolo_dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (yolo_dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
        return yolo_dataset_path
        
    def extract_frames_and_create_labels(self, yolo_dataset_path, frames_per_video=5):
        """
        Extract frames from videos and create YOLO labels for human detection
        
        Args:
            yolo_dataset_path (Path): Path to YOLO dataset directory
            frames_per_video (int): Number of frames to extract per video
        """
        print("Extracting frames and creating labels...")
        
        for split in ['train', 'val', 'test']:
            split_video_dir = self.dataset_path / 'videos' / split
            split_images_dir = yolo_dataset_path / 'images' / split
            split_labels_dir = yolo_dataset_path / 'labels' / split
            
            if not split_video_dir.exists():
                continue
                
            print(f"Processing {split} split...")
            
            # Get all video files from all categories
            video_files = []
            for category_dir in split_video_dir.iterdir():
                if category_dir.is_dir():
                    video_files.extend(list(category_dir.glob('*.mp4')))
                    video_files.extend(list(category_dir.glob('*.avi')))
                    
            # Process each video
            for video_path in tqdm(video_files, desc=f"Processing {split} videos"):
                self._extract_frames_from_video(
                    video_path, split_images_dir, split_labels_dir, frames_per_video
                )
                
    def _extract_frames_from_video(self, video_path, images_dir, labels_dir, frames_per_video):
        """
        Extract frames from a single video and create labels
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return
            
        # Calculate frame indices to extract
        if total_frames <= frames_per_video:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // frames_per_video
            frame_indices = [i * step for i in range(frames_per_video)]
            
        video_name = video_path.stem
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Save frame
            frame_name = f"{video_name}_frame_{i:03d}.jpg"
            frame_path = images_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            
            # Create label file (assuming human is present in all frames)
            # For now, we'll create a bounding box covering the center area
            # In a real scenario, you'd use a human detection model or manual annotation
            label_path = labels_dir / f"{video_name}_frame_{i:03d}.txt"
            
            # Create a simple label assuming human presence
            # Format: class_id center_x center_y width height (normalized)
            # Class 0 for person/human
            with open(label_path, 'w') as f:
                # Simple assumption: human occupies center 60% of the frame
                f.write("0 0.5 0.5 0.6 0.8\n")
                
        cap.release()
        
    def create_dataset_yaml(self, yolo_dataset_path):
        """
        Create YAML configuration file for YOLO training
        """
        yaml_content = {
            'path': str(yolo_dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # number of classes
            'names': ['person']  # class names
        }
        
        yaml_path = yolo_dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f"Dataset YAML created: {yaml_path}")
        return yaml_path
        
    def train_model(self, yaml_path, epochs=100, imgsz=640, batch_size=16):
        """
        Train YOLO model for human detection
        
        Args:
            yaml_path (Path): Path to dataset YAML file
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size for training
        """
        print("Starting YOLO training...")
        
        # Load pre-trained model
        model = YOLO(str(self.model_path))
        
        # Train the model
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='human_detection',
            project=str(self.runs_dir),
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=4,
            patience=20,  # Early stopping patience
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # DFL loss gain
            pose=12.0,  # Pose loss gain (for pose models)
            kobj=2.0,  # Keypoint objective loss gain (for pose models)
            label_smoothing=0.0,
            nbs=64,  # Nominal batch size
            hsv_h=0.015,  # Image HSV-Hue augmentation (fraction)
            hsv_s=0.7,  # Image HSV-Saturation augmentation (fraction)
            hsv_v=0.4,  # Image HSV-Value augmentation (fraction)
            degrees=0.0,  # Image rotation (+/- deg)
            translate=0.1,  # Image translation (+/- fraction)
            scale=0.5,  # Image scale (+/- gain)
            shear=0.0,  # Image shear (+/- deg)
            perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
            flipud=0.0,  # Image flip up-down (probability)
            fliplr=0.5,  # Image flip left-right (probability)
            mosaic=1.0,  # Image mosaic (probability)
            mixup=0.0,  # Image mixup (probability)
            copy_paste=0.0  # Segment copy-paste (probability)
        )
        
        print("Training completed!")
        return results
        
    def evaluate_model(self, yaml_path, model_path=None):
        """
        Evaluate the trained model
        
        Args:
            yaml_path (Path): Path to dataset YAML file
            model_path (str): Path to trained model (if None, uses best from training)
        """
        print("Evaluating model...")
        
        if model_path is None:
            # Use the best model from training
            model_path = self.runs_dir / 'human_detection' / 'weights' / 'best.pt'
            
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return None
            
        model = YOLO(str(model_path))
        
        # Validate the model
        results = model.val(
            data=str(yaml_path),
            imgsz=640,
            batch=16,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dnn=False,
            plots=True,
            rect=False,
            split='val'
        )
        
        print("Evaluation completed!")
        return results
        
    def save_training_info(self, results, yaml_path):
        """
        Save training information and configuration
        """
        info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'model_path': str(self.model_path),
            'yaml_path': str(yaml_path),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'pytorch_version': torch.__version__,
        }
        
        info_path = self.output_dir / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"Training info saved: {info_path}")
        
    def run_full_pipeline(self, epochs=100, imgsz=640, batch_size=16, frames_per_video=5):
        """
        Run the complete training pipeline
        """
        print("="*60)
        print("YOLO v11 Human Detection Fine-tuning Pipeline")
        print("="*60)
        
        try:
            # Step 1: Create YOLO dataset structure
            yolo_dataset_path = self.create_yolo_dataset_structure()
            
            # Step 2: Extract frames and create labels
            self.extract_frames_and_create_labels(yolo_dataset_path, frames_per_video)
            
            # Step 3: Create dataset YAML
            yaml_path = self.create_dataset_yaml(yolo_dataset_path)
            
            # Step 4: Train model
            results = self.train_model(yaml_path, epochs, imgsz, batch_size)
            
            # Step 5: Evaluate model
            eval_results = self.evaluate_model(yaml_path)
            
            # Step 6: Save training info
            self.save_training_info(results, yaml_path)
            
            print("="*60)
            print("Training pipeline completed successfully!")
            print(f"Results saved in: {self.output_dir}")
            print("="*60)
            
            return results, eval_results
            
        except Exception as e:
            print(f"Error in training pipeline: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO v11 for human detection')
    parser.add_argument('--dataset', '-d', 
                       default=r'D:\abnormal_detection_dataset',
                       help='Path to abnormal detection dataset')
    parser.add_argument('--model', '-m',
                       default=r'D:\SPHAR-Dataset\models\yolo11s.pt',
                       help='Path to pre-trained YOLO model')
    parser.add_argument('--output', '-o',
                       default=r'D:\SPHAR-Dataset\train\human_detection_results',
                       help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--frames-per-video', type=int, default=5,
                       help='Number of frames to extract per video')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = YOLOHumanDetectionTrainer(
        dataset_path=args.dataset,
        model_path=args.model,
        output_dir=args.output
    )
    
    # Run training pipeline
    trainer.run_full_pipeline(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        frames_per_video=args.frames_per_video
    )


if __name__ == "__main__":
    main()
