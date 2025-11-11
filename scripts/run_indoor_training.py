#!/usr/bin/env python3
"""
Convenient runner script for indoor human detection training pipeline.
This script orchestrates the complete process from dataset creation to model training.

Usage:
    python run_indoor_training.py --mode full          # Complete pipeline
    python run_indoor_training.py --mode dataset       # Only create dataset
    python run_indoor_training.py --mode train         # Only train model
    python run_indoor_training.py --mode quick         # Quick training (50 epochs)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

class IndoorTrainingRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = self.base_dir / 'scripts'
        
        # Default paths
        self.sphar_videos = self.base_dir / 'videos'
        self.toyota_videos = self.base_dir / 'videos' / 'toyota'
        self.dataset_output = self.base_dir / 'train' / 'indoor_focused_dataset'
        self.models_dir = self.base_dir / 'models'
        self.base_model = self.models_dir / 'yolo11s.pt'
        self.finetune_output = self.models_dir / 'finetune-more'
        
    def check_prerequisites(self):
        """Check if all prerequisites are available"""
        print("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check directories
        if not self.sphar_videos.exists():
            issues.append(f"SPHAR videos directory not found: {self.sphar_videos}")
        
        if not self.toyota_videos.exists():
            issues.append(f"Toyota videos directory not found: {self.toyota_videos}")
        
        # Check Toyota videos
        toyota_video_count = len(list(self.toyota_videos.glob('*.mp4')))
        if toyota_video_count == 0:
            issues.append(f"No Toyota videos found in: {self.toyota_videos}")
        else:
            print(f"✅ Found {toyota_video_count} Toyota videos")
        
        # Check base model
        if not self.base_model.exists():
            print(f"⚠️ Base model not found at {self.base_model}, will download automatically")
        else:
            print(f"✅ Base model found: {self.base_model}")
        
        # Check scripts
        dataset_script = self.scripts_dir / 'create_indoor_focused_dataset.py'
        training_script = self.scripts_dir / 'finetune_indoor_yolo11s.py'
        
        if not dataset_script.exists():
            issues.append(f"Dataset creation script not found: {dataset_script}")
        
        if not training_script.exists():
            issues.append(f"Training script not found: {training_script}")
        
        if issues:
            print("❌ Prerequisites check failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ All prerequisites satisfied!")
        return True
    
    def create_dataset(self, **kwargs):
        """Create indoor-focused dataset"""
        print("\n" + "="*60)
        print("📊 CREATING INDOOR-FOCUSED DATASET")
        print("="*60)
        
        dataset_script = self.scripts_dir / 'create_indoor_focused_dataset.py'
        
        cmd = [
            sys.executable, str(dataset_script),
            '--source', str(self.sphar_videos),
            '--toyota', str(self.toyota_videos),
            '--output', str(self.dataset_output)
        ]
        
        # Add optional parameters
        if 'interval' in kwargs:
            cmd.extend(['--interval', str(kwargs['interval'])])
        if 'max_videos' in kwargs:
            cmd.extend(['--max-videos', str(kwargs['max_videos'])])
        if 'min_size' in kwargs:
            cmd.extend(['--min-size', str(kwargs['min_size'])])
        if 'conf_threshold' in kwargs:
            cmd.extend(['--conf-threshold', str(kwargs['conf_threshold'])])
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("✅ Dataset creation completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Dataset creation failed: {e}")
            return False
    
    def train_model(self, **kwargs):
        """Train the model"""
        print("\n" + "="*60)
        print("🚀 TRAINING INDOOR HUMAN DETECTION MODEL")
        print("="*60)
        
        # Check if dataset exists
        if not (self.dataset_output / 'dataset.yaml').exists():
            print(f"❌ Dataset not found at {self.dataset_output}")
            print("Please run dataset creation first or use --mode full")
            return False
        
        training_script = self.scripts_dir / 'finetune_indoor_yolo11s.py'
        
        cmd = [
            sys.executable, str(training_script),
            '--base-model', str(self.base_model),
            '--dataset', str(self.dataset_output),
            '--output', str(self.finetune_output)
        ]
        
        # Add optional parameters
        if 'epochs' in kwargs:
            cmd.extend(['--epochs', str(kwargs['epochs'])])
        if 'batch' in kwargs:
            cmd.extend(['--batch', str(kwargs['batch'])])
        if 'imgsz' in kwargs:
            cmd.extend(['--imgsz', str(kwargs['imgsz'])])
        if 'export_name' in kwargs:
            cmd.extend(['--export-name', str(kwargs['export_name'])])
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("✅ Model training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Model training failed: {e}")
            return False
    
    def run_full_pipeline(self, **kwargs):
        """Run the complete pipeline"""
        print("🏠 INDOOR HUMAN DETECTION - COMPLETE PIPELINE")
        print("="*70)
        
        start_time = time.time()
        
        # Step 1: Create dataset
        print("\n📊 Step 1: Creating indoor-focused dataset...")
        if not self.create_dataset(**kwargs):
            print("❌ Pipeline failed at dataset creation")
            return False
        
        # Step 2: Train model
        print("\n🚀 Step 2: Training model...")
        if not self.train_model(**kwargs):
            print("❌ Pipeline failed at model training")
            return False
        
        # Success
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print(f"⏱️ Total time: {duration/3600:.1f} hours ({duration/60:.1f} minutes)")
        print(f"📁 Results saved in: {self.finetune_output}")
        print(f"🤖 Model: {self.finetune_output / kwargs.get('export_name', 'yolo11s-indoor-detect.pt')}")
        print("="*70)
        
        return True
    
    def run_quick_training(self, **kwargs):
        """Run quick training for testing"""
        print("⚡ QUICK TRAINING MODE")
        
        # Override parameters for quick training
        quick_params = {
            'epochs': 50,
            'batch': 8,
            'imgsz': 416,
            'interval': 30,  # Less frequent frame extraction
            'max_videos': 50,  # Fewer videos
            'export_name': 'yolo11s-indoor-detect-quick.pt'
        }
        
        # Merge with user parameters
        quick_params.update(kwargs)
        
        return self.run_full_pipeline(**quick_params)

def main():
    parser = argparse.ArgumentParser(description='Indoor Human Detection Training Runner')
    parser.add_argument('--mode', choices=['full', 'dataset', 'train', 'quick'], 
                       default='full', help='Training mode')
    
    # Dataset creation parameters
    parser.add_argument('--interval', type=int, default=20,
                       help='Frame extraction interval (default: 20)')
    parser.add_argument('--max-videos', type=int, default=100,
                       help='Maximum videos per category (default: 100)')
    parser.add_argument('--min-size', type=float, default=0.02,
                       help='Minimum human size ratio (default: 0.02)')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for occlusion detection (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--export-name', default='yolo11s-indoor-detect.pt',
                       help='Export model name (default: yolo11s-indoor-detect.pt)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = IndoorTrainingRunner()
    
    # Check prerequisites
    if not runner.check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues above.")
        return 1
    
    # Convert args to kwargs
    kwargs = {
        'interval': args.interval,
        'max_videos': args.max_videos,
        'min_size': args.min_size,
        'conf_threshold': args.conf_threshold,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'export_name': args.export_name
    }
    
    # Run based on mode
    success = False
    
    if args.mode == 'full':
        success = runner.run_full_pipeline(**kwargs)
    elif args.mode == 'dataset':
        success = runner.create_dataset(**kwargs)
    elif args.mode == 'train':
        success = runner.train_model(**kwargs)
    elif args.mode == 'quick':
        success = runner.run_quick_training(**kwargs)
    
    if success:
        print(f"\n✅ {args.mode.upper()} mode completed successfully!")
        return 0
    else:
        print(f"\n❌ {args.mode.upper()} mode failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
