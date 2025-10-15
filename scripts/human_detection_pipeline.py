#!/usr/bin/env python3
"""
Complete pipeline for human detection training and classification.

This script provides a complete workflow:
1. Create human detection dataset from videos
2. Train YOLO model for human detection
3. Classify frames using trained model
4. Generate reports and statistics

Author: Generated for complete human detection pipeline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

class HumanDetectionPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.scripts_dir = self.base_dir / 'scripts'
        self.videos_dir = self.base_dir / 'videos'
        self.train_dir = self.base_dir / 'train'
        
        # Script paths
        self.create_dataset_script = self.scripts_dir / 'create_human_detection_dataset.py'
        self.train_script = self.scripts_dir / 'train_human_detection.py'
        self.classify_script = self.scripts_dir / 'classify_frames_with_human.py'
    
    def check_requirements(self):
        """Check if all required files and dependencies exist"""
        print("Checking requirements...")
        
        # Check script files
        required_scripts = [
            self.create_dataset_script,
            self.train_script,
            self.classify_script
        ]
        
        missing_scripts = []
        for script in required_scripts:
            if not script.exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print("Missing required scripts:")
            for script in missing_scripts:
                print(f"  - {script}")
            return False
        
        # Check Python packages
        try:
            import cv2
            import ultralytics
            import yaml
            import tqdm
            print("✓ All required packages are installed")
        except ImportError as e:
            print(f"Missing required package: {e}")
            print("Please install: pip install opencv-python ultralytics pyyaml tqdm")
            return False
        
        print("✓ All requirements satisfied")
        return True
    
    def create_dataset(self, videos_dir=None, output_dir=None, frame_interval=30, **kwargs):
        """Step 1: Create human detection dataset"""
        print("\\n" + "="*60)
        print("STEP 1: CREATING HUMAN DETECTION DATASET")
        print("="*60)
        
        if videos_dir is None:
            videos_dir = self.videos_dir
        if output_dir is None:
            output_dir = self.train_dir / 'human_detection_dataset'
        
        cmd = [
            sys.executable, str(self.create_dataset_script),
            '--source', str(videos_dir),
            '--output', str(output_dir),
            '--frame-interval', str(frame_interval)
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if key.startswith('dataset_'):
                arg_name = key.replace('dataset_', '').replace('_', '-')
                cmd.extend([f'--{arg_name}', str(value)])
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Dataset creation completed successfully")
            return True, str(output_dir)
        except subprocess.CalledProcessError as e:
            print(f"✗ Dataset creation failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False, None
    
    def train_model(self, dataset_dir, model_name='yolov8n.pt', epochs=100, **kwargs):
        """Step 2: Train YOLO model"""
        print("\\n" + "="*60)
        print("STEP 2: TRAINING HUMAN DETECTION MODEL")
        print("="*60)
        
        cmd = [
            sys.executable, str(self.train_script),
            '--dataset', str(dataset_dir),
            '--model', model_name,
            '--epochs', str(epochs)
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if key.startswith('train_'):
                arg_name = key.replace('train_', '').replace('_', '-')
                cmd.extend([f'--{arg_name}', str(value)])
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Model training completed successfully")
            
            # Find trained model path
            training_results_dir = Path(dataset_dir) / 'training_results'
            model_path = training_results_dir / 'best_human_detection_model.pt'
            
            if model_path.exists():
                return True, str(model_path)
            else:
                # Look for YOLO training results
                yolo_results = list(training_results_dir.glob('**/weights/best.pt'))
                if yolo_results:
                    return True, str(yolo_results[0])
                else:
                    print("Warning: Could not find trained model file")
                    return True, None
                    
        except subprocess.CalledProcessError as e:
            print(f"✗ Model training failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False, None
    
    def classify_frames(self, model_path, input_dir, output_dir, mode='videos', **kwargs):
        """Step 3: Classify frames using trained model"""
        print("\\n" + "="*60)
        print("STEP 3: CLASSIFYING FRAMES WITH TRAINED MODEL")
        print("="*60)
        
        cmd = [
            sys.executable, str(self.classify_script),
            '--model', str(model_path),
            '--input', str(input_dir),
            '--output', str(output_dir),
            '--mode', mode
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if key.startswith('classify_'):
                arg_name = key.replace('classify_', '').replace('_', '-')
                cmd.extend([f'--{arg_name}', str(value)])
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Frame classification completed successfully")
            return True, str(output_dir)
        except subprocess.CalledProcessError as e:
            print(f"✗ Frame classification failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False, None
    
    def run_complete_pipeline(self, videos_dir=None, **kwargs):
        """Run the complete pipeline"""
        print("="*80)
        print("HUMAN DETECTION COMPLETE PIPELINE")
        print("="*80)
        
        # Check requirements
        if not self.check_requirements():
            print("\\n✗ Requirements check failed. Please fix the issues above.")
            return False
        
        # Step 1: Create dataset
        dataset_success, dataset_dir = self.create_dataset(
            videos_dir=videos_dir,
            frame_interval=kwargs.get('frame_interval', 30),
            dataset_train_ratio=kwargs.get('train_ratio', 0.7),
            dataset_val_ratio=kwargs.get('val_ratio', 0.15),
            dataset_test_ratio=kwargs.get('test_ratio', 0.15)
        )
        
        if not dataset_success:
            print("\\n✗ Pipeline failed at dataset creation step")
            return False
        
        # Step 2: Train model
        train_success, model_path = self.train_model(
            dataset_dir=dataset_dir,
            model_name=kwargs.get('model_name', 'yolov8n.pt'),
            epochs=kwargs.get('epochs', 100),
            train_imgsz=kwargs.get('imgsz', 640),
            train_batch=kwargs.get('batch_size', 16)
        )
        
        if not train_success:
            print("\\n✗ Pipeline failed at model training step")
            return False
        
        # Step 3: Classify frames (optional)
        if kwargs.get('classify_input'):
            classify_success, classify_output = self.classify_frames(
                model_path=model_path,
                input_dir=kwargs['classify_input'],
                output_dir=kwargs.get('classify_output', self.base_dir / 'classified_frames'),
                mode=kwargs.get('classify_mode', 'videos'),
                classify_confidence=kwargs.get('confidence', 0.5),
                classify_frame_interval=kwargs.get('classify_frame_interval', 30)
            )
            
            if not classify_success:
                print("\\n✗ Pipeline failed at frame classification step")
                return False
        
        # Generate final report
        self.generate_final_report(dataset_dir, model_path, kwargs.get('classify_output'))
        
        print("\\n" + "="*80)
        print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*80)
        
        return True
    
    def generate_final_report(self, dataset_dir, model_path, classify_output=None):
        """Generate final pipeline report"""
        print("\\nGenerating final report...")
        
        report_path = self.base_dir / 'human_detection_pipeline_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HUMAN DETECTION PIPELINE REPORT\\n")
            f.write("="*50 + "\\n\\n")
            
            # Dataset info
            dataset_info_path = Path(dataset_dir) / 'dataset_info.json'
            if dataset_info_path.exists():
                with open(dataset_info_path, 'r') as df:
                    dataset_info = json.load(df)
                
                f.write("DATASET INFORMATION:\\n")
                f.write(f"  Dataset path: {dataset_dir}\\n")
                f.write(f"  Total frames: {dataset_info.get('total_frames', 'Unknown')}\\n")
                
                splits = dataset_info.get('splits', {})
                for split_name, stats in splits.items():
                    f.write(f"  {split_name}: {stats.get('total_frames', 0)} frames\\n")
                f.write("\\n")
            
            # Model info
            f.write("MODEL INFORMATION:\\n")
            f.write(f"  Model path: {model_path}\\n")
            
            # Training results
            training_summary_path = Path(dataset_dir) / 'training_results' / 'training_summary.json'
            if training_summary_path.exists():
                with open(training_summary_path, 'r') as tf:
                    training_info = json.load(tf)
                
                metrics = training_info.get('validation_metrics', {})
                f.write(f"  mAP50: {metrics.get('mAP50', 'N/A')}\\n")
                f.write(f"  mAP50-95: {metrics.get('mAP50_95', 'N/A')}\\n")
                f.write(f"  Precision: {metrics.get('precision', 'N/A')}\\n")
                f.write(f"  Recall: {metrics.get('recall', 'N/A')}\\n")
            f.write("\\n")
            
            # Classification results
            if classify_output:
                f.write("CLASSIFICATION RESULTS:\\n")
                f.write(f"  Output path: {classify_output}\\n")
                
                stats_path = Path(classify_output) / 'metadata' / 'classification_stats.json'
                if stats_path.exists():
                    with open(stats_path, 'r') as cf:
                        classify_stats = json.load(cf)
                    
                    f.write(f"  Total frames processed: {classify_stats.get('total_frames_processed', 0)}\\n")
                    f.write(f"  Frames with human: {classify_stats.get('frames_with_human', 0)}\\n")
                    f.write(f"  Frames without human: {classify_stats.get('frames_without_human', 0)}\\n")
                f.write("\\n")
            
            f.write("PIPELINE COMPLETED SUCCESSFULLY!\\n")
        
        print(f"Final report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Complete human detection pipeline')
    
    # Input/Output paths
    parser.add_argument('--videos-dir', default=None,
                       help='Directory containing source videos')
    parser.add_argument('--base-dir', default=None,
                       help='Base directory for the project')
    
    # Dataset creation parameters
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Extract 1 frame every N frames (default: 30)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation data ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test data ratio (default: 0.15)')
    
    # Training parameters
    parser.add_argument('--model-name', default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    
    # Classification parameters (optional)
    parser.add_argument('--classify-input', default=None,
                       help='Input directory for frame classification (optional)')
    parser.add_argument('--classify-output', default=None,
                       help='Output directory for classified frames')
    parser.add_argument('--classify-mode', choices=['videos', 'images'], default='videos',
                       help='Classification mode (default: videos)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for classification (default: 0.5)')
    parser.add_argument('--classify-frame-interval', type=int, default=30,
                       help='Frame interval for classification (default: 30)')
    
    # Pipeline control
    parser.add_argument('--step', choices=['dataset', 'train', 'classify', 'all'], default='all',
                       help='Which step to run (default: all)')
    
    args = parser.parse_args()
    
    try:
        # Create pipeline
        pipeline = HumanDetectionPipeline(base_dir=args.base_dir)
        
        # Convert args to kwargs
        kwargs = vars(args)
        
        if args.step == 'all':
            # Run complete pipeline
            success = pipeline.run_complete_pipeline(**kwargs)
        elif args.step == 'dataset':
            # Only create dataset
            success, _ = pipeline.create_dataset(
                videos_dir=args.videos_dir,
                frame_interval=args.frame_interval
            )
        elif args.step == 'train':
            # Only train model (requires existing dataset)
            dataset_dir = pipeline.train_dir / 'human_detection_dataset'
            success, _ = pipeline.train_model(
                dataset_dir=dataset_dir,
                model_name=args.model_name,
                epochs=args.epochs
            )
        elif args.step == 'classify':
            # Only classify frames (requires trained model)
            if not args.classify_input:
                print("Error: --classify-input is required for classification step")
                return
            
            # Find trained model
            model_path = pipeline.train_dir / 'human_detection_dataset' / 'training_results' / 'best_human_detection_model.pt'
            if not model_path.exists():
                print(f"Error: Trained model not found at {model_path}")
                return
            
            success, _ = pipeline.classify_frames(
                model_path=model_path,
                input_dir=args.classify_input,
                output_dir=args.classify_output or (pipeline.base_dir / 'classified_frames'),
                mode=args.classify_mode
            )
        
        if success:
            print("\\n✓ Pipeline completed successfully!")
        else:
            print("\\n✗ Pipeline failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
