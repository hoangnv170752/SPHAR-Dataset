#!/usr/bin/env python3
"""
Script to train YOLO model for human detection using the created dataset.

This script will:
1. Load the human detection dataset
2. Train YOLOv8 model for human detection
3. Evaluate the model performance
4. Save the trained model

Author: Generated for human detection training
"""

import os
import json
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

class HumanDetectionTrainer:
    def __init__(self, dataset_path, model_name='yolov8n.pt', epochs=100, imgsz=640, batch_size=16):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Check for dataset.yaml
        self.config_path = self.dataset_path / 'dataset.yaml'
        if not self.config_path.exists():
            raise ValueError(f"Dataset config file not found: {self.config_path}")
        
        # Load dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Set up output directory
        self.output_dir = self.dataset_path / 'training_results'
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_dataset_info(self):
        """Load dataset information"""
        info_path = self.dataset_path / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_dataset_config(self):
        """Load and validate dataset configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Dataset Configuration:")
        print(f"  Path: {config.get('path', 'Not specified')}")
        print(f"  Classes: {config.get('nc', 'Not specified')}")
        print(f"  Class names: {config.get('names', 'Not specified')}")
        
        return config
    
    def print_dataset_stats(self):
        """Print dataset statistics"""
        if not self.dataset_info:
            print("No dataset info available")
            return
        
        print("\\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Dataset: {self.dataset_info.get('dataset_name', 'Unknown')}")
        print(f"Total Frames: {self.dataset_info.get('total_frames', 'Unknown')}")
        
        splits = self.dataset_info.get('splits', {})
        for split_name, stats in splits.items():
            print(f"\\n{split_name.upper()} Split:")
            print(f"  Total frames: {stats.get('total_frames', 0)}")
            print(f"  Frames with human: {stats.get('frames_with_human', 0)}")
            print(f"  Frames without human: {stats.get('frames_without_human', 0)}")
            print(f"  Human ratio: {stats.get('human_ratio', 0):.2%}")
        
        print("="*50)
    
    def setup_training_environment(self):
        """Setup training environment and check requirements"""
        print("Setting up training environment...")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")
        
        # Load and validate dataset config
        config = self._load_dataset_config()
        
        return config
    
    def train_model(self, resume=False, save_period=10):
        """Train the YOLO model"""
        print("\\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        
        try:
            # Initialize model
            print(f"Loading model: {self.model_name}")
            model = YOLO(self.model_name)
            
            # Training parameters
            train_args = {
                'data': str(self.config_path),
                'epochs': self.epochs,
                'imgsz': self.imgsz,
                'batch': self.batch_size,
                'project': str(self.output_dir),
                'name': 'human_detection_training',
                'save_period': save_period,
                'patience': 50,  # Early stopping patience
                'save': True,
                'plots': True,
                'val': True,
                'resume': resume
            }
            
            print("Training parameters:")
            for key, value in train_args.items():
                print(f"  {key}: {value}")
            
            print("\\nStarting training...")
            results = model.train(**train_args)
            
            print("\\nTraining completed!")
            return model, results
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def evaluate_model(self, model):
        """Evaluate the trained model"""
        print("\\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        try:
            # Run validation
            print("Running validation...")
            val_results = model.val(data=str(self.config_path))
            
            # Print key metrics
            if hasattr(val_results, 'box'):
                metrics = val_results.box
                print(f"\\nValidation Results:")
                print(f"  mAP50: {metrics.map50:.4f}")
                print(f"  mAP50-95: {metrics.map:.4f}")
                print(f"  Precision: {metrics.mp:.4f}")
                print(f"  Recall: {metrics.mr:.4f}")
            
            return val_results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
    
    def test_model_on_samples(self, model, num_samples=5):
        """Test model on sample images"""
        print("\\n" + "="*50)
        print("TESTING ON SAMPLE IMAGES")
        print("="*50)
        
        try:
            # Get test images
            test_img_dir = self.dataset_path / 'images' / 'test'
            if not test_img_dir.exists():
                print("Test images directory not found")
                return
            
            # Get sample images
            image_files = list(test_img_dir.glob('*.jpg'))[:num_samples]
            
            if not image_files:
                print("No test images found")
                return
            
            print(f"Testing on {len(image_files)} sample images...")
            
            for img_path in image_files:
                print(f"\\nTesting: {img_path.name}")
                
                # Run inference
                results = model(str(img_path))
                
                # Print results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        print(f"  Detected {len(boxes)} human(s)")
                        for i, box in enumerate(boxes):
                            conf = box.conf[0].item()
                            print(f"    Human {i+1}: confidence = {conf:.3f}")
                    else:
                        print("  No humans detected")
                
                # Save result image
                result_path = self.output_dir / 'sample_results' / f"result_{img_path.name}"
                result_path.parent.mkdir(exist_ok=True)
                
                for result in results:
                    result.save(str(result_path))
                
                print(f"  Result saved: {result_path}")
        
        except Exception as e:
            print(f"Error during testing: {e}")
    
    def save_training_summary(self, model, results, val_results):
        """Save training summary and model info"""
        print("\\nSaving training summary...")
        
        try:
            summary = {
                'model_name': self.model_name,
                'training_parameters': {
                    'epochs': self.epochs,
                    'image_size': self.imgsz,
                    'batch_size': self.batch_size
                },
                'dataset_info': self.dataset_info,
                'training_completed': True
            }
            
            # Add validation metrics if available
            if val_results and hasattr(val_results, 'box'):
                metrics = val_results.box
                summary['validation_metrics'] = {
                    'mAP50': float(metrics.map50),
                    'mAP50_95': float(metrics.map),
                    'precision': float(metrics.mp),
                    'recall': float(metrics.mr)
                }
            
            # Save summary
            summary_path = self.output_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Training summary saved: {summary_path}")
            
            # Save best model to a standard location
            if model:
                best_model_path = self.output_dir / 'best_human_detection_model.pt'
                model.save(str(best_model_path))
                print(f"Best model saved: {best_model_path}")
        
        except Exception as e:
            print(f"Error saving training summary: {e}")
    
    def run_complete_training(self, resume=False, test_samples=True):
        """Run complete training pipeline"""
        print("="*60)
        print("HUMAN DETECTION MODEL TRAINING")
        print("="*60)
        
        # Print dataset statistics
        self.print_dataset_stats()
        
        # Setup environment
        config = self.setup_training_environment()
        
        # Train model
        model, results = self.train_model(resume=resume)
        
        if model is None:
            print("Training failed!")
            return False
        
        # Evaluate model
        val_results = self.evaluate_model(model)
        
        # Test on samples
        if test_samples:
            self.test_model_on_samples(model)
        
        # Save summary
        self.save_training_summary(model, results, val_results)
        
        print("\\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED!")
        print(f"Results saved in: {self.output_dir}")
        print("="*60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for human detection')
    parser.add_argument('--dataset', '-d',
                       default=r'D:\\SPHAR-Dataset\\train\\human_detection_dataset',
                       help='Path to human detection dataset')
    parser.add_argument('--model', '-m', default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', '-i', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing on sample images')
    
    args = parser.parse_args()
    
    try:
        # Create trainer
        trainer = HumanDetectionTrainer(
            dataset_path=args.dataset,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch
        )
        
        # Run training
        success = trainer.run_complete_training(
            resume=args.resume,
            test_samples=not args.no_test
        )
        
        if success:
            print("\\nTraining completed successfully!")
            print("\\nNext steps:")
            print("1. Check training results in the output directory")
            print("2. Use the trained model for inference")
            print("3. Fine-tune parameters if needed")
        else:
            print("\\nTraining failed. Please check the error messages above.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
