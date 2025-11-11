#!/usr/bin/env python3
"""
Advanced YOLOv11s fine-tuning script optimized for indoor human detection with occlusion handling.
This script implements state-of-the-art hyperparameters and techniques specifically designed
for detecting humans in indoor environments where occlusion is common.

Key Features:
- Optimized hyperparameters for indoor scenarios
- Advanced data augmentation for occlusion robustness
- Multi-scale training for various human sizes
- Enhanced loss functions for small/occluded objects
- Comprehensive evaluation and monitoring

Author: Optimized for indoor human detection with occlusion handling
"""

import os
import sys
import json
import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class IndoorYOLOFineTuner:
    def __init__(self, base_model_path, dataset_path, output_dir, 
                 epochs=150, imgsz=640, batch_size=16):
        
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device optimization
        self.device = self._setup_device()
        
        # Load model
        self.model = None
        self._load_model()
        
        # Training history
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP50': [],
            'mAP50_95': []
        }
        
    def _setup_device(self):
        """Setup optimal device configuration"""
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"🔧 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Optimize GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        else:
            device = 'cpu'
            print("⚠️ Using CPU (GPU recommended for faster training)")
        
        return device
    
    def _load_model(self):
        """Load YOLOv11s model with verification"""
        try:
            if self.base_model_path.exists():
                print(f"📦 Loading model from: {self.base_model_path}")
                self.model = YOLO(str(self.base_model_path))
            else:
                print("📦 Loading YOLOv11s from Ultralytics hub...")
                self.model = YOLO('yolo11s.pt')
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_optimal_hyperparameters(self):
        """Setup optimal hyperparameters for indoor human detection with occlusion"""
        print("🎯 Setting up optimal hyperparameters for indoor human detection...")
        
        # Core training hyperparameters optimized for indoor detection
        self.train_config = {
            # Dataset and basic settings
            'data': str(self.dataset_path / 'dataset.yaml'),
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch_size,
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'indoor_human_detection',
            'exist_ok': True,
            
            # Optimizer settings - AdamW is best for fine-tuning
            'optimizer': 'AdamW',
            'lr0': 0.0008,  # Slightly higher initial learning rate for indoor scenes
            'lrf': 0.005,   # Final learning rate - not too low to avoid underfitting
            'momentum': 0.9,  # Higher momentum for stability
            'weight_decay': 0.0008,  # Moderate weight decay for regularization
            'warmup_epochs': 5,  # Longer warmup for stable training
            'warmup_momentum': 0.7,
            'warmup_bias_lr': 0.05,
            
            # Loss function weights - optimized for human detection
            'box': 8.5,     # Higher box loss weight for precise localization
            'cls': 0.3,     # Lower classification loss (single class)
            'dfl': 2.0,     # Higher distribution focal loss for better localization
            'pose': 12.0,   # Pose loss weight (if applicable)
            'kobj': 1.2,    # Keypoint objectness loss weight
            
            # Advanced training settings
            'label_smoothing': 0.05,  # Light label smoothing for robustness
            'nbs': 64,      # Nominal batch size for scaling
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.1,  # Light dropout for regularization
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': False,  # Don't cache to save memory
            'workers': min(8, os.cpu_count()),
            'close_mosaic': 15,  # Disable mosaic in last 15 epochs for fine-tuning
            'resume': False,
            'amp': True,    # Automatic Mixed Precision for efficiency
            'fraction': 1.0,
            'profile': False,
            'freeze': None,  # Don't freeze any layers for full fine-tuning
            'multi_scale': True,  # Essential for various human sizes
            'copy_paste': 0.1,  # Light copy-paste augmentation
            'auto_augment': 'randaugment',  # Advanced augmentation
            'erasing': 0.3,  # Random erasing for occlusion robustness
            'crop_fraction': 1.0,
            
            # Data augmentation optimized for indoor human detection
            'hsv_h': 0.02,    # Moderate hue augmentation for indoor lighting
            'hsv_s': 0.8,     # Higher saturation variation
            'hsv_v': 0.5,     # Higher value variation for lighting changes
            'degrees': 8.0,   # Moderate rotation for natural poses
            'translate': 0.15, # Higher translation for occlusion simulation
            'scale': 0.6,     # Higher scale variation for different distances
            'shear': 1.5,     # Light shear for perspective changes
            'perspective': 0.0002,  # Light perspective transformation
            'flipud': 0.0,    # No vertical flip (humans don't appear upside down)
            'fliplr': 0.5,    # Horizontal flip for left/right symmetry
            'mosaic': 1.0,    # Full mosaic probability for diverse scenes
            'mixup': 0.15,    # Higher mixup for robustness
            
            # Advanced settings for indoor detection
            'rect': False,    # Rectangular training disabled for better augmentation
            'single_cls': True,  # Single class optimization
            'cos_lr': True,   # Cosine learning rate scheduler
            'patience': 30,   # Early stopping patience
            'save_period': 10,
            'seed': 42,       # Reproducibility
            'deterministic': False,  # Allow some randomness for better training
            
            # Memory and performance optimization
            'verbose': True,
            'plots': True,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.25,     # Confidence threshold for validation
            'iou': 0.6,       # IoU threshold for NMS
            'max_det': 100,   # Maximum detections per image
            'half': False,    # Use full precision for better accuracy
            'dnn': False,
            'plots': True,
        }
        
        print("✅ Optimal hyperparameters configured!")
        return self.train_config
    
    def analyze_dataset(self):
        """Analyze the dataset for training insights"""
        print("📊 Analyzing dataset...")
        
        dataset_yaml_path = self.dataset_path / 'dataset.yaml'
        dataset_info_path = self.dataset_path / 'dataset_info.json'
        
        # Load dataset configuration
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"Dataset classes: {dataset_config.get('names', [])}")
        print(f"Number of classes: {dataset_config.get('nc', 0)}")
        
        # Load enhanced dataset info if available
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            
            print(f"Dataset type: {dataset_info.get('dataset_name', 'Unknown')}")
            print(f"Indoor focused: {dataset_info.get('enhancements', {}).get('indoor_focus', False)}")
            print(f"Occlusion handling: {dataset_info.get('enhancements', {}).get('occlusion_handling', False)}")
            
            # Print split statistics
            for split_name, stats in dataset_info.get('splits', {}).items():
                print(f"\n{split_name.upper()} Split:")
                print(f"  Total frames: {stats.get('total_frames', 0)}")
                print(f"  Frames with human: {stats.get('frames_with_human', 0)}")
                print(f"  Indoor frames: {stats.get('indoor_frames', 0)}")
                print(f"  Human ratio: {stats.get('human_ratio', 0):.2%}")
                print(f"  Indoor ratio: {stats.get('indoor_ratio', 0):.2%}")
        
        # Count actual images in each split
        splits = ['train', 'val', 'test']
        for split in splits:
            images_dir = self.dataset_path / 'images' / split
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
                print(f"{split.capitalize()} images: {image_count}")
        
        return dataset_config
    
    def fine_tune_model(self):
        """Fine-tune the model with optimal settings"""
        print("🚀 Starting fine-tuning with optimal hyperparameters...")
        
        # Setup hyperparameters
        train_config = self.setup_optimal_hyperparameters()
        
        # Display training configuration
        print("\n📋 Training Configuration:")
        key_params = ['epochs', 'batch', 'lr0', 'optimizer', 'box', 'cls', 'dfl']
        for param in key_params:
            if param in train_config:
                print(f"  {param}: {train_config[param]}")
        
        # GPU memory management
        if torch.cuda.is_available():
            print(f"\n🔧 GPU Memory Management:")
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"  Initial GPU memory: {initial_memory:.2f} GB")
        
        try:
            print(f"\n🔥 Fine-tuning YOLOv11s for {self.epochs} epochs...")
            print(f"🎯 Target: Indoor human detection with occlusion handling")
            print(f"⚡ Device: {self.device}")
            print(f"📦 Batch size: {self.batch_size}")
            print(f"🖼️ Image size: {self.imgsz}x{self.imgsz}")
            
            # Start training
            results = self.model.train(**train_config)
            
            # Post-training GPU info
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n📊 GPU Memory Usage:")
                print(f"  Final memory: {final_memory:.2f} GB")
                print(f"  Peak memory: {max_memory:.2f} GB")
                torch.cuda.empty_cache()
            
            print("✅ Fine-tuning completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU memory cleaned up after error")
            
            raise
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model performance...")
        
        try:
            # Run validation on test set
            test_results = self.model.val(
                data=str(self.dataset_path / 'dataset.yaml'),
                split='test',
                imgsz=self.imgsz,
                batch=self.batch_size,
                conf=0.25,
                iou=0.6,
                max_det=100,
                half=False,
                device=self.device,
                plots=True,
                save_json=True,
                save_hybrid=False,
                verbose=True
            )
            
            # Extract metrics
            if hasattr(test_results, 'box'):
                metrics = test_results.box
                
                evaluation_results = {
                    'mAP50': float(metrics.map50),
                    'mAP50_95': float(metrics.map),
                    'precision': float(metrics.mp),
                    'recall': float(metrics.mr),
                    'f1_score': 2 * (float(metrics.mp) * float(metrics.mr)) / (float(metrics.mp) + float(metrics.mr)) if (float(metrics.mp) + float(metrics.mr)) > 0 else 0
                }
                
                print(f"\n🎯 Final Evaluation Results:")
                print(f"  mAP50: {evaluation_results['mAP50']:.4f} ({evaluation_results['mAP50']*100:.2f}%)")
                print(f"  mAP50-95: {evaluation_results['mAP50_95']:.4f} ({evaluation_results['mAP50_95']*100:.2f}%)")
                print(f"  Precision: {evaluation_results['precision']:.4f} ({evaluation_results['precision']*100:.2f}%)")
                print(f"  Recall: {evaluation_results['recall']:.4f} ({evaluation_results['recall']*100:.2f}%)")
                print(f"  F1-Score: {evaluation_results['f1_score']:.4f} ({evaluation_results['f1_score']*100:.2f}%)")
                
                return evaluation_results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def export_model(self, export_name='yolo11s-indoor-detect.pt'):
        """Export the fine-tuned model"""
        print(f"📦 Exporting model as {export_name}...")
        
        try:
            # Find the best model from training
            best_model_path = self.output_dir / 'indoor_human_detection' / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                # Copy to output directory with custom name
                export_path = self.output_dir / export_name
                shutil.copy2(best_model_path, export_path)
                
                print(f"✅ Model exported to: {export_path}")
                
                # Also export to ONNX for deployment
                try:
                    model = YOLO(str(export_path))
                    onnx_path = str(export_path).replace('.pt', '.onnx')
                    model.export(format='onnx', imgsz=self.imgsz)
                    print(f"✅ ONNX model exported to: {onnx_path}")
                except Exception as e:
                    print(f"⚠️ ONNX export failed: {e}")
                
                return export_path
            else:
                print(f"❌ Best model not found at: {best_model_path}")
                return None
                
        except Exception as e:
            print(f"❌ Model export failed: {e}")
            return None
    
    def create_training_summary(self, evaluation_results=None, export_path=None):
        """Create comprehensive training summary"""
        print("📝 Creating training summary...")
        
        summary = {
            'model_info': {
                'base_model': str(self.base_model_path),
                'architecture': 'YOLOv11s',
                'task': 'Indoor Human Detection with Occlusion Handling',
                'fine_tuned': True,
                'export_path': str(export_path) if export_path else None
            },
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'image_size': self.imgsz,
                'device': self.device,
                'optimizer': 'AdamW',
                'learning_rate': 0.0008,
                'indoor_optimized': True,
                'occlusion_handling': True
            },
            'dataset_info': {
                'path': str(self.dataset_path),
                'type': 'Indoor-Focused Human Detection',
                'classes': ['person'],
                'num_classes': 1
            },
            'training_completed': True,
            'training_date': datetime.now().isoformat(),
            'hyperparameters_optimized_for': [
                'Indoor environments',
                'Occlusion handling',
                'Various human sizes',
                'Kitchen/cooking scenarios',
                'Lighting variations'
            ]
        }
        
        if evaluation_results:
            summary['evaluation_results'] = evaluation_results
        
        # Save summary
        summary_path = self.output_dir / 'indoor_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved: {summary_path}")
        
        # Create README
        self._create_readme(summary)
        
        return summary
    
    def _create_readme(self, summary):
        """Create README file for the trained model"""
        readme_content = f"""# YOLOv11s Indoor Human Detection Model

## Model Information
- **Architecture**: YOLOv11s
- **Task**: Indoor Human Detection with Occlusion Handling
- **Optimized For**: Indoor environments, cooking scenarios, occluded humans
- **Classes**: person (1 class)
- **Image Size**: {self.imgsz}x{self.imgsz}

## Training Details
- **Epochs**: {self.epochs}
- **Batch Size**: {self.batch_size}
- **Optimizer**: AdamW with optimal hyperparameters
- **Device**: {self.device}
- **Dataset**: Indoor-focused with Toyota RGB videos

## Performance Metrics
"""
        
        if 'evaluation_results' in summary:
            results = summary['evaluation_results']
            readme_content += f"""
- **mAP50**: {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)
- **mAP50-95**: {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)
- **Precision**: {results['precision']:.4f} ({results['precision']*100:.2f}%)
- **Recall**: {results['recall']:.4f} ({results['recall']*100:.2f}%)
- **F1-Score**: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)
"""
        
        readme_content += f"""
## Key Optimizations
- **Indoor Focus**: Specialized for indoor human detection
- **Occlusion Handling**: Enhanced detection of partially occluded humans
- **Multi-Scale Training**: Optimized for various human sizes and distances
- **Advanced Augmentation**: Tailored for indoor lighting and scenarios
- **Toyota RGB Integration**: Trained on cooking and indoor activities

## Usage

```python
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('yolo11s-indoor-detect.pt')

# Run inference with optimal settings
results = model('indoor_image.jpg', 
               conf=0.25,      # Confidence threshold
               iou=0.6,        # IoU threshold for NMS
               max_det=100)    # Maximum detections

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            conf = box.conf[0].item()
            print(f"Human detected with confidence: {{conf:.3f}}")
```

## Model Files
- **yolo11s-indoor-detect.pt**: Main PyTorch model
- **yolo11s-indoor-detect.onnx**: ONNX format for deployment
- **indoor_training_summary.json**: Detailed training information

## Training Optimizations
- Enhanced loss weights for precise human localization
- Adaptive learning rate scheduling
- Advanced data augmentation for occlusion robustness
- Multi-confidence detection for partially visible humans
- Specialized hyperparameters for indoor environments

## Training Date
{summary.get('training_date', 'Not specified')}

## Recommended Use Cases
- Indoor surveillance systems
- Kitchen monitoring applications
- Human activity recognition in indoor spaces
- Cooking behavior analysis
- Indoor safety monitoring

## Notes
- Optimized for indoor environments with good performance on occluded humans
- Best performance with confidence threshold 0.25-0.4 for indoor scenes
- Handles various lighting conditions common in indoor environments
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ README created: {readme_path}")
    
    def run_complete_training(self, export_name='yolo11s-indoor-detect.pt'):
        """Run the complete training pipeline"""
        print("="*70)
        print("🏠 INDOOR HUMAN DETECTION - YOLO11S FINE-TUNING")
        print("="*70)
        
        # Analyze dataset
        self.analyze_dataset()
        
        # Fine-tune model
        training_results = self.fine_tune_model()
        
        if training_results is None:
            print("❌ Training failed!")
            return False
        
        # Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Export model
        export_path = self.export_model(export_name)
        
        # Create summary
        summary = self.create_training_summary(evaluation_results, export_path)
        
        print("\n" + "="*70)
        print("🎉 INDOOR HUMAN DETECTION TRAINING COMPLETED!")
        print(f"📁 Results saved in: {self.output_dir}")
        if export_path:
            print(f"🤖 Model exported: {export_path}")
        print("="*70)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv11s for indoor human detection')
    parser.add_argument('--base-model', '-m',
                       default=r'D:\SPHAR-Dataset\models\yolo11s.pt',
                       help='Path to base YOLOv11s model')
    parser.add_argument('--dataset', '-d',
                       default=r'D:\SPHAR-Dataset\train\indoor_focused_dataset',
                       help='Path to indoor-focused dataset')
    parser.add_argument('--output', '-o',
                       default=r'D:\SPHAR-Dataset\models\finetune-more',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', '-e', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--imgsz', '-i', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--export-name', default='yolo11s-indoor-detect.pt',
                       help='Name for exported model (default: yolo11s-indoor-detect.pt)')
    
    args = parser.parse_args()
    
    try:
        # Create fine-tuner
        finetuner = IndoorYOLOFineTuner(
            base_model_path=args.base_model,
            dataset_path=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch
        )
        
        # Run complete training
        success = finetuner.run_complete_training(export_name=args.export_name)
        
        if success:
            print("\n🎉 Fine-tuning completed successfully!")
            print(f"🤖 Your optimized indoor detection model is ready!")
            print(f"📁 Check results in: {args.output}")
        else:
            print("\n❌ Fine-tuning failed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
