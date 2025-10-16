#!/usr/bin/env python3
"""
Script to fine-tune YOLOv11s model specifically for human detection in various conditions.
This script will:
1. Load pre-trained YOLOv11s model
2. Fine-tune on human detection dataset
3. Optimize for different lighting, poses, and environments
4. Export as yolo11s-detect.pt

Author: Generated for human detection fine-tuning
"""

import os
import json
import yaml
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import threading
import time
import psutil

class YOLOv11HumanFineTuner:
    def __init__(self, base_model_path, dataset_path, output_dir, epochs=100, imgsz=640, batch_size=16):
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability and optimize settings
        self._check_gpu_setup()
        
        # Validate inputs
        self._validate_inputs()
        
        # Load model
        self.model = None
        self._load_base_model()
        
    def _validate_inputs(self):
        """Validate input paths and files"""
        if not self.base_model_path.exists():
            raise ValueError(f"Base model not found: {self.base_model_path}")
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path not found: {self.dataset_path}")
        
        # Check for dataset.yaml
        dataset_yaml = self.dataset_path / 'dataset.yaml'
        if not dataset_yaml.exists():
            raise ValueError(f"Dataset config not found: {dataset_yaml}")
        
        print("✅ Input validation passed")
    
    def _check_gpu_setup(self):
        """Check GPU availability and optimize settings"""
        print("🔍 Checking GPU setup...")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU Count: {gpu_count}")
            print(f"✅ Current GPU: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            
            # GPU memory recommendations (but don't auto-change)
            if gpu_memory < 6:  # Less than 6GB
                if self.batch_size > 8:
                    print(f"⚠️ Warning: Batch size {self.batch_size} may be too large for {gpu_memory:.1f}GB GPU")
                    print(f"� Recommended batch size: 8 or lower")
                    print(f"🔧 Consider using --batch 8 if you encounter OOM errors")
            elif gpu_memory < 12:  # 6-12GB
                if self.batch_size > 16:
                    print(f"⚠️ Warning: Batch size {self.batch_size} may be too large for {gpu_memory:.1f}GB GPU")
                    print(f"💡 Recommended batch size: 16 or lower")
            else:  # 12GB+
                if self.batch_size < 32:
                    print(f"💡 High-end GPU detected. You can increase batch size to 32+ for faster training")
            
            print(f"🎯 Using batch size: {self.batch_size}")
            
            # Set device
            self.device = f'cuda:{current_device}'
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
            torch.backends.cudnn.allow_tf32 = True
            
            print(f"🚀 GPU optimizations enabled")
            
        else:
            print("⚠️ CUDA not available, using CPU")
            print("💡 For faster training, consider using a GPU-enabled environment")
            self.device = 'cpu'
            
            # Reduce batch size for CPU
            if self.batch_size > 4:
                print(f"🔧 Reducing batch size to 4 for CPU training")
                self.batch_size = 4
    
    def _load_base_model(self):
        """Load the base YOLOv11s model"""
        try:
            print(f"Loading base model: {self.base_model_path}")
            self.model = YOLO(str(self.base_model_path))
            print("✅ Base model loaded successfully")
            
            # Print model info
            print(f"Model architecture: {self.model.model}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            raise ValueError(f"Failed to load base model: {e}")
    
    def setup_training_config(self):
        """Setup optimized training configuration for human detection"""
        print("Setting up training configuration...")
        
        # Training hyperparameters optimized for human detection
        self.train_config = {
            'data': str(self.dataset_path / 'dataset.yaml'),
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch_size,
            'project': str(self.output_dir),
            'name': 'human_detection_finetune',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Better for fine-tuning
            'lr0': 0.001,  # Lower learning rate for fine-tuning
            'lrf': 0.01,   # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,    # Box loss weight
            'cls': 0.5,    # Classification loss weight
            'dfl': 1.5,    # Distribution focal loss weight
            'pose': 12.0,  # Pose loss weight (if applicable)
            'kobj': 1.0,   # Keypoint objectness loss weight
            'label_smoothing': 0.0,
            'nbs': 64,     # Nominal batch size
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': False,  # Don't cache images to save memory
            'device': self.device,  # Use optimized device
            'workers': min(8, os.cpu_count()),  # Optimize workers based on CPU cores
            'close_mosaic': 10,  # Disable mosaic augmentation in last 10 epochs
            'resume': False,
            'amp': True,     # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,  # Don't freeze any layers for fine-tuning
            'multi_scale': True,  # Multi-scale training
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',  # Advanced augmentation
            'erasing': 0.4,  # Random erasing probability
            'crop_fraction': 1.0,
        }
        
        # Data augmentation optimized for human detection
        self.augmentation_config = {
            'hsv_h': 0.015,    # Hue augmentation
            'hsv_s': 0.7,      # Saturation augmentation
            'hsv_v': 0.4,      # Value augmentation
            'degrees': 10.0,   # Rotation degrees
            'translate': 0.1,  # Translation fraction
            'scale': 0.5,      # Scale factor
            'shear': 2.0,      # Shear degrees
            'perspective': 0.0001,  # Perspective transformation
            'flipud': 0.0,     # Vertical flip probability
            'fliplr': 0.5,     # Horizontal flip probability
            'mosaic': 1.0,     # Mosaic probability
            'mixup': 0.1,      # Mixup probability
        }
        
        # Merge augmentation config into training config
        self.train_config.update(self.augmentation_config)
        
        print("✅ Training configuration setup complete")
        return self.train_config
    
    def analyze_dataset(self):
        """Analyze the dataset to understand class distribution"""
        print("Analyzing dataset...")
        
        dataset_yaml_path = self.dataset_path / 'dataset.yaml'
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"Dataset classes: {dataset_config.get('names', [])}")
        print(f"Number of classes: {dataset_config.get('nc', 0)}")
        
        # Count images in each split
        splits = ['train', 'val', 'test']
        dataset_stats = {}
        
        for split in splits:
            images_dir = self.dataset_path / 'images' / split
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
                dataset_stats[split] = image_count
                print(f"{split.capitalize()} images: {image_count}")
        
        return dataset_stats
    
    def create_custom_yaml(self):
        """Create custom dataset YAML with optimized settings"""
        print("Creating custom dataset configuration...")
        
        custom_yaml_path = self.output_dir / 'human_detection_config.yaml'
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['person'],
            
            # Custom settings for human detection
            'download': False,
            'yaml_file': str(custom_yaml_path),
            
            # Optimization settings
            'rect': True,  # Rectangular training
            'single_cls': True,  # Single class (person only)
            'model': str(self.base_model_path),
        }
        
        with open(custom_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Custom config saved: {custom_yaml_path}")
        return custom_yaml_path
    
    def fine_tune_model(self):
        """Fine-tune the YOLOv11s model for human detection"""
        print("🚀 Starting fine-tuning process...")
        
        # Setup training configuration
        train_config = self.setup_training_config()
        
        # Use existing dataset.yaml instead of creating custom one
        existing_yaml = self.dataset_path / 'dataset.yaml'
        if existing_yaml.exists():
            train_config['data'] = str(existing_yaml)
            print(f"✅ Using existing dataset.yaml: {existing_yaml}")
        else:
            # Fallback to custom yaml
            custom_yaml = self.create_custom_yaml()
            train_config['data'] = str(custom_yaml)
            print(f"⚠️ Created custom dataset.yaml: {custom_yaml}")
        
        print("Training configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        
        # GPU memory management
        if torch.cuda.is_available():
            print(f"\n🔧 GPU Memory Management:")
            torch.cuda.empty_cache()  # Clear cache before training
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"  Initial GPU memory: {initial_memory:.2f} GB")
            
            # Enable memory optimization
            if hasattr(torch.cuda, 'memory_efficient_attention'):
                torch.cuda.memory_efficient_attention = True
        
        try:
            # Start training with memory monitoring
            print(f"\n🔥 Fine-tuning YOLOv11s for {self.epochs} epochs...")
            print(f"🎯 Target: Human detection optimization")
            print(f"⚡ Device: {self.device}")
            print(f"📦 Batch size: {self.batch_size}")
            print(f"🖼️ Image size: {self.imgsz}x{self.imgsz}")
            
            results = self.model.train(**train_config)
            
            # Post-training GPU info
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"\n📊 GPU Memory Usage:")
                print(f"  Final memory: {final_memory:.2f} GB")
                print(f"  Peak memory: {max_memory:.2f} GB")
                torch.cuda.empty_cache()  # Clean up after training
            
            print("✅ Fine-tuning completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            
            # GPU memory cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU memory cleaned up after error")
            
            raise
    
    def validate_model(self):
        """Validate the fine-tuned model"""
        print("🔍 Validating fine-tuned model...")
        
        try:
            # Run validation
            val_results = self.model.val(
                data=str(self.output_dir / 'human_detection_config.yaml'),
                imgsz=self.imgsz,
                batch=self.batch_size,
                save_json=True,
                plots=True
            )
            
            # Print validation metrics
            if hasattr(val_results, 'box'):
                metrics = val_results.box
                print(f"\n📊 Validation Results:")
                print(f"  mAP50: {metrics.map50:.4f}")
                print(f"  mAP50-95: {metrics.map:.4f}")
                print(f"  Precision: {metrics.mp:.4f}")
                print(f"  Recall: {metrics.mr:.4f}")
                print(f"  F1-Score: {2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr):.4f}")
            
            return val_results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None
    
    def export_model(self, export_name='yolo11s-detect.pt'):
        """Export the fine-tuned model"""
        print(f"📦 Exporting model as {export_name}...")
        
        try:
            # Find the best model from training
            best_model_path = self.output_dir / 'human_detection_finetune' / 'weights' / 'best.pt'
            
            if not best_model_path.exists():
                print("❌ Best model not found, using current model")
                best_model_path = self.base_model_path
            
            # Load the best model
            best_model = YOLO(str(best_model_path))
            
            # Export to specified location
            export_path = self.output_dir / export_name
            
            # Copy the model file
            import shutil
            shutil.copy2(best_model_path, export_path)
            
            print(f"✅ Model exported to: {export_path}")
            
            # Also export in different formats
            try:
                # Export to ONNX for deployment
                onnx_path = self.output_dir / export_name.replace('.pt', '.onnx')
                best_model.export(format='onnx', imgsz=self.imgsz)
                print(f"✅ ONNX model exported")
                
                # Export to TensorRT if available
                try:
                    best_model.export(format='engine', imgsz=self.imgsz)
                    print(f"✅ TensorRT model exported")
                except:
                    print("⚠️ TensorRT export skipped (not available)")
                    
            except Exception as e:
                print(f"⚠️ Additional format export failed: {e}")
            
            return export_path
            
        except Exception as e:
            print(f"❌ Model export failed: {e}")
            return None
    
    def test_on_samples(self, num_samples=10):
        """Test the fine-tuned model on sample images"""
        print(f"🧪 Testing model on {num_samples} sample images...")
        
        try:
            # Get test images
            test_images_dir = self.dataset_path / 'images' / 'test'
            if not test_images_dir.exists():
                test_images_dir = self.dataset_path / 'images' / 'val'
            
            if not test_images_dir.exists():
                print("⚠️ No test images found")
                return
            
            # Get sample images
            image_files = list(test_images_dir.glob('*.jpg'))[:num_samples]
            
            if not image_files:
                print("⚠️ No test images found")
                return
            
            # Create results directory
            results_dir = self.output_dir / 'test_results'
            results_dir.mkdir(exist_ok=True)
            
            print(f"Testing on {len(image_files)} images...")
            
            detection_stats = {'with_human': 0, 'without_human': 0, 'total': 0}
            
            for img_path in tqdm(image_files, desc="Testing"):
                try:
                    # Run inference
                    results = self.model(str(img_path), conf=0.25, iou=0.45)
                    
                    # Check if humans detected
                    has_human = False
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            has_human = True
                            break
                    
                    # Update stats
                    detection_stats['total'] += 1
                    if has_human:
                        detection_stats['with_human'] += 1
                    else:
                        detection_stats['without_human'] += 1
                    
                    # Save result image
                    result_path = results_dir / f"result_{img_path.name}"
                    for result in results:
                        result.save(str(result_path))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Print test statistics
            print(f"\n📈 Test Results:")
            print(f"  Total images tested: {detection_stats['total']}")
            print(f"  Images with human detected: {detection_stats['with_human']}")
            print(f"  Images without human: {detection_stats['without_human']}")
            if detection_stats['total'] > 0:
                detection_rate = detection_stats['with_human'] / detection_stats['total']
                print(f"  Human detection rate: {detection_rate:.2%}")
            
            print(f"✅ Test results saved to: {results_dir}")
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
    
    def save_training_summary(self, results, val_results, export_path):
        """Save comprehensive training summary"""
        print("💾 Saving training summary...")
        
        summary = {
            'model_info': {
                'base_model': str(self.base_model_path),
                'architecture': 'YOLOv11s',
                'task': 'Human Detection',
                'fine_tuned': True,
                'export_path': str(export_path) if export_path else None
            },
            'training_config': self.train_config,
            'dataset_info': {
                'path': str(self.dataset_path),
                'classes': ['person'],
                'num_classes': 1
            },
            'training_completed': True,
            'epochs_trained': self.epochs,
            'image_size': self.imgsz,
            'batch_size': self.batch_size
        }
        
        # Add validation metrics if available
        if val_results and hasattr(val_results, 'box'):
            metrics = val_results.box
            summary['validation_metrics'] = {
                'mAP50': float(metrics.map50),
                'mAP50_95': float(metrics.map),
                'precision': float(metrics.mp),
                'recall': float(metrics.mr),
                'f1_score': float(2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr)) if (metrics.mp + metrics.mr) > 0 else 0.0
            }
        
        # Save summary
        summary_path = self.output_dir / 'finetune_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved: {summary_path}")
        
        # Create README
        self._create_model_readme(summary)
    
    def _create_model_readme(self, summary):
        """Create README for the fine-tuned model"""
        readme_content = f"""# YOLOv11s Human Detection Model

## Model Information
- **Base Model**: YOLOv11s
- **Task**: Human Detection
- **Fine-tuned**: Yes
- **Classes**: person (1 class)
- **Image Size**: {self.imgsz}x{self.imgsz}

## Training Details
- **Epochs**: {self.epochs}
- **Batch Size**: {self.batch_size}
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Dataset**: {self.dataset_path.name}

## Performance Metrics
"""
        
        if 'validation_metrics' in summary:
            metrics = summary['validation_metrics']
            readme_content += f"""
- **mAP50**: {metrics['mAP50']:.4f}
- **mAP50-95**: {metrics['mAP50_95']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
"""
        
        readme_content += f"""
## Usage

```python
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('yolo11s-detect.pt')

# Run inference
results = model('image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            print(f"Human detected with confidence: {{box.conf[0]:.2f}}")
```

## Model Files
- **yolo11s-detect.pt**: Main PyTorch model
- **yolo11s-detect.onnx**: ONNX format for deployment
- **finetune_summary.json**: Detailed training information

## Fine-tuning Optimizations
- Optimized for human detection in various conditions
- Enhanced data augmentation for robustness
- Lower learning rate for stable fine-tuning
- Multi-scale training for better generalization
- Automatic Mixed Precision for efficiency

## Training Date
{summary.get('training_date', 'Not specified')}
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ Model README created: {readme_path}")
    
    def run_complete_finetune(self, test_samples=True, export_name='yolo11s-detect.pt'):
        """Run the complete fine-tuning pipeline"""
        print("="*80)
        print("🎯 YOLOv11s HUMAN DETECTION FINE-TUNING")
        print("="*80)
        
        try:
            # Analyze dataset
            dataset_stats = self.analyze_dataset()
            
            # Fine-tune model
            results = self.fine_tune_model()
            
            # Validate model
            val_results = self.validate_model()
            
            # Export model
            export_path = self.export_model(export_name)
            
            # Test on samples
            if test_samples:
                self.test_on_samples()
            
            # Plot training charts
            self.plot_training_charts(results)
            
            # Save summary
            self.save_training_summary(results, val_results, export_path)
            
            print("\n" + "="*80)
            print("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📁 Output directory: {self.output_dir}")
            if export_path:
                print(f"🤖 Fine-tuned model: {export_path}")
            print("📊 Check finetune_summary.json for detailed results")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_training_charts(self, results):
        """Plot training accuracy and loss charts like the provided image"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from pathlib import Path
            
            print("📊 Creating training charts...")
            
            # Get results CSV file
            results_dir = self.output_dir / 'human_detection_finetune'
            csv_file = results_dir / 'results.csv'
            
            if not csv_file.exists():
                print(f"⚠️ Results CSV not found: {csv_file}")
                return
            
            # Read training results
            df = pd.read_csv(csv_file)
            
            # Create figure with 2x2 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('YOLOv11s Human Detection Fine-tuning Results', fontsize=16, fontweight='bold')
            
            # Extract metrics (YOLO CSV columns)
            epochs = df.index + 1
            
            # Chart 1: Model Accuracy (Top Left)
            if 'metrics/precision(B)' in df.columns and 'val/precision(B)' in df.columns:
                ax1.plot(epochs, df['metrics/precision(B)'], 'b-', label='Training Precision', linewidth=2)
                ax1.plot(epochs, df['val/precision(B)'], 'r-', label='Validation Precision', linewidth=2)
            elif 'train/precision' in df.columns and 'val/precision' in df.columns:
                ax1.plot(epochs, df['train/precision'], 'b-', label='Training Precision', linewidth=2)
                ax1.plot(epochs, df['val/precision'], 'r-', label='Validation Precision', linewidth=2)
            
            ax1.set_title('Model Precision', fontweight='bold')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Precision')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Chart 2: Model Loss (Top Right)
            if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                ax2.plot(epochs, df['train/box_loss'], 'b-', label='Training Loss', linewidth=2)
                ax2.plot(epochs, df['val/box_loss'], 'r-', label='Validation Loss', linewidth=2)
            elif 'loss' in df.columns:
                ax2.plot(epochs, df['loss'], 'b-', label='Training Loss', linewidth=2)
            
            ax2.set_title('Model Loss', fontweight='bold')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Model Recall (Bottom Left)
            if 'metrics/recall(B)' in df.columns and 'val/recall(B)' in df.columns:
                ax3.plot(epochs, df['metrics/recall(B)'], 'b-', label='Training Recall', linewidth=2)
                ax3.plot(epochs, df['val/recall(B)'], 'r-', label='Validation Recall', linewidth=2)
            elif 'train/recall' in df.columns and 'val/recall' in df.columns:
                ax3.plot(epochs, df['train/recall'], 'b-', label='Training Recall', linewidth=2)
                ax3.plot(epochs, df['val/recall'], 'r-', label='Validation Recall', linewidth=2)
            
            ax3.set_title('Model Recall', fontweight='bold')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Recall')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Chart 4: mAP (Bottom Right)
            if 'metrics/mAP50(B)' in df.columns and 'val/mAP50(B)' in df.columns:
                ax4.plot(epochs, df['metrics/mAP50(B)'], 'b-', label='Training mAP@0.5', linewidth=2)
                ax4.plot(epochs, df['val/mAP50(B)'], 'r-', label='Validation mAP@0.5', linewidth=2)
            elif 'train/mAP50' in df.columns and 'val/mAP50' in df.columns:
                ax4.plot(epochs, df['train/mAP50'], 'b-', label='Training mAP@0.5', linewidth=2)
                ax4.plot(epochs, df['val/mAP50'], 'r-', label='Validation mAP@0.5', linewidth=2)
            
            ax4.set_title('Model mAP@0.5', fontweight='bold')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('mAP@0.5')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save charts
            charts_path = self.output_dir / 'training_charts.png'
            plt.savefig(charts_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"✅ Training charts saved: {charts_path}")
            
            # Also create individual charts
            self._create_individual_charts(df, epochs)
            
        except Exception as e:
            print(f"⚠️ Failed to create charts: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_individual_charts(self, df, epochs):
        """Create individual accuracy and loss charts like the reference image"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with 1x2 subplots (like reference image)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Chart 1: Model Accuracy
            if 'metrics/precision(B)' in df.columns and 'val/precision(B)' in df.columns:
                ax1.plot(epochs, df['metrics/precision(B)'], 'b-', label='Training Accuracy', linewidth=2)
                ax1.plot(epochs, df['val/precision(B)'], 'orange', label='Validation Accuracy', linewidth=2)
            
            ax1.set_title('Model Accuracy', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Epochs', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, len(epochs))
            
            # Chart 2: Model Loss
            if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                ax2.plot(epochs, df['train/box_loss'], 'b-', label='Training Loss', linewidth=2)
                ax2.plot(epochs, df['val/box_loss'], 'orange', label='Validation Loss', linewidth=2)
            
            ax2.set_title('Model Loss', fontweight='bold', fontsize=14)
            ax2.set_xlabel('Epochs', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, len(epochs))
            
            plt.tight_layout()
            
            # Save individual charts
            individual_charts_path = self.output_dir / 'accuracy_loss_charts.png'
            plt.savefig(individual_charts_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"✅ Individual charts saved: {individual_charts_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create individual charts: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv11s for human detection')
    parser.add_argument('--base-model', '-m', 
                       default=r'D:\SPHAR-Dataset\models\yolo11s.pt',
                       help='Path to base YOLOv11s model')
    parser.add_argument('--dataset', '-d',
                       default=r'D:\SPHAR-Dataset\train\human_focused_dataset',
                       help='Path to human detection dataset')
    parser.add_argument('--output', '-o',
                       default=r'D:\SPHAR-Dataset\models\finetuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--imgsz', '-i', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', '-b', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--export-name', default='yolo11s-detect.pt',
                       help='Name for exported model (default: yolo11s-detect.pt)')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing on sample images')
    
    args = parser.parse_args()
    
    try:
        # Create fine-tuner
        finetuner = YOLOv11HumanFineTuner(
            base_model_path=args.base_model,
            dataset_path=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch
        )
        
        # Run fine-tuning
        success = finetuner.run_complete_finetune(
            test_samples=not args.no_test,
            export_name=args.export_name
        )
        
        if success:
            print("\n🎉 Fine-tuning completed successfully!")
            print(f"🤖 Your fine-tuned model is ready: {args.output}/{args.export_name}")
        else:
            print("\n❌ Fine-tuning failed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
