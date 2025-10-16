#!/usr/bin/env python3
"""
Complete Improved Training Pipeline
- Advanced data augmentation
- Better model architecture
- Advanced training strategies
- Automatic hyperparameter tuning
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

# Import improved components
from improved_model_architecture import ImprovedSlowFastModel
from improved_trainer import AdvancedTrainer
from improved_data_augmentation import AdvancedVideoAugmentation
from image_action_trainer import ImageSequenceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedImageSequenceDataset(ImageSequenceDataset):
    """Enhanced dataset with advanced augmentation"""
    
    def __init__(self, data_dir, split, sequence_length=8, transform=None, class_mapping=None, use_augmentation=True):
        super().__init__(data_dir, split, sequence_length, transform, class_mapping)
        
        # Advanced augmentation for training
        if split == 'train' and use_augmentation:
            self.augmentation = AdvancedVideoAugmentation(sequence_length)
        else:
            self.augmentation = None
    
    def __getitem__(self, idx):
        sequence_dir = self.sequences[idx]
        label = self.labels[idx]
        
        # Load image sequence
        frames = self._load_image_sequence(sequence_dir)
        
        # Apply advanced augmentation
        if self.augmentation:
            frames = self.augmentation(frames)
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label

def get_improved_config():
    """Get improved training configuration"""
    return {
        'learning_rate': 2e-4,  # Lower initial LR
        'weight_decay': 1e-3,   # Stronger regularization
        'use_focal_loss': True,
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'use_mixup': True,
        'use_cutmix': True,
        'gradient_clip_norm': 0.5,  # Gradient clipping
        'early_stopping_patience': 20,
        'early_stopping_min_delta': 0.001,
        'warmup_epochs': 5,
        'cosine_restarts': True,
        'accumulation_steps': 2,  # Gradient accumulation
    }

def run_improved_training(epochs=50, batch_size=12, sequence_length=8):
    """Run improved training pipeline"""
    
    print("="*80)
    print("🚀 IMPROVED ACTION RECOGNITION TRAINING")
    print("="*80)
    print("🎯 Improvements:")
    print("   ✅ Advanced data augmentation (spatial + temporal)")
    print("   ✅ Improved model architecture with attention")
    print("   ✅ Focal Loss + Label Smoothing")
    print("   ✅ Mixup + CutMix augmentation")
    print("   ✅ Gradient accumulation + clipping")
    print("   ✅ Early stopping + advanced scheduling")
    print("   ✅ Residual connections + batch normalization")
    
    # Paths
    image_dataset_dir = r'D:\SPHAR-Dataset\action_recognition_images'
    model_save_path = r'D:\SPHAR-Dataset\models\improved_action_model.pt'
    
    # Check if image dataset exists
    if not Path(image_dataset_dir).exists():
        print(f"\n❌ Image dataset not found: {image_dataset_dir}")
        print("Please run video to image conversion first:")
        print("python convert_videos_to_images.py")
        return False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load class mapping
    class_mapping_path = Path(image_dataset_dir).parent / 'action_recognition_optimized' / 'class_mapping.json'
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Load class counts
    dataset_info_path = Path(image_dataset_dir).parent / 'action_recognition_optimized' / 'dataset_info.json'
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    class_counts = {action: info['total'] for action, info in dataset_info.items()}
    
    print(f"\n📊 Dataset Info:")
    for action, count in class_counts.items():
        print(f"   {action}: {count} samples")
    
    # Create improved datasets
    train_dataset = ImprovedImageSequenceDataset(
        image_dataset_dir, 'train', sequence_length, 
        class_mapping=class_mapping, use_augmentation=True
    )
    val_dataset = ImprovedImageSequenceDataset(
        image_dataset_dir, 'val', sequence_length, 
        class_mapping=class_mapping, use_augmentation=False
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        drop_last=True  # For batch normalization stability
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Model: ImprovedSlowFastModel")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create improved model
    model = ImprovedSlowFastModel(
        num_classes=len(class_mapping), 
        sequence_length=sequence_length,
        dropout_rate=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create advanced trainer
    config = get_improved_config()
    trainer = AdvancedTrainer(
        model, train_loader, val_loader, device, 
        len(class_mapping), class_counts, config
    )
    
    # Train model
    logger.info("Starting improved training...")
    best_accuracy = trainer.train(epochs, model_save_path)
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    
    # Create advanced charts
    charts_dir = Path(image_dataset_dir).parent / 'charts_improved'
    charts_dir.mkdir(exist_ok=True)
    
    # Plot advanced metrics
    metrics_path = charts_dir / 'improved_training_metrics.png'
    trainer.plot_advanced_metrics(save_path=metrics_path)
    
    print(f"\n📊 Advanced charts saved to: {charts_dir}")
    
    return True

def estimate_improvements():
    """Show expected improvements"""
    print("="*80)
    print("📈 EXPECTED IMPROVEMENTS")
    print("="*80)
    
    print("🎯 Stability Improvements:")
    print("   ✅ Reduced overfitting (early stopping + regularization)")
    print("   ✅ Smoother convergence (advanced scheduling)")
    print("   ✅ Better generalization (data augmentation)")
    print("   ✅ Stable gradients (gradient clipping + accumulation)")
    
    print("\n📊 Performance Improvements:")
    print("   ✅ Higher accuracy (+5-10% expected)")
    print("   ✅ Better class balance (focal loss)")
    print("   ✅ Reduced validation gap")
    print("   ✅ More robust predictions")
    
    print("\n⚡ Training Improvements:")
    print("   ✅ Automatic early stopping")
    print("   ✅ Learning rate optimization")
    print("   ✅ Memory efficient (gradient accumulation)")
    print("   ✅ Advanced monitoring")

def main():
    parser = argparse.ArgumentParser(description='Improved Action Recognition Training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=12,
                       help='Batch size (default: 12)')
    parser.add_argument('--sequence-length', type=int, default=8,
                       help='Sequence length (default: 8)')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only show improvement estimates')
    
    args = parser.parse_args()
    
    print("🎬 IMPROVED ACTION RECOGNITION TRAINING")
    
    # Show improvement estimates
    estimate_improvements()
    
    if args.estimate_only:
        return
    
    print(f"\n🚀 Ready to start improved training:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.sequence_length}")
    
    response = input("\nProceed with improved training? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run improved training
    success = run_improved_training(args.epochs, args.batch_size, args.sequence_length)
    
    if success:
        print("\n" + "="*80)
        print("🎉 IMPROVED TRAINING COMPLETED!")
        print("="*80)
        print("📁 Outputs:")
        print("   Model: D:\\SPHAR-Dataset\\models\\improved_action_model.pt")
        print("   Charts: D:\\SPHAR-Dataset\\charts_improved\\")
        
        print("\n🚀 Next steps:")
        print("   # Test improved model")
        print("   python test_action_recognition.py --model improved_action_model.pt")
        print("   ")
        print("   # Compare with previous models")
        print("   python compare_models.py")
        
        print("\n📈 Expected Results:")
        print("   ✅ Much more stable training curves")
        print("   ✅ Higher validation accuracy")
        print("   ✅ Reduced overfitting")
        print("   ✅ Better generalization")
    else:
        print("\n❌ Improved training setup failed. Check requirements.")

if __name__ == "__main__":
    main()
