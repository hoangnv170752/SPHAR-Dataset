#!/usr/bin/env python3
"""
Optimized training script for SlowFast Tiny with reduced dataset
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_optimized_training(epochs=15, batch_size=8, sequence_length=8):
    """Run optimized training with reduced dataset"""
    
    print("="*80)
    print("🚀 OPTIMIZED ACTION RECOGNITION TRAINING")
    print("="*80)
    print(f"🎯 Configuration:")
    print(f"   Model: SlowFast Tiny")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Dataset: Reduced size for faster training")
    print(f"   Class weights: Enabled for balance")
    
    # Step 1: Organize reduced dataset
    print(f"\n🎯 Step 1: Organizing Reduced Dataset")
    print("="*80)
    
    organize_cmd = [
        sys.executable,
        "action_recognition_trainer.py",
        "--videos-root", r"D:\SPHAR-Dataset\videos",
        "--output-dir", r"D:\SPHAR-Dataset\action_recognition_optimized",
        "--organize-only"
    ]
    
    try:
        result = subprocess.run(organize_cmd, check=True, capture_output=False)
        print("✅ Dataset organization completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset organization failed: {e}")
        return False
    
    # Step 2: Train model
    print(f"\n🎯 Step 2: Training SlowFast Tiny Model")
    print("="*80)
    
    train_cmd = [
        sys.executable,
        "action_recognition_trainer.py",
        "--videos-root", r"D:\SPHAR-Dataset\videos",
        "--output-dir", r"D:\SPHAR-Dataset\action_recognition_optimized",
        "--model-save-path", r"D:\SPHAR-Dataset\models\action_recognition_slowfast_tiny.pt",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--sequence-length", str(sequence_length)
    ]
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False)
        print("✅ Training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def estimate_training_time(epochs, batch_size, sequence_length):
    """Estimate training time for optimized setup"""
    
    # Reduced dataset estimates
    estimated_videos = {
        'fall': 175,      # Keep all
        'hitting': 400,   # Reduced from 782
        'running': 300,   # Reduced from 568
        'warning': 400,   # Reduced from 744
        'normal': 800     # Reduced from 5,705
    }
    
    total_videos = sum(estimated_videos.values())
    train_videos = int(total_videos * 0.7)
    
    batches_per_epoch = train_videos // batch_size
    
    # Time estimates (SlowFast Tiny is ~50% faster than full)
    time_per_batch = {
        8: 12,   # 12 seconds per batch with sequence_length 8
        12: 18,  # 18 seconds per batch with sequence_length 12
        16: 25   # 25 seconds per batch with sequence_length 16
    }
    
    batch_time = time_per_batch.get(sequence_length, 15)
    
    time_per_epoch = (batches_per_epoch * batch_time) / 3600  # hours
    total_time = time_per_epoch * epochs
    
    print("="*80)
    print("⏱️ TRAINING TIME ESTIMATION")
    print("="*80)
    print(f"📊 Reduced Dataset:")
    for action, count in estimated_videos.items():
        print(f"   {action:10s}: {count:3d} videos")
    
    print(f"\n📈 Training Stats:")
    print(f"   Total videos: {total_videos:,}")
    print(f"   Training videos: {train_videos:,}")
    print(f"   Batches per epoch: {batches_per_epoch}")
    print(f"   Time per batch: {batch_time}s")
    print(f"   Time per epoch: {time_per_epoch:.1f} hours")
    
    print(f"\n⏰ Time Estimates:")
    print(f"   {epochs} epochs: {total_time:.1f} hours ({total_time/24:.1f} days)")
    
    if total_time < 24:
        print(f"   ✅ Training will complete in less than 1 day!")
    elif total_time < 48:
        print(f"   ⚠️ Training will take 1-2 days")
    else:
        print(f"   ⚠️ Training will take more than 2 days")

def main():
    parser = argparse.ArgumentParser(description='Optimized Action Recognition Training')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--sequence-length', type=int, default=8,
                       help='Number of frames per sequence (default: 8)')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only show time estimation')
    
    args = parser.parse_args()
    
    print("🎬 OPTIMIZED ACTION RECOGNITION TRAINING")
    
    # Show time estimation
    estimate_training_time(args.epochs, args.batch_size, args.sequence_length)
    
    if args.estimate_only:
        return
    
    # Ask for confirmation
    print(f"\n🚀 Ready to start training with:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sequence length: {args.sequence_length}")
    
    response = input("\nProceed with training? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    success = run_optimized_training(args.epochs, args.batch_size, args.sequence_length)
    
    if success:
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Model saved: D:\\SPHAR-Dataset\\models\\action_recognition_slowfast_tiny.pt")
        print(f"📊 Dataset: D:\\SPHAR-Dataset\\action_recognition_optimized")
        
        print(f"\n🚀 Next steps:")
        print(f"   # Test the model")
        print(f"   python test_action_recognition.py --model action_recognition_slowfast_tiny.pt")
        print(f"   ")
        print(f"   # Test integrated system")
        print(f"   python integrated_detection_action.py --action-model action_recognition_slowfast_tiny.pt")
    else:
        print("\n❌ Training failed. Check logs above for details.")

if __name__ == "__main__":
    main()
