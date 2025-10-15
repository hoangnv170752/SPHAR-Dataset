#!/usr/bin/env python3
"""
Test class weights calculation for imbalanced dataset
"""

import json
from pathlib import Path

def calculate_class_weights():
    """Calculate and display class weights"""
    
    # Load dataset info
    dataset_info_path = Path(r'D:\SPHAR-Dataset\action_recognition\dataset_info.json')
    
    if not dataset_info_path.exists():
        print("❌ Dataset info not found. Run organization first:")
        print("python run_action_training.py --organize-only")
        return
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    print("="*80)
    print("⚖️ CLASS WEIGHTS CALCULATION")
    print("="*80)
    
    # Calculate class counts and weights
    class_counts = {action: info['total'] for action, info in dataset_info.items()}
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    print(f"📊 Dataset Summary:")
    print(f"   Total videos: {total_samples:,}")
    print(f"   Number of classes: {num_classes}")
    
    print(f"\n📈 Class Distribution:")
    
    # Sort classes by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    class_weights = {}
    for class_name, count in sorted_classes:
        percentage = (count / total_samples) * 100
        weight = total_samples / (num_classes * count)  # Inverse frequency weighting
        class_weights[class_name] = weight
        
        print(f"   {class_name:10s}: {count:5,} videos ({percentage:5.1f}%) → weight: {weight:.3f}")
    
    print(f"\n⚖️ Class Weights Effect:")
    print(f"   Without weights: All classes treated equally")
    print(f"   With weights: Rare classes get higher importance")
    
    # Show weight ratios
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    
    print(f"\n📊 Weight Analysis:")
    print(f"   Minimum weight: {min_weight:.3f} (most common class)")
    print(f"   Maximum weight: {max_weight:.3f} (rarest class)")
    print(f"   Weight ratio: {max_weight/min_weight:.1f}x")
    
    # Show expected training balance
    print(f"\n🎯 Training Impact:")
    for class_name, weight in class_weights.items():
        effective_samples = class_counts[class_name] * weight
        print(f"   {class_name:10s}: {effective_samples:7.0f} effective samples (was {class_counts[class_name]:,})")
    
    print(f"\n✅ Class weights will help balance training!")
    print(f"💡 Rare classes (fall, running) will get more attention")
    print(f"💡 Common classes (normal) will be down-weighted")

def main():
    print("🎬 Class Weights Calculator")
    calculate_class_weights()

if __name__ == "__main__":
    main()
