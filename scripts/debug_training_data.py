#!/usr/bin/env python3
"""
Debug training data to find why no instances are loaded
"""

import yaml
from pathlib import Path
import cv2
import random

def check_dataset_yaml(dataset_path):
    """Check dataset.yaml configuration"""
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / 'dataset.yaml'
    
    print("🔍 Checking dataset.yaml...")
    
    if not yaml_path.exists():
        print(f"❌ dataset.yaml not found: {yaml_path}")
        return False
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ dataset.yaml found")
    print(f"  Path: {config.get('path', 'Not set')}")
    print(f"  Train: {config.get('train', 'Not set')}")
    print(f"  Val: {config.get('val', 'Not set')}")
    print(f"  Classes: {config.get('nc', 'Not set')}")
    print(f"  Names: {config.get('names', 'Not set')}")
    
    # Check if paths exist
    base_path = Path(config.get('path', dataset_path))
    train_path = base_path / config.get('train', 'images/train')
    val_path = base_path / config.get('val', 'images/val')
    
    print(f"\n📁 Checking directories:")
    print(f"  Base path exists: {base_path.exists()}")
    print(f"  Train path exists: {train_path.exists()}")
    print(f"  Val path exists: {val_path.exists()}")
    
    if train_path.exists():
        train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        print(f"  Train images: {len(train_images)}")
    
    if val_path.exists():
        val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
        print(f"  Val images: {len(val_images)}")
    
    return True

def check_label_files(dataset_path, num_samples=10):
    """Check label files for issues"""
    dataset_path = Path(dataset_path)
    
    print(f"\n🏷️ Checking label files (sampling {num_samples})...")
    
    # Check train labels
    train_labels_dir = dataset_path / 'labels' / 'train'
    if not train_labels_dir.exists():
        print(f"❌ Train labels directory not found: {train_labels_dir}")
        return False
    
    label_files = list(train_labels_dir.glob('*.txt'))
    print(f"  Total train label files: {len(label_files)}")
    
    if not label_files:
        print("❌ No label files found!")
        return False
    
    # Sample some files
    sample_files = random.sample(label_files, min(num_samples, len(label_files)))
    
    valid_files = 0
    empty_files = 0
    corrupted_files = 0
    total_instances = 0
    
    for label_file in sample_files:
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
            
            if not content:
                empty_files += 1
                print(f"  📄 Empty: {label_file.name}")
            else:
                lines = content.split('\n')
                valid_lines = 0
                
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            try:
                                cls = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                # Check if coordinates are valid (0-1 range)
                                if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                    valid_lines += 1
                                    total_instances += 1
                                else:
                                    print(f"  ⚠️ Invalid coordinates in {label_file.name}: {line}")
                            except ValueError:
                                print(f"  ❌ Invalid format in {label_file.name}: {line}")
                        else:
                            print(f"  ❌ Wrong format in {label_file.name}: {line}")
                
                if valid_lines > 0:
                    valid_files += 1
                    print(f"  ✅ Valid: {label_file.name} ({valid_lines} instances)")
                else:
                    corrupted_files += 1
                    print(f"  ❌ Corrupted: {label_file.name}")
                    
        except Exception as e:
            corrupted_files += 1
            print(f"  ❌ Error reading {label_file.name}: {e}")
    
    print(f"\n📊 Label files summary:")
    print(f"  Valid files: {valid_files}")
    print(f"  Empty files: {empty_files}")
    print(f"  Corrupted files: {corrupted_files}")
    print(f"  Total instances found: {total_instances}")
    
    return valid_files > 0

def check_image_label_pairs(dataset_path, num_samples=5):
    """Check if images have corresponding labels"""
    dataset_path = Path(dataset_path)
    
    print(f"\n🖼️ Checking image-label pairs (sampling {num_samples})...")
    
    train_images_dir = dataset_path / 'images' / 'train'
    train_labels_dir = dataset_path / 'labels' / 'train'
    
    if not train_images_dir.exists():
        print(f"❌ Train images directory not found: {train_images_dir}")
        return False
    
    image_files = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
    print(f"  Total train images: {len(image_files)}")
    
    if not image_files:
        print("❌ No image files found!")
        return False
    
    # Sample some files
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in sample_images:
        label_file = train_labels_dir / (img_file.stem + '.txt')
        
        print(f"  📸 Image: {img_file.name}")
        print(f"    Label exists: {label_file.exists()}")
        
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                
                if content:
                    lines = content.split('\n')
                    valid_lines = [line for line in lines if line.strip()]
                    print(f"    Label lines: {len(valid_lines)}")
                    
                    # Show first line as example
                    if valid_lines:
                        print(f"    Example: {valid_lines[0]}")
                else:
                    print(f"    Label: Empty (no objects)")
                    
            except Exception as e:
                print(f"    Label error: {e}")
        
        # Check image
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                print(f"    Image size: {w}x{h}")
            else:
                print(f"    ❌ Cannot read image")
        except Exception as e:
            print(f"    Image error: {e}")
        
        print()
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug training data issues')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--samples', type=int, default=10, help='Number of files to sample')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🐛 TRAINING DATA DEBUG")
    print("="*60)
    
    # Check dataset.yaml
    yaml_ok = check_dataset_yaml(args.dataset_path)
    
    # Check label files
    labels_ok = check_label_files(args.dataset_path, args.samples)
    
    # Check image-label pairs
    pairs_ok = check_image_label_pairs(args.dataset_path, 5)
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"Dataset YAML: {'✅' if yaml_ok else '❌'}")
    print(f"Label files: {'✅' if labels_ok else '❌'}")
    print(f"Image-label pairs: {'✅' if pairs_ok else '❌'}")
    
    if yaml_ok and labels_ok and pairs_ok:
        print("\n🎉 Dataset looks good! Training should work.")
    else:
        print("\n⚠️ Issues found. Fix them before training.")

if __name__ == "__main__":
    main()
