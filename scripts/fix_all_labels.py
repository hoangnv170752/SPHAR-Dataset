#!/usr/bin/env python3
"""
Aggressively fix ALL corrupted label files
"""

from pathlib import Path
from tqdm import tqdm
import re

def fix_label_aggressive(label_path):
    """Aggressively fix a label file"""
    try:
        # Read raw content
        with open(label_path, 'rb') as f:
            raw_content = f.read()
        
        # Decode to string
        content = raw_content.decode('utf-8', errors='ignore')
        
        # Fix all variations of escaped newlines
        content = content.replace('\\n', '\n')
        content = content.replace('\\r', '\r')
        
        # Split into lines
        lines = content.split('\n')
        
        # Process each line
        valid_lines = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lone '0'
            if line == '0':
                continue
            
            # Remove trailing '0' that's not part of coordinates
            if line.endswith(' 0') and len(line.split()) == 6:
                line = ' '.join(line.split()[:5])
            
            # Check if valid YOLO format
            parts = line.split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Validate coordinates (0-1 range)
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        valid_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                except (ValueError, IndexError):
                    pass
        
        # Write fixed content
        with open(label_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')
        
        return True, len(valid_lines)
        
    except Exception as e:
        print(f"Error fixing {label_path}: {e}")
        return False, 0

def fix_all_labels_in_dataset(dataset_path):
    """Fix ALL label files in dataset"""
    dataset_path = Path(dataset_path)
    
    print(f"🔧 Fixing ALL labels in: {dataset_path}")
    
    # Get all label directories
    label_dirs = [
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val',
        dataset_path / 'labels' / 'test'
    ]
    
    total_files = 0
    fixed_count = 0
    total_instances = 0
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            print(f"⚠️ Directory not found: {label_dir}")
            continue
        
        print(f"\n📁 Processing: {label_dir}")
        label_files = list(label_dir.glob('*.txt'))
        
        if not label_files:
            print("  No label files found")
            continue
        
        print(f"  Found {len(label_files)} label files")
        
        for label_file in tqdm(label_files, desc=f"  Fixing {label_dir.name}"):
            total_files += 1
            success, num_instances = fix_label_aggressive(label_file)
            
            if success:
                fixed_count += 1
                total_instances += num_instances
    
    print(f"\n✅ Label fixing completed:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Valid instances recovered: {total_instances}")
    
    return total_files, fixed_count, total_instances

def validate_fixed_labels(dataset_path):
    """Validate that all labels are now correct"""
    dataset_path = Path(dataset_path)
    
    print(f"\n🔍 Validating fixed labels...")
    
    label_dirs = [
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val',
        dataset_path / 'labels' / 'test'
    ]
    
    total_files = 0
    valid_files = 0
    total_instances = 0
    corrupted_files = []
    
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        
        for label_file in tqdm(label_files, desc=f"  Validating {label_dir.name}"):
            total_files += 1
            
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for corruption patterns
                if '\\n' in content or '\\r' in content:
                    corrupted_files.append(str(label_file))
                    continue
                
                # Validate format
                lines = content.strip().split('\n') if content.strip() else []
                file_valid = True
                file_instances = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        file_valid = False
                        break
                    
                    try:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            file_valid = False
                            break
                        
                        file_instances += 1
                    except ValueError:
                        file_valid = False
                        break
                
                if file_valid:
                    valid_files += 1
                    total_instances += file_instances
                else:
                    corrupted_files.append(str(label_file))
                    
            except Exception as e:
                corrupted_files.append(str(label_file))
    
    print(f"\n📊 Validation Results:")
    print(f"  Total files: {total_files}")
    print(f"  Valid files: {valid_files}")
    print(f"  Total instances: {total_instances}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n❌ Still corrupted:")
        for f in corrupted_files[:5]:
            print(f"    {Path(f).name}")
        if len(corrupted_files) > 5:
            print(f"    ... and {len(corrupted_files) - 5} more")
        return False
    else:
        print(f"\n✅ ALL labels are now valid!")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggressively fix ALL corrupted labels')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 AGGRESSIVE LABEL FIXER")
    print("="*70)
    
    # Fix all labels
    total, fixed, instances = fix_all_labels_in_dataset(args.dataset_path)
    
    # Validate
    is_valid = validate_fixed_labels(args.dataset_path)
    
    print("\n" + "="*70)
    if is_valid:
        print("🎉 SUCCESS! Dataset is ready for training!")
        print(f"   Total instances: {instances}")
    else:
        print("⚠️ Some issues remain. Try running again or recreate dataset.")
    print("="*70)

if __name__ == "__main__":
    main()
