#!/usr/bin/env python3
"""
Inspect raw content of label files
"""

from pathlib import Path
import sys

def inspect_label(label_path):
    """Show raw content of a label file"""
    print(f"\n📄 File: {label_path.name}")
    print(f"   Path: {label_path}")
    
    # Read as binary to see raw bytes
    with open(label_path, 'rb') as f:
        raw_bytes = f.read()
    
    print(f"   Size: {len(raw_bytes)} bytes")
    print(f"   Raw bytes: {raw_bytes[:100]}")
    
    # Read as text
    with open(label_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"   Text repr: {repr(content[:100])}")
    print(f"   Actual content:")
    print("   " + "-"*50)
    for i, line in enumerate(content.split('\n')[:5], 1):
        print(f"   Line {i}: {repr(line)}")
    print("   " + "-"*50)
    
    # Check for issues
    issues = []
    if '\\n' in content:
        issues.append("Contains escaped newline \\\\n")
    if '\\r' in content:
        issues.append("Contains escaped carriage return \\\\r")
    
    lines = content.strip().split('\n') if content.strip() else []
    for line in lines:
        parts = line.strip().split()
        if line.strip() and len(parts) != 5:
            issues.append(f"Invalid format: {repr(line)}")
    
    if issues:
        print(f"   ⚠️ Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ No issues found")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_labels.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    val_labels = dataset_path / 'labels' / 'val'
    
    if not val_labels.exists():
        print(f"Directory not found: {val_labels}")
        sys.exit(1)
    
    # Get some problematic files from the error message
    problem_files = [
        'carcrash_livevid_crash3_frame_045740.txt',
        'vandalizing_casia_horizontalview_p01_car_a1_frame_000060.txt',
        'walking_casia_angleview_p01_walk_a2_frame_000060.txt'
    ]
    
    print("="*60)
    print("🔍 INSPECTING LABEL FILES")
    print("="*60)
    
    for filename in problem_files:
        label_path = val_labels / filename
        if label_path.exists():
            inspect_label(label_path)
        else:
            print(f"\n❌ Not found: {filename}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
