#!/usr/bin/env python3
"""
Clear YOLO cache files
"""

from pathlib import Path
import sys

def clear_cache(dataset_path):
    """Clear all YOLO cache files"""
    dataset_path = Path(dataset_path)
    
    print(f"🧹 Clearing YOLO cache files in: {dataset_path}")
    
    # Find all .cache files
    cache_files = list(dataset_path.rglob('*.cache'))
    
    if not cache_files:
        print("✅ No cache files found")
        return
    
    print(f"Found {len(cache_files)} cache files:")
    for cache_file in cache_files:
        print(f"  📁 {cache_file.relative_to(dataset_path)}")
        try:
            cache_file.unlink()
            print(f"     ✅ Deleted")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print(f"\n✅ Cache cleared! YOLO will rebuild cache on next training.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clear_yolo_cache.py <dataset_path>")
        sys.exit(1)
    
    clear_cache(sys.argv[1])
