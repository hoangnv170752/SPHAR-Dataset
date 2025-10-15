#!/usr/bin/env python3
"""
Quick benchmark runner with preset configurations
Supports single comparison and multi-model benchmark
"""

import subprocess
import sys
from pathlib import Path

BENCHMARK_CONFIGS = {
    "quick": {
        "video": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01_walk_a1.mp4",
        "max_frames": 100,
        "desc": "Quick test (100 frames, 1 person)"
    },
    "2people": {
        "video": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01p02_meettogether_a1.mp4",
        "max_frames": None,
        "desc": "Full video (2 people meeting)"
    },
    "ntu": {
        "video": r"D:\SPHAR-Dataset\videos\NTU\A001\S001C001P001R001A001_rgb.avi",
        "max_frames": None,
        "desc": "NTU action dataset"
    },
    "full": {
        "video": r"D:\SPHAR-Dataset\videos\walking\casia_angleview_p01p02_meettogether_a1.mp4",
        "max_frames": None,
        "desc": "Full comprehensive test"
    }
}

def run_benchmark(config_name, conf=0.25):
    """Run benchmark with specific configuration"""
    if config_name not in BENCHMARK_CONFIGS:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available: {list(BENCHMARK_CONFIGS.keys())}")
        return
    
    config = BENCHMARK_CONFIGS[config_name]
    video_path = Path(config["video"])
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("="*80)
    print(f"🏁 BENCHMARK: {config['desc']}")
    print("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "benchmark_models.py",
        "--model1", r"D:\SPHAR-Dataset\models\yolo11s.pt",
        "--model2", r"D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt",
        "--video", str(video_path),
        "--conf", str(conf),
        "--output", f"D:\\SPHAR-Dataset\\benchmark_{config_name}.json"
    ]
    
    if config["max_frames"]:
        cmd.extend(["--max-frames", str(config["max_frames"])])
    
    print(f"\n🚀 Running benchmark...\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")

def run_multi_model_benchmark(conf=0.25, max_frames=None):
    """Run multi-model benchmark on multiple videos"""
    print("="*80)
    print("🏁 MULTI-MODEL BENCHMARK")
    print("   Comparing: YOLOv8s, YOLOv11s, YOLOv11s-FT")
    print("   Videos: 5 test videos")
    print("="*80)
    
    cmd = [
        sys.executable,
        "benchmark_multi_models.py",
        "--conf", str(conf),
        "--output", r"D:\SPHAR-Dataset\multi_model_benchmark.json"
    ]
    
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    
    print(f"\n🚀 Running multi-model benchmark...\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick benchmark runner')
    parser.add_argument('config', nargs='?', default='quick',
                       help='Benchmark configuration (or "multi" for multi-model)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--list', action='store_true',
                       help='List available configurations')
    parser.add_argument('--multi', action='store_true',
                       help='Run multi-model benchmark')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Max frames for multi-model benchmark')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n🎯 Available benchmark configurations:")
        print("="*80)
        print("  SINGLE MODEL COMPARISON:")
        for name, config in BENCHMARK_CONFIGS.items():
            video_name = Path(config['video']).name
            frames = f"{config['max_frames']} frames" if config['max_frames'] else "full video"
            print(f"    {name:15s} - {config['desc']}")
            print(f"    {'':15s}   Video: {video_name} ({frames})")
        print("\n  MULTI-MODEL BENCHMARK:")
        print("    multi          - Compare 3 models on 5 videos")
        print("                     (YOLOv8s, YOLOv11s, YOLOv11s-FT)")
        print("="*80)
        return
    
    # Multi-model benchmark
    if args.multi or args.config == 'multi':
        run_multi_model_benchmark(args.conf, args.max_frames)
    # Single model comparison
    elif args.config in BENCHMARK_CONFIGS:
        run_benchmark(args.config, args.conf)
    else:
        print(f"❌ Unknown config: {args.config}")
        print("Use --list to see available configurations")

if __name__ == "__main__":
    main()
