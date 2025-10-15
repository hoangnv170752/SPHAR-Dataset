#!/usr/bin/env python3
"""
Helper script to run YOLOv11s fine-tuning with different configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_finetune_config(config_name, **kwargs):
    """Run fine-tuning with specific configuration"""
    
    configs = {
        'quick': {
            'epochs': 50,
            'batch': 8,
            'imgsz': 416,
            'description': 'Quick test run (50 epochs, small images)'
        },
        'standard': {
            'epochs': 200,
            'batch': 16,
            'imgsz': 640,
            'description': 'Standard training (200 epochs, balanced)'
        },
        'high_quality': {
            'epochs': 300,
            'batch': 8,
            'imgsz': 832,
            'description': 'High quality training (300 epochs, large images)'
        },
        'production': {
            'epochs': 500,
            'batch': 32,
            'imgsz': 640,
            'description': 'Production ready (500 epochs, large batch)'
        }
    }
    
    if config_name not in configs:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list(configs.keys())}")
        return False
    
    config = configs[config_name]
    print(f"🚀 Running {config_name} configuration:")
    print(f"   {config['description']}")
    
    # Override with any provided kwargs
    config.update(kwargs)
    
    # Build command
    script_path = Path(__file__).parent / 'finetune_yolo11s_human.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--epochs', str(config['epochs']),
        '--batch', str(config['batch']),
        '--imgsz', str(config['imgsz'])
    ]
    
    # Add other arguments
    for key, value in kwargs.items():
        if key not in ['epochs', 'batch', 'imgsz']:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {config_name} configuration completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {config_name} configuration failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv11s fine-tuning with preset configurations')
    parser.add_argument('config', choices=['quick', 'standard', 'high_quality', 'production'],
                       help='Training configuration to use')
    parser.add_argument('--base-model', default=r'D:\SPHAR-Dataset\models\yolo11s.pt',
                       help='Path to base model')
    parser.add_argument('--dataset', default=r'D:\SPHAR-Dataset\train\human_focused_dataset',
                       help='Path to dataset')
    parser.add_argument('--output', default=r'D:\SPHAR-Dataset\models\finetuned',
                       help='Output directory')
    parser.add_argument('--export-name', default='yolo11s-detect.pt',
                       help='Export model name')
    
    args = parser.parse_args()
    
    # Convert args to kwargs
    kwargs = {
        'base_model': args.base_model,
        'dataset': args.dataset,
        'output': args.output,
        'export_name': args.export_name
    }
    
    # Run configuration
    success = run_finetune_config(args.config, **kwargs)
    
    if success:
        print(f"\n🎉 Fine-tuning with {args.config} config completed!")
        print(f"📁 Check output: {args.output}")
        print(f"🤖 Model: {args.output}/{args.export_name}")
    else:
        print(f"\n❌ Fine-tuning with {args.config} config failed!")

if __name__ == "__main__":
    main()
