#!/usr/bin/env python3
"""
Action Recognition Training Pipeline
Complete workflow from dataset organization to model training
"""

import subprocess
import sys
import json
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run command with progress indication"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def check_dataset_info(output_dir):
    """Check and display dataset information"""
    dataset_info_path = Path(output_dir) / 'dataset_info.json'
    
    if not dataset_info_path.exists():
        print("❌ Dataset info not found")
        return False
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    print("\n" + "="*80)
    print("📊 DATASET INFORMATION")
    print("="*80)
    
    total_videos = 0
    for action, info in dataset_info.items():
        total_videos += info['total']
        print(f"\n🎬 {action.upper()}:")
        print(f"   Description: {info['description']}")
        print(f"   Total videos: {info['total']}")
        print(f"   Train: {info['train']}, Val: {info['val']}, Test: {info['test']}")
    
    print(f"\n📈 TOTAL DATASET: {total_videos} videos")
    print(f"📁 Classes: {len(dataset_info)} actions")
    
    # Check balance
    min_videos = min(info['total'] for info in dataset_info.values())
    max_videos = max(info['total'] for info in dataset_info.values())
    balance_ratio = min_videos / max_videos
    
    print(f"\n⚖️ Dataset Balance: {balance_ratio:.2f}")
    if balance_ratio < 0.5:
        print("⚠️ Dataset is imbalanced - consider data augmentation")
    else:
        print("✅ Dataset is reasonably balanced")
    
    return True

def organize_dataset(videos_root, output_dir):
    """Organize dataset for training"""
    cmd = [
        sys.executable,
        "action_recognition_trainer.py",
        "--videos-root", str(videos_root),
        "--output-dir", str(output_dir),
        "--organize-only"
    ]
    
    success = run_command(cmd, "Dataset Organization")
    
    if success:
        check_dataset_info(output_dir)
    
    return success

def train_model(videos_root, output_dir, model_save_path, epochs, batch_size, sequence_length):
    """Train action recognition model"""
    cmd = [
        sys.executable,
        "action_recognition_trainer.py",
        "--videos-root", str(videos_root),
        "--output-dir", str(output_dir),
        "--model-save-path", str(model_save_path),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--sequence-length", str(sequence_length)
    ]
    
    return run_command(cmd, f"Model Training ({epochs} epochs)")

def test_model(model_path, class_mapping_path, test_source):
    """Test trained model"""
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    cmd = [
        sys.executable,
        "test_action_recognition.py",
        "--model", str(model_path),
        "--class-mapping", str(class_mapping_path),
        "--source", str(test_source)
    ]
    
    return run_command(cmd, f"Model Testing on {test_source}")

def test_integrated_system(yolo_model, action_model, class_mapping, test_source):
    """Test integrated detection + action recognition"""
    if not Path(action_model).exists():
        print(f"❌ Action model not found: {action_model}")
        return False
    
    cmd = [
        sys.executable,
        "integrated_detection_action.py",
        "--yolo-model", str(yolo_model),
        "--action-model", str(action_model),
        "--class-mapping", str(class_mapping),
        "--source", str(test_source)
    ]
    
    return run_command(cmd, f"Integrated System Testing on {test_source}")

def main():
    parser = argparse.ArgumentParser(description='Action Recognition Training Pipeline')
    parser.add_argument('--videos-root', default=r'D:\SPHAR-Dataset\videos',
                       help='Root directory of videos')
    parser.add_argument('--output-dir', default=r'D:\SPHAR-Dataset\action_recognition',
                       help='Output directory for organized dataset')
    parser.add_argument('--model-save-path', default=r'D:\SPHAR-Dataset\models\action_recognition_slowfast.pt',
                       help='Path to save trained model')
    parser.add_argument('--yolo-model', default=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                       help='Path to YOLO model for integrated testing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per video clip')
    
    # Pipeline control
    parser.add_argument('--organize-only', action='store_true',
                       help='Only organize dataset')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model (skip organization)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test model (skip training)')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline: organize + train + test')
    
    # Testing
    parser.add_argument('--test-source', default='webcam',
                       help='Test source: webcam, video file, or IITB video ID')
    parser.add_argument('--skip-integrated', action='store_true',
                       help='Skip integrated system testing')
    
    args = parser.parse_args()
    
    # Paths
    videos_root = Path(args.videos_root)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_save_path)
    class_mapping_path = output_dir / 'class_mapping.json'
    
    print("="*80)
    print("🎬 ACTION RECOGNITION TRAINING PIPELINE")
    print("="*80)
    print(f"📁 Videos root: {videos_root}")
    print(f"📁 Output dir: {output_dir}")
    print(f"🤖 Model path: {model_path}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎬 Sequence length: {args.sequence_length}")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Step 1: Organize Dataset
    if not args.train_only and not args.test_only:
        print(f"\n🎯 Step 1: Dataset Organization")
        success = organize_dataset(videos_root, output_dir)
        
        if not success:
            print("❌ Dataset organization failed. Exiting.")
            return
        
        if args.organize_only:
            print("\n✅ Dataset organization completed. Exiting as requested.")
            return
    
    # Step 2: Train Model
    if not args.test_only and success:
        print(f"\n🎯 Step 2: Model Training")
        success = train_model(
            videos_root, output_dir, model_path,
            args.epochs, args.batch_size, args.sequence_length
        )
        
        if not success:
            print("❌ Model training failed. Exiting.")
            return
        
        if args.train_only:
            print("\n✅ Model training completed. Exiting as requested.")
            return
    
    # Step 3: Test Action Recognition Model
    if success:
        print(f"\n🎯 Step 3: Action Recognition Testing")
        success = test_model(model_path, class_mapping_path, args.test_source)
        
        if not success:
            print("⚠️ Action recognition testing failed, but continuing...")
    
    # Step 4: Test Integrated System
    if success and not args.skip_integrated:
        print(f"\n🎯 Step 4: Integrated System Testing")
        success = test_integrated_system(
            args.yolo_model, model_path, class_mapping_path, args.test_source
        )
        
        if not success:
            print("⚠️ Integrated system testing failed")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 PIPELINE SUMMARY")
    print("="*80)
    
    if success:
        print("✅ Pipeline completed successfully!")
        print(f"\n📁 Outputs:")
        print(f"   📊 Dataset: {output_dir}")
        print(f"   🤖 Model: {model_path}")
        print(f"   🗂️ Class mapping: {class_mapping_path}")
        
        print(f"\n🚀 Next steps:")
        print(f"   # Test on webcam")
        print(f"   python test_action_recognition.py --source webcam")
        print(f"   ")
        print(f"   # Test integrated system")
        print(f"   python integrated_detection_action.py --source webcam")
        print(f"   ")
        print(f"   # Test on IITB video")
        print(f"   python integrated_detection_action.py --source 000209")
    else:
        print("❌ Pipeline completed with errors")
        print("Check the logs above for details")

if __name__ == "__main__":
    main()
