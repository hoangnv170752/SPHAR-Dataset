#!/usr/bin/env python3
"""
Script chạy huấn luyện SlowFast cho dataset 3-class
Tối ưu hóa cho GTX 1050 (2GB VRAM)

Usage:
python run_slowfast_training.py --dataset /path/to/abnormal_detection_3class_dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import json

def check_dataset(dataset_path):
    """Kiểm tra dataset có hợp lệ không"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return False, f"Dataset không tồn tại: {dataset_path}"
    
    # Kiểm tra cấu trúc dataset
    required_dirs = [
        'videos/train/normal',
        'videos/train/abnormal_physical', 
        'videos/train/abnormal_biological',
        'videos/val/normal',
        'videos/val/abnormal_physical',
        'videos/val/abnormal_biological'
    ]
    
    for dir_path in required_dirs:
        if not (dataset_path / dir_path).exists():
            return False, f"Thiếu thư mục: {dir_path}"
    
    # Kiểm tra dataset_info.json
    info_file = dataset_path / 'dataset_info.json'
    if not info_file.exists():
        return False, "Thiếu file dataset_info.json"
    
    try:
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        if 'classes' not in info:
            return False, "dataset_info.json không có thông tin classes"
        
        expected_classes = ['normal', 'abnormal_physical', 'abnormal_biological']
        actual_classes = list(info['classes'].keys())
        
        if set(expected_classes) != set(actual_classes):
            return False, f"Classes không đúng. Cần: {expected_classes}, có: {actual_classes}"
            
    except Exception as e:
        return False, f"Lỗi đọc dataset_info.json: {e}"
    
    return True, "Dataset hợp lệ"

def count_videos(dataset_path):
    """Đếm số lượng video trong dataset"""
    dataset_path = Path(dataset_path)
    
    counts = {
        'train': {'normal': 0, 'abnormal_physical': 0, 'abnormal_biological': 0},
        'val': {'normal': 0, 'abnormal_physical': 0, 'abnormal_biological': 0}
    }
    
    for split in ['train', 'val']:
        for class_name in ['normal', 'abnormal_physical', 'abnormal_biological']:
            class_dir = dataset_path / 'videos' / split / class_name
            if class_dir.exists():
                video_count = len(list(class_dir.glob('*.mp4'))) + len(list(class_dir.glob('*.avi')))
                counts[split][class_name] = video_count
    
    return counts

def run_slowfast_training(dataset_path, epochs=50, batch_size=2, device='cuda', output_dir='slowfast_results'):
    """Chạy huấn luyện SlowFast"""
    
    print(f"🚀 Bắt đầu huấn luyện SlowFast")
    print(f"📁 Dataset: {dataset_path}")
    print(f"🔢 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size} (tối ưu cho GTX 1050)")
    print(f"🖥️  Device: {device}")
    
    # Tạo output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tạo command
    cmd = [
        sys.executable,
        "scripts/train_slowfast_activity.py",
        "--dataset", str(dataset_path),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("⏳ Đang huấn luyện...")
    
    start_time = time.time()
    
    try:
        # Chạy training
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Huấn luyện hoàn thành!")
            print(f"⏱️  Tổng thời gian: {total_time:.2f} giây ({total_time/3600:.2f} giờ)")
            
            # Lưu log
            log_file = output_path / 'training_log.txt'
            with open(log_file, 'w') as f:
                f.write(f"Training completed successfully\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
            
            print(f"📋 Log lưu tại: {log_file}")
            
            # In output
            if result.stdout:
                print("\n📊 Training results:")
                print(result.stdout)
            
            return True
        else:
            print(f"❌ Lỗi khi huấn luyện!")
            print(f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Exception khi huấn luyện: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Huấn luyện SlowFast cho dataset 3-class')
    parser.add_argument('--dataset', '-d',
                       default='/home/nguyenhoang/Downloads/abnormal_detection_dataset',
                       help='Đường dẫn đến dataset 3-class')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Số epochs huấn luyện')
    parser.add_argument('--batch-size', '-b', type=int, default=2,
                       help='Batch size (2 cho GTX 1050)')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', '-o', default='slowfast_results',
                       help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    print("🎯 SlowFast Training for 3-Class Activity Detection")
    print("=" * 60)
    
    # Kiểm tra dataset
    print("🔍 Kiểm tra dataset...")
    is_valid, message = check_dataset(args.dataset)
    
    if not is_valid:
        print(f"❌ {message}")
        print("💡 Hãy tạo dataset 3-class trước bằng:")
        print("   python scripts/create_abnormal_detection_dataset.py --output /path/to/abnormal_detection_3class_dataset")
        return
    
    print(f"✅ {message}")
    
    # Đếm videos
    print("\n📊 Thống kê dataset:")
    video_counts = count_videos(args.dataset)
    
    for split in ['train', 'val']:
        print(f"  {split.upper()}:")
        for class_name, count in video_counts[split].items():
            print(f"    {class_name}: {count} videos")
    
    total_train = sum(video_counts['train'].values())
    total_val = sum(video_counts['val'].values())
    print(f"  Tổng train: {total_train}, val: {total_val}")
    
    # Kiểm tra script training
    if not Path("scripts/train_slowfast_activity.py").exists():
        print("❌ Không tìm thấy script train_slowfast_activity.py")
        return
    
    # Cảnh báo về GPU memory
    if args.batch_size > 2:
        print(f"⚠️  Cảnh báo: Batch size {args.batch_size} có thể quá lớn cho GTX 1050 (2GB VRAM)")
        print("   Khuyến nghị sử dụng batch size = 2")
    
    # Chạy training
    print(f"\n🎓 Bắt đầu huấn luyện...")
    success = run_slowfast_training(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output
    )
    
    if success:
        print("\n🎉 Huấn luyện hoàn thành!")
        print(f"💾 Model được lưu tại: best_slowfast_model.pth")
        print(f"📋 Logs tại: {args.output}/")
        print("\n🔮 Bước tiếp theo:")
        print("   - Kiểm tra accuracy trên test set")
        print("   - Tích hợp với YOLO + DeepSORT pipeline")
        print("   - Benchmark trên toàn bộ dataset")
    else:
        print("\n💥 Huấn luyện thất bại!")
        print("💡 Thử giảm batch size hoặc kiểm tra GPU memory")

if __name__ == "__main__":
    main()
