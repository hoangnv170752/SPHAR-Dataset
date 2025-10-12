#!/usr/bin/env python3
"""
Script chạy benchmark detection cho dataset 3-class
Sử dụng để đo thời gian xử lý cho toàn bộ dataset train

Usage:
python run_dataset_benchmark.py --dataset /home/nguyenhoang/Downloads/abnormal_detection_3class_dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

def run_benchmark(dataset_path, output_dir="benchmark_results", splits=["train"], models=["yolov8n.pt"], sample_size=None):
    """Chạy benchmark detection cho dataset với nhiều models"""
    
    # Kiểm tra dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Lỗi: Không tìm thấy dataset tại {dataset_path}")
        return False
    
    print(f"🚀 Bắt đầu benchmark detection cho dataset: {dataset_path}")
    print(f"📊 Splits sẽ xử lý: {splits}")
    print(f"🤖 Models sẽ test: {models}")
    if sample_size:
        print(f"🎲 Sample size: {sample_size} videos ngẫu nhiên")
    print(f"💾 Kết quả sẽ lưu tại: {output_dir}")
    
    all_results = {}
    
    # Chạy benchmark cho từng model
    for model in models:
        print(f"\n{'='*60}")
        print(f"🔥 Benchmark với model: {model}")
        print(f"{'='*60}")
        
        model_output_dir = f"{output_dir}/{model.replace('.pt', '')}"
        
        # Tạo command để chạy benchmark
        cmd = [
            sys.executable,  # python
            "scripts/batch_detection_benchmark.py",
            "--dataset", str(dataset_path),
            "--output", model_output_dir,
            "--splits"] + splits + [
            "--model", model,
            "--conf", "0.5",
            "--device", "cuda"
        ]
        
        # Thêm sample-size nếu có
        if sample_size:
            cmd.extend(["--sample-size", str(sample_size)])
    
        print(f"🔧 Command: {' '.join(cmd)}")
        print("⏳ Đang chạy benchmark...")
        
        model_start_time = time.time()
        
        try:
            # Chạy benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            model_end_time = time.time()
            model_time = model_end_time - model_start_time
            
            if result.returncode == 0:
                print(f"✅ Benchmark {model} hoàn thành!")
                print(f"⏱️  Thời gian {model}: {model_time:.2f} giây ({model_time/3600:.2f} giờ)")
                print(f"📁 Kết quả lưu tại: {model_output_dir}")
                
                all_results[model] = {
                    'success': True,
                    'time': model_time,
                    'output_dir': model_output_dir
                }
                
                # In output summary
                if result.stdout:
                    lines = result.stdout.split('\n')
                    summary_lines = [line for line in lines if 'hoàn thành' in line or 'videos/giờ' in line or 'Tổng' in line]
                    if summary_lines:
                        print("📊 Summary:")
                        for line in summary_lines[-3:]:  # Last 3 summary lines
                            print(f"   {line}")
                
            else:
                print(f"❌ Lỗi khi chạy benchmark {model}!")
                print(f"Exit code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                
                all_results[model] = {
                    'success': False,
                    'time': model_time,
                    'error': result.stderr
                }
                
        except Exception as e:
            print(f"❌ Exception khi chạy benchmark {model}: {e}")
            all_results[model] = {
                'success': False,
                'time': 0,
                'error': str(e)
            }
    
    # Tổng kết tất cả models
    print(f"\n{'='*80}")
    print(f"📊 TỔNG KẾT BENCHMARK TẤT CẢ MODELS")
    print(f"{'='*80}")
    
    successful_models = 0
    total_time = 0
    
    for model, result in all_results.items():
        status = "✅ Thành công" if result['success'] else "❌ Thất bại"
        time_str = f"{result['time']:.2f}s" if result['time'] > 0 else "N/A"
        print(f"{model:<15} | {status:<12} | Thời gian: {time_str}")
        
        if result['success']:
            successful_models += 1
            total_time += result['time']
    
    print(f"\n📈 Kết quả:")
    print(f"   Thành công: {successful_models}/{len(models)} models")
    print(f"   Tổng thời gian: {total_time:.2f}s ({total_time/3600:.2f}h)")
    if successful_models > 0:
        print(f"   Thời gian TB/model: {total_time/successful_models:.2f}s")
    
    return successful_models > 0

def main():
    parser = argparse.ArgumentParser(description='Chạy benchmark detection cho dataset 3-class với nhiều YOLO models')
    parser.add_argument('--dataset', '-d', 
                       default='/home/nguyenhoang/Downloads/abnormal_detection_dataset',
                       help='Đường dẫn đến dataset 3-class')
    parser.add_argument('--output', '-o', 
                       default='benchmark_results',
                       help='Thư mục lưu kết quả benchmark')
    parser.add_argument('--splits', nargs='+', 
                       default=['train'],
                       help='Danh sách splits cần benchmark (train, val, test)')
    parser.add_argument('--models', nargs='+',
                       default=['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt'],
                       help='Danh sách YOLO models cần benchmark')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Số lượng videos random để benchmark (default: tất cả)')
    
    args = parser.parse_args()
    
    print("🎯 Dataset Detection Benchmark")
    print("=" * 50)
    
    # Kiểm tra dataset có tồn tại không
    if not Path(args.dataset).exists():
        print(f"❌ Dataset không tồn tại: {args.dataset}")
        print("💡 Hãy tạo dataset 3-class trước bằng:")
        print("   python scripts/create_abnormal_detection_dataset.py --output /path/to/abnormal_detection_3class_dataset")
        return
    
    # Kiểm tra có script benchmark không
    if not Path("scripts/batch_detection_benchmark.py").exists():
        print("❌ Không tìm thấy script batch_detection_benchmark.py")
        return
    
    # Chạy benchmark
    success = run_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output,
        splits=args.splits,
        models=args.models,
        sample_size=args.sample_size
    )
    
    if success:
        print("\n🎉 Benchmark hoàn thành!")
        print(f"📊 Kiểm tra kết quả tại: {args.output}/")
        print("   - benchmark_results.json: Kết quả chi tiết")
        print("   - video_benchmark.csv: Thống kê từng video")
        print("   - category_benchmark.csv: Thống kê theo category")
    else:
        print("\n💥 Benchmark thất bại!")

if __name__ == "__main__":
    main()
