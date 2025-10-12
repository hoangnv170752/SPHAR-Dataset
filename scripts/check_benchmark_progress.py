#!/usr/bin/env python3
"""
Script kiểm tra progress của benchmark đang chạy
Đọc log files để hiển thị tiến độ hiện tại

Usage:
python check_benchmark_progress.py --benchmark-dir benchmark_results
"""

import argparse
import json
from pathlib import Path
import time

def check_progress(benchmark_dir):
    """Kiểm tra tiến độ benchmark"""
    benchmark_dir = Path(benchmark_dir)
    
    print(f"🔍 Kiểm tra tiến độ benchmark tại: {benchmark_dir}")
    print(f"⏰ Thời gian kiểm tra: {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Tìm các thư mục model
    model_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("❌ Không tìm thấy thư mục benchmark nào")
        return
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n🤖 Model: {model_name}")
        print("-" * 40)
        
        # Kiểm tra file kết quả
        json_file = model_dir / 'benchmark_results.json'
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Thông tin dataset
                dataset_info = data.get('dataset_info', {})
                total_videos = dataset_info.get('total_videos', 0)
                
                # Thông tin progress
                video_results = data.get('video_results', {})
                processed_videos = len(video_results)
                
                # Thống kê thời gian
                overall_stats = data.get('overall_stats', {})
                total_time = overall_stats.get('total_processing_time', 0)
                avg_time = overall_stats.get('avg_time_per_video', 0)
                
                if total_videos > 0:
                    progress_percent = (processed_videos / total_videos) * 100
                    remaining_videos = total_videos - processed_videos
                    estimated_remaining_time = remaining_videos * avg_time if avg_time > 0 else 0
                    
                    print(f"📊 Tiến độ: {processed_videos}/{total_videos} videos ({progress_percent:.1f}%)")
                    print(f"⏱️  Đã xử lý: {total_time/3600:.2f}h")
                    print(f"🕐 Còn lại: ~{estimated_remaining_time/3600:.2f}h")
                    print(f"⚡ Tốc độ TB: {avg_time:.2f}s/video")
                    
                    if progress_percent == 100:
                        print("✅ HOÀN THÀNH!")
                    else:
                        print(f"🔄 ĐANG CHẠY... ({remaining_videos} videos còn lại)")
                else:
                    print("📊 Chưa có thông tin tổng số videos")
                    print(f"📹 Đã xử lý: {processed_videos} videos")
                    print(f"⏱️  Thời gian: {total_time/3600:.2f}h")
                
            except Exception as e:
                print(f"❌ Lỗi đọc file kết quả: {e}")
        else:
            print("⏳ Chưa có file kết quả (có thể đang khởi tạo)")
    
    print(f"\n🔄 Để cập nhật tiến độ, chạy lại script này")

def main():
    parser = argparse.ArgumentParser(description='Kiểm tra tiến độ benchmark')
    parser.add_argument('--benchmark-dir', default='benchmark_results',
                       help='Thư mục chứa kết quả benchmark')
    
    args = parser.parse_args()
    check_progress(args.benchmark_dir)

if __name__ == "__main__":
    main()
