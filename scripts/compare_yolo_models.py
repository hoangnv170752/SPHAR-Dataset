#!/usr/bin/env python3
"""
So sánh hiệu suất YOLOv8n, YOLOv10n, YOLOv11n
"""

import json
import pandas as pd
from pathlib import Path
import argparse

def compare_models(benchmark_dir):
    """So sánh kết quả benchmark các models"""
    benchmark_dir = Path(benchmark_dir)
    results = []
    
    # Load kết quả từng model
    for model_dir in benchmark_dir.iterdir():
        if model_dir.is_dir():
            json_file = model_dir / 'benchmark_results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    stats = data.get('overall_stats', {})
                    
                    results.append({
                        'Model': model_dir.name,
                        'Total_Videos': stats.get('total_videos_processed', 0),
                        'Total_Time_s': stats.get('total_processing_time', 0),
                        'Avg_Time_per_Video_s': stats.get('avg_time_per_video', 0),
                        'Videos_per_Hour': stats.get('videos_per_hour', 0)
                    })
    
    if not results:
        print("❌ Không tìm thấy kết quả benchmark")
        return
    
    # Tạo DataFrame và hiển thị
    df = pd.DataFrame(results)
    print("\n📊 SO SÁNH HIỆU SUẤT YOLO MODELS:")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.2f'))
    
    # Tìm model tốt nhất
    if len(df) > 0:
        fastest = df.loc[df['Videos_per_Hour'].idxmax(), 'Model']
        most_efficient = df.loc[df['Avg_Time_per_Video_s'].idxmin(), 'Model']
        
        print(f"\n🏆 Model nhanh nhất: {fastest}")
        print(f"⚡ Model hiệu quả nhất: {most_efficient}")
    
    # Lưu kết quả
    df.to_csv(benchmark_dir / 'models_comparison.csv', index=False)
    print(f"\n💾 Kết quả lưu tại: {benchmark_dir / 'models_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description='So sánh YOLO models')
    parser.add_argument('--benchmark-dir', default='benchmark_results', 
                       help='Thư mục chứa kết quả benchmark')
    
    args = parser.parse_args()
    compare_models(args.benchmark_dir)

if __name__ == "__main__":
    main()
