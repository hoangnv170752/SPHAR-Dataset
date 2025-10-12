#!/usr/bin/env python3
"""
Batch Detection Benchmark Script
Chạy YOLO detection cho toàn bộ dataset train để đo thời gian xử lý

Usage:
python batch_detection_benchmark.py --dataset /path/to/abnormal_detection_3class_dataset --output benchmark_results/
"""

import argparse
import json
import time
import csv
from pathlib import Path
from collections import defaultdict
import pandas as pd

from yolo_detect import YOLOHumanDetector

class DatasetDetectionBenchmark:
    def __init__(self, yolo_model="yolov8n.pt", device='cuda', conf_threshold=0.5):
        """Initialize benchmark với YOLO detector"""
        print("Khởi tạo Dataset Detection Benchmark...")
        
        self.detector = YOLOHumanDetector(
            model_path=yolo_model,
            device=device,
            conf_threshold=conf_threshold
        )
        
        self.benchmark_results = {
            'dataset_info': {},
            'video_results': {},
            'category_stats': defaultdict(lambda: {
                'total_videos': 0,
                'total_time': 0,
                'avg_time_per_video': 0,
                'total_detections': 0,
                'avg_detections_per_video': 0
            }),
            'overall_stats': {}
        }
        
    def process_dataset(self, dataset_path, output_dir, splits=['train'], save_detection_results=False, sample_size=None):
        """
        Xử lý toàn bộ dataset và đo benchmark
        
        Args:
            dataset_path: Đường dẫn đến dataset (abnormal_detection_3class_dataset)
            output_dir: Thư mục lưu kết quả benchmark
            splits: Danh sách splits cần xử lý ['train', 'val', 'test']
            save_detection_results: Có lưu kết quả detection không
            sample_size: Số lượng videos random để benchmark (None = tất cả)
        """
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Bắt đầu benchmark dataset: {dataset_path}")
        print(f"Kết quả sẽ lưu tại: {output_dir}")
        if sample_size:
            print(f"🎲 Chế độ sampling: {sample_size} videos ngẫu nhiên")
        
        # Collect tất cả video files trước
        all_video_files = []
        dataset_structure = {}
        
        for split in splits:
            split_path = dataset_path / 'videos' / split
            if split_path.exists():
                dataset_structure[split] = {}
                for category_dir in split_path.iterdir():
                    if category_dir.is_dir():
                        category_name = category_dir.name
                        video_files = list(category_dir.glob('*.mp4')) + list(category_dir.glob('*.avi'))
                        dataset_structure[split][category_name] = len(video_files)
                        
                        # Thêm vào danh sách tổng với metadata
                        for video_file in video_files:
                            all_video_files.append({
                                'path': video_file,
                                'split': split,
                                'category': category_name
                            })
        
        total_videos_count = len(all_video_files)
        
        # Random sampling nếu cần
        if sample_size and sample_size < total_videos_count:
            import random
            random.seed(42)  # Để kết quả reproducible
            sampled_videos = random.sample(all_video_files, sample_size)
            print(f"🎯 Đã chọn {sample_size} videos từ {total_videos_count} videos")
        else:
            sampled_videos = all_video_files
            print(f"🎯 Xử lý tất cả {total_videos_count} videos")
        
        print(f"\n📊 TỔNG QUAN DATASET:")
        print(f"{'='*50}")
        for split, categories in dataset_structure.items():
            split_total = sum(categories.values())
            print(f"📁 {split.upper()}: {split_total} videos")
            for category, count in categories.items():
                print(f"   └── {category}: {count} videos")
        print(f"🎯 TỔNG CỘNG: {total_videos_count} videos")
        print(f"⏰ Ước tính thời gian: ~{total_videos_count * 0.5 / 3600:.1f}h (0.5s/video)")
        print(f"{'='*50}\n")
        
        # Lưu thông tin dataset
        self.benchmark_results['dataset_info'] = {
            'dataset_path': str(dataset_path),
            'splits_processed': splits,
            'total_videos': total_videos_count,
            'sampled_videos': len(sampled_videos),
            'sample_size': sample_size,
            'dataset_structure': dataset_structure,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        total_videos_processed = 0
        total_processing_time = 0
        
        print(f"\n🚀 BẮT ĐẦU XỬ LÝ {len(sampled_videos)} VIDEOS")
        print(f"{'='*60}")
        
        # Group sampled videos by category for stats
        category_stats = defaultdict(lambda: {'total_time': 0, 'total_detections': 0, 'count': 0})
        
        # Xử lý từng video đã được sample
        for i, video_info in enumerate(sampled_videos, 1):
            video_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"📹 [{video_info['split']}] [{video_info['category']}] Video {i}/{len(sampled_videos)}")
            print(f"📁 File: {video_info['path'].name}")
            print(f"⏰ Thời gian: {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            # Tạo output directory cho video này nếu cần
            video_output_dir = None
            if save_detection_results:
                video_output_dir = output_dir / 'detection_results' / video_info['split'] / video_info['category'] / video_info['path'].stem
            
            # Chạy detection
            detection_results = self.detector.process_video(
                video_path=video_info['path'],
                output_dir=video_output_dir,
                save_frames=False,
                visualize=False
            )
            
            processing_time = time.time() - video_start_time
            
            if detection_results:
                total_detections = detection_results['processing_stats']['total_detections']
                fps_achieved = detection_results['processing_stats']['fps_achieved']
                total_frames = detection_results['video_info']['total_frames']
                
                # Lưu kết quả cho video này
                video_key = f"{video_info['split']}_{video_info['category']}_{video_info['path'].stem}"
                self.benchmark_results['video_results'][video_key] = {
                    'video_path': str(video_info['path']),
                    'split': video_info['split'],
                    'category': video_info['category'],
                    'processing_time': processing_time,
                    'total_frames': total_frames,
                    'total_detections': total_detections,
                    'fps_achieved': fps_achieved,
                    'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
                }
                
                category_key = f"{video_info['split']}_{video_info['category']}"
                category_stats[category_key]['total_time'] += processing_time
                category_stats[category_key]['total_detections'] += total_detections
                category_stats[category_key]['count'] += 1
                
                total_videos_processed += 1
                total_processing_time += processing_time
                
                # Progress info với thời gian ước tính
                remaining_videos = len(sampled_videos) - i
                avg_time_per_video = total_processing_time / i
                estimated_remaining_time = remaining_videos * avg_time_per_video
                
                print(f"✅ HOÀN THÀNH trong {processing_time:.2f}s")
                print(f"📊 Detections: {total_detections} | FPS: {fps_achieved:.1f}")
                print(f"⏱️  Còn lại: {remaining_videos} videos (~{estimated_remaining_time/60:.1f} phút)")
                print(f"📈 Tiến độ: {i}/{len(sampled_videos)} ({i/len(sampled_videos)*100:.1f}%)")
                
            else:
                print(f"❌ LỖI xử lý video: {video_info['path'].name}")
        
        # Tạo category stats từ sampled data
        for category_key, stats in category_stats.items():
            if stats['count'] > 0:
                self.benchmark_results['category_stats'][category_key] = {
                    'split': category_key.split('_')[0],
                    'category': '_'.join(category_key.split('_')[1:]),
                    'total_videos': stats['count'],
                    'total_time': stats['total_time'],
                    'avg_time_per_video': stats['total_time'] / stats['count'],
                    'total_detections': stats['total_detections'],
                    'avg_detections_per_video': stats['total_detections'] / stats['count']
                }
        
        # Tính toán overall stats
        self.benchmark_results['overall_stats'] = {
            'total_videos_processed': total_videos_processed,
            'total_processing_time': total_processing_time,
            'avg_time_per_video': total_processing_time / total_videos_processed if total_videos_processed > 0 else 0,
            'videos_per_hour': 3600 / (total_processing_time / total_videos_processed) if total_videos_processed > 0 else 0
        }
        
        # Lưu kết quả
        self.save_benchmark_results(output_dir)
        
        # In summary
        self.print_benchmark_summary()
        
        return self.benchmark_results
    
    def save_benchmark_results(self, output_dir):
        """Lưu kết quả benchmark"""
        output_dir = Path(output_dir)
        
        # Lưu JSON đầy đủ
        json_path = output_dir / 'benchmark_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.benchmark_results, f, indent=2, ensure_ascii=False)
        print(f"Kết quả JSON lưu tại: {json_path}")
        
        # Lưu CSV cho video results
        video_csv_path = output_dir / 'video_benchmark.csv'
        video_data = []
        for video_key, video_info in self.benchmark_results['video_results'].items():
            video_data.append(video_info)
        
        if video_data:
            df_videos = pd.DataFrame(video_data)
            df_videos.to_csv(video_csv_path, index=False, encoding='utf-8')
            print(f"Video benchmark CSV lưu tại: {video_csv_path}")
        
        # Lưu CSV cho category stats
        category_csv_path = output_dir / 'category_benchmark.csv'
        category_data = []
        for category_key, category_info in self.benchmark_results['category_stats'].items():
            category_data.append(category_info)
        
        if category_data:
            df_categories = pd.DataFrame(category_data)
            df_categories.to_csv(category_csv_path, index=False, encoding='utf-8')
            print(f"Category benchmark CSV lưu tại: {category_csv_path}")
    
    def print_benchmark_summary(self):
        """In tóm tắt kết quả benchmark"""
        stats = self.benchmark_results['overall_stats']
        
        print(f"\n{'='*80}")
        print(f"TÓM TẮT KẾT QUẢ BENCHMARK")
        print(f"{'='*80}")
        print(f"Tổng số videos xử lý: {stats['total_videos_processed']}")
        print(f"Tổng thời gian xử lý: {stats['total_processing_time']:.2f} giây ({stats['total_processing_time']/3600:.2f} giờ)")
        print(f"Thời gian trung bình/video: {stats['avg_time_per_video']:.2f} giây")
        print(f"Tốc độ xử lý: {stats['videos_per_hour']:.1f} videos/giờ")
        
        print(f"\nKẾT QUẢ THEO CATEGORY:")
        print(f"{'Category':<30} {'Videos':<8} {'Tổng thời gian':<15} {'TB/video':<12} {'TB detections':<15}")
        print(f"{'-'*80}")
        
        for category_key, category_stats in self.benchmark_results['category_stats'].items():
            print(f"{category_key:<30} {category_stats['total_videos']:<8} "
                  f"{category_stats['total_time']:<15.2f} {category_stats['avg_time_per_video']:<12.2f} "
                  f"{category_stats['avg_detections_per_video']:<15.1f}")

def main():
    parser = argparse.ArgumentParser(description='Dataset Detection Benchmark')
    parser.add_argument('--dataset', '-d', required=True, 
                       help='Đường dẫn đến dataset (abnormal_detection_3class_dataset)')
    parser.add_argument('--output', '-o', required=True, 
                       help='Thư mục lưu kết quả benchmark')
    parser.add_argument('--splits', nargs='+', default=['train'], 
                       help='Danh sách splits cần xử lý (default: train)')
    parser.add_argument('--model', '-m', default='yolov8n.pt', 
                       help='YOLO model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--device', default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--save-detections', action='store_true',
                       help='Lưu kết quả detection cho từng video')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Số lượng videos random để benchmark (None = tất cả)')
    
    args = parser.parse_args()
    
    # Kiểm tra dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Lỗi: Không tìm thấy dataset tại {dataset_path}")
        return
    
    # Khởi tạo benchmark
    benchmark = DatasetDetectionBenchmark(
        yolo_model=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    # Chạy benchmark
    results = benchmark.process_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        splits=args.splits,
        save_detection_results=args.save_detections,
        sample_size=args.sample_size
    )
    
    print(f"\nBenchmark hoàn thành! Kết quả lưu tại: {args.output}")

if __name__ == "__main__":
    main()
