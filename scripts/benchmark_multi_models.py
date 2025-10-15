#!/usr/bin/env python3
"""
Benchmark multiple YOLO models on multiple test videos
"""

import cv2
import torch
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

class MultiModelBenchmark:
    def __init__(self):
        """Initialize multi-model benchmark"""
        self.models = {}
        self.results = {}
        
    def load_models(self, model_paths):
        """Load multiple models"""
        print("="*80)
        print("🔥 LOADING MODELS")
        print("="*80)
        
        for name, path in model_paths.items():
            path = Path(path)
            if not path.exists():
                print(f"⚠️  {name}: Not found - {path}")
                continue
            
            print(f"📦 {name}: {path.name}")
            try:
                model = YOLO(str(path))
                self.models[name] = {
                    'model': model,
                    'path': path
                }
                print(f"   ✅ Loaded successfully")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n⚡ Device: {device.upper()}")
        print(f"📊 Models loaded: {len(self.models)}")
        
    def benchmark_video(self, model_name, video_path, conf_threshold=0.25, max_frames=None):
        """Benchmark single model on single video"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]['model']
        video_path = Path(video_path)
        
        if not video_path.exists():
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_video_frames = min(total_video_frames, max_frames)
        
        # Metrics
        frame_count = 0
        total_detections = 0
        total_inference_time = 0
        fps_history = []
        confidence_scores = []
        frame_detections = []
        
        pbar = tqdm(total=total_video_frames, desc=f"   {model_name:15s}", leave=False)
        
        while frame_count < total_video_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            start_time = time.time()
            results = model(frame, conf=conf_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            # Extract person detections (class 0)
            num_detections = 0
            frame_confidences = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    person_mask = classes == 0
                    person_confidences = confidences[person_mask]
                    
                    num_detections = len(person_confidences)
                    frame_confidences = person_confidences.tolist()
            
            # Update metrics
            frame_count += 1
            total_detections += num_detections
            total_inference_time += inference_time
            fps_history.append(1.0 / inference_time if inference_time > 0 else 0)
            confidence_scores.extend(frame_confidences)
            frame_detections.append(num_detections)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Calculate metrics
        if frame_count == 0:
            return None
        
        metrics = {
            'model_name': model_name,
            'video_name': video_path.name,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count,
            'avg_inference_time_ms': (total_inference_time / frame_count) * 1000,
            'avg_fps': np.mean(fps_history),
            'min_fps': np.min(fps_history),
            'max_fps': np.max(fps_history),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
            'frames_with_detections': sum(1 for d in frame_detections if d > 0),
            'detection_rate': sum(1 for d in frame_detections if d > 0) / frame_count * 100
        }
        
        return metrics
    
    def benchmark_all(self, video_paths, conf_threshold=0.25, max_frames=None):
        """Benchmark all models on all videos"""
        print("\n" + "="*80)
        print("🎬 BENCHMARKING ALL MODELS ON ALL VIDEOS")
        print("="*80)
        
        all_results = []
        
        for video_name, video_path in video_paths.items():
            print(f"\n📹 Video: {video_name} ({Path(video_path).name})")
            print("-"*80)
            
            for model_name in self.models.keys():
                metrics = self.benchmark_video(model_name, video_path, conf_threshold, max_frames)
                
                if metrics:
                    metrics['video_id'] = video_name
                    all_results.append(metrics)
        
        self.results = all_results
        return all_results
    
    def create_comparison_table(self):
        """Create comparison table"""
        if not self.results:
            return None
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Pivot table for each metric
        print("\n" + "="*80)
        print("📊 COMPARISON TABLES")
        print("="*80)
        
        metrics_to_show = [
            ('avg_fps', 'Average FPS', '{:.1f}'),
            ('avg_inference_time_ms', 'Inference Time (ms)', '{:.1f}'),
            ('avg_detections_per_frame', 'Detections/Frame', '{:.2f}'),
            ('detection_rate', 'Detection Rate (%)', '{:.1f}'),
            ('avg_confidence', 'Avg Confidence', '{:.3f}'),
        ]
        
        for metric_key, metric_name, fmt in metrics_to_show:
            print(f"\n📈 {metric_name}")
            print("-"*80)
            
            pivot = df.pivot_table(
                values=metric_key,
                index='video_id',
                columns='model_name',
                aggfunc='mean'
            )
            
            # Format and print
            print(pivot.to_string(float_format=lambda x: fmt.format(x)))
        
        return df
    
    def calculate_overall_scores(self):
        """Calculate overall scores for each model"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        # Group by model
        model_stats = df.groupby('model_name').agg({
            'avg_fps': 'mean',
            'avg_detections_per_frame': 'mean',
            'detection_rate': 'mean',
            'avg_confidence': 'mean',
            'total_detections': 'sum'
        }).round(2)
        
        print("\n" + "="*80)
        print("🏆 OVERALL MODEL PERFORMANCE")
        print("="*80)
        print(model_stats.to_string())
        
        # Calculate composite score
        # Normalize metrics to 0-100 scale
        for col in model_stats.columns:
            col_min = model_stats[col].min()
            col_max = model_stats[col].max()
            if col_max > col_min:
                model_stats[f'{col}_normalized'] = ((model_stats[col] - col_min) / (col_max - col_min)) * 100
        
        # Composite score (weighted average)
        model_stats['composite_score'] = (
            model_stats.get('avg_fps_normalized', 0) * 0.2 +
            model_stats.get('avg_detections_per_frame_normalized', 0) * 0.3 +
            model_stats.get('detection_rate_normalized', 0) * 0.3 +
            model_stats.get('avg_confidence_normalized', 0) * 0.2
        )
        
        print("\n" + "="*80)
        print("🎯 COMPOSITE SCORES (0-100)")
        print("="*80)
        
        scores = model_stats['composite_score'].sort_values(ascending=False)
        for i, (model, score) in enumerate(scores.items(), 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
            print(f"{medal} {model:20s}: {score:.1f}")
        
        return model_stats
    
    def save_results(self, output_path):
        """Save results to JSON"""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'models': list(self.models.keys()),
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark multiple YOLO models')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames per video')
    parser.add_argument('--output', default=r'D:\SPHAR-Dataset\multi_model_benchmark.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Define models
    model_paths = {
        'YOLOv8s': r'D:\SPHAR-Dataset\models\yolov8s.pt',
        'YOLOv11s': r'D:\SPHAR-Dataset\models\yolo11s.pt',
        'YOLOv11s-FT': r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
    }
    
    # Define test videos
    videos_dir = Path(r'D:\SPHAR-Dataset\videos')
    test_videos = {
        '1person_walk': videos_dir / 'walking' / 'casia_angleview_p01_walk_a1.mp4',
        '2people_meet': videos_dir / 'walking' / 'casia_angleview_p01p02_meettogether_a1.mp4',
        '2people_follow': videos_dir / 'walking' / 'casia_angleview_p01p02_followtogether_a1.mp4',
        '2people_overtake': videos_dir / 'walking' / 'casia_angleview_p01p02_overtake_a1.mp4',
        'ntu_action': videos_dir / 'NTU' / 'A001' / 'S001C001P001R001A001_rgb.avi',
    }
    
    # Filter existing videos
    test_videos = {k: v for k, v in test_videos.items() if v.exists()}
    
    print("="*80)
    print("🏁 MULTI-MODEL BENCHMARK")
    print("="*80)
    print(f"🎯 Confidence: {args.conf}")
    print(f"📹 Test videos: {len(test_videos)}")
    print(f"🤖 Models: {len([p for p in model_paths.values() if Path(p).exists()])}")
    if args.max_frames:
        print(f"🎬 Max frames: {args.max_frames}")
    
    # Run benchmark
    benchmark = MultiModelBenchmark()
    benchmark.load_models(model_paths)
    
    if not benchmark.models:
        print("❌ No models loaded")
        return
    
    benchmark.benchmark_all(test_videos, args.conf, args.max_frames)
    
    # Show results
    benchmark.create_comparison_table()
    benchmark.calculate_overall_scores()
    
    # Save results
    benchmark.save_results(args.output)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
