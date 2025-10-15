#!/usr/bin/env python3
"""
Benchmark script to compare YOLOv11s pretrained vs fine-tuned model
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

class ModelBenchmark:
    def __init__(self, model_path, model_name, conf_threshold=0.25):
        """Initialize model for benchmarking"""
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        print(f"\n🔥 Loading {model_name}: {self.model_path.name}")
        self.model = YOLO(str(self.model_path))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ Device: {self.device}")
        
        # Metrics
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.total_frames = 0
        self.total_detections = 0
        self.total_inference_time = 0
        self.fps_history = []
        self.confidence_scores = []
        self.frame_detections = []
        
    def benchmark_video(self, video_path, max_frames=None):
        """Benchmark model on video"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        print(f"\n📹 Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return None
        
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_video_frames = min(total_video_frames, max_frames)
        
        print(f"   Frames to process: {total_video_frames}")
        
        self.reset_metrics()
        
        pbar = tqdm(total=total_video_frames, desc=f"   {self.model_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and self.total_frames >= max_frames:
                break
            
            # Inference
            start_time = time.time()
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            # Extract detections
            num_detections = 0
            frame_confidences = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Filter for person class (class 0)
                    boxes = result.boxes
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    
                    person_mask = classes == 0  # person class
                    person_confidences = confidences[person_mask]
                    
                    num_detections = len(person_confidences)
                    frame_confidences = person_confidences.tolist()
            
            # Update metrics
            self.total_frames += 1
            self.total_detections += num_detections
            self.total_inference_time += inference_time
            self.fps_history.append(1.0 / inference_time if inference_time > 0 else 0)
            self.confidence_scores.extend(frame_confidences)
            self.frame_detections.append(num_detections)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        return self.get_metrics()
    
    def get_metrics(self):
        """Get benchmark metrics"""
        if self.total_frames == 0:
            return None
        
        metrics = {
            'model_name': self.model_name,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / self.total_frames,
            'avg_inference_time': self.total_inference_time / self.total_frames,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_fps': np.min(self.fps_history) if self.fps_history else 0,
            'max_fps': np.max(self.fps_history) if self.fps_history else 0,
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'min_confidence': np.min(self.confidence_scores) if self.confidence_scores else 0,
            'max_confidence': np.max(self.confidence_scores) if self.confidence_scores else 0,
            'frames_with_detections': sum(1 for d in self.frame_detections if d > 0),
            'detection_rate': sum(1 for d in self.frame_detections if d > 0) / self.total_frames * 100
        }
        
        return metrics

def compare_models(model1_metrics, model2_metrics):
    """Compare two model metrics"""
    print("\n" + "="*80)
    print("📊 BENCHMARK COMPARISON")
    print("="*80)
    
    # Table header
    print(f"\n{'Metric':<30} {'Model 1 (Pretrained)':<25} {'Model 2 (Fine-tuned)':<25} {'Improvement':<15}")
    print("-"*95)
    
    metrics_to_compare = [
        ('Average FPS', 'avg_fps', '{:.2f}', 'higher'),
        ('Avg Inference Time (ms)', 'avg_inference_time', '{:.1f}', 'lower', 1000),
        ('Total Detections', 'total_detections', '{:.0f}', 'higher'),
        ('Avg Detections/Frame', 'avg_detections_per_frame', '{:.2f}', 'higher'),
        ('Detection Rate (%)', 'detection_rate', '{:.1f}%', 'higher'),
        ('Avg Confidence', 'avg_confidence', '{:.3f}', 'higher'),
        ('Min Confidence', 'min_confidence', '{:.3f}', 'higher'),
        ('Frames w/ Detections', 'frames_with_detections', '{:.0f}', 'higher'),
    ]
    
    improvements = {}
    
    for metric_name, key, fmt, better, *multiplier in metrics_to_compare:
        mult = multiplier[0] if multiplier else 1
        val1 = model1_metrics.get(key, 0) * mult
        val2 = model2_metrics.get(key, 0) * mult
        
        # Calculate improvement
        if better == 'higher':
            improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
            symbol = "🟢" if val2 > val1 else "🔴"
        else:  # lower is better
            improvement = ((val1 - val2) / val1 * 100) if val1 > 0 else 0
            symbol = "🟢" if val2 < val1 else "🔴"
        
        improvements[key] = improvement
        
        val1_str = fmt.format(val1)
        val2_str = fmt.format(val2)
        imp_str = f"{symbol} {improvement:+.1f}%"
        
        print(f"{metric_name:<30} {val1_str:<25} {val2_str:<25} {imp_str:<15}")
    
    print("="*80)
    
    # Summary
    print("\n📋 SUMMARY:")
    if improvements['avg_fps'] > 0:
        print(f"   🚀 Speed: Fine-tuned model is {improvements['avg_fps']:.1f}% faster")
    else:
        print(f"   🐌 Speed: Fine-tuned model is {-improvements['avg_fps']:.1f}% slower")
    
    if improvements['avg_detections_per_frame'] > 0:
        print(f"   🎯 Detection: Fine-tuned model detects {improvements['avg_detections_per_frame']:.1f}% more people")
    else:
        print(f"   ⚠️  Detection: Fine-tuned model detects {-improvements['avg_detections_per_frame']:.1f}% fewer people")
    
    if improvements['avg_confidence'] > 0:
        print(f"   💪 Confidence: Fine-tuned model is {improvements['avg_confidence']:.1f}% more confident")
    else:
        print(f"   ⚠️  Confidence: Fine-tuned model is {-improvements['avg_confidence']:.1f}% less confident")
    
    # Overall verdict
    score = (
        improvements['avg_detections_per_frame'] * 0.4 +
        improvements['avg_confidence'] * 0.3 +
        improvements['detection_rate'] * 0.3
    )
    
    print(f"\n🏆 OVERALL SCORE: {score:+.1f}%")
    if score > 10:
        print("   ✅ Fine-tuned model shows SIGNIFICANT improvement!")
    elif score > 0:
        print("   ✅ Fine-tuned model shows improvement")
    else:
        print("   ⚠️  Fine-tuned model needs more training")
    
    return improvements

def save_benchmark_results(model1_metrics, model2_metrics, improvements, output_path):
    """Save benchmark results to JSON"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model1': model1_metrics,
        'model2': model2_metrics,
        'improvements': improvements,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark YOLOv11s models')
    parser.add_argument('--model1', default=r'D:\SPHAR-Dataset\models\yolo11s.pt',
                       help='Path to pretrained model')
    parser.add_argument('--model2', default=r'D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt',
                       help='Path to fine-tuned model')
    parser.add_argument('--video', default=None,
                       help='Test video path (if not provided, use sample)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--output', default=r'D:\SPHAR-Dataset\benchmark_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Find test video if not provided
    if not args.video:
        videos_dir = Path(r'D:\SPHAR-Dataset\videos')
        test_videos = [
            videos_dir / 'walking' / 'casia_angleview_p01p02_meettogether_a1.mp4',
            videos_dir / 'walking' / 'casia_angleview_p01_walk_a1.mp4',
        ]
        
        for video in test_videos:
            if video.exists():
                args.video = str(video)
                break
        
        if not args.video:
            print("❌ No test video found. Please specify --video")
            return
    
    print("="*80)
    print("🏁 YOLO MODEL BENCHMARK")
    print("="*80)
    print(f"📹 Test video: {Path(args.video).name}")
    print(f"🎯 Confidence threshold: {args.conf}")
    if args.max_frames:
        print(f"🎬 Max frames: {args.max_frames}")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Benchmark Model 1 (Pretrained)
    print("\n" + "="*80)
    print("📦 MODEL 1: Pretrained YOLOv11s")
    print("="*80)
    benchmark1 = ModelBenchmark(args.model1, "Pretrained", args.conf)
    metrics1 = benchmark1.benchmark_video(args.video, args.max_frames)
    
    if metrics1 is None:
        print("❌ Benchmark 1 failed")
        return
    
    # Benchmark Model 2 (Fine-tuned)
    print("\n" + "="*80)
    print("🎯 MODEL 2: Fine-tuned YOLOv11s")
    print("="*80)
    benchmark2 = ModelBenchmark(args.model2, "Fine-tuned", args.conf)
    metrics2 = benchmark2.benchmark_video(args.video, args.max_frames)
    
    if metrics2 is None:
        print("❌ Benchmark 2 failed")
        return
    
    # Compare
    improvements = compare_models(metrics1, metrics2)
    
    # Save results
    save_benchmark_results(metrics1, metrics2, improvements, args.output)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
