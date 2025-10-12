#!/usr/bin/env python3
"""
Detection + Tracking Pipeline
Combines YOLO detection and DeepSORT tracking for human activity analysis

Usage:
python detection_tracking_pipeline.py --video video.mp4 --output results/
"""

import argparse
import json
import pickle
from pathlib import Path
import time

from yolo_detect import YOLOHumanDetector
from deepsort_track import DeepSORTTracker

class DetectionTrackingPipeline:
    def __init__(self, yolo_model="yolov8n.pt", conf_threshold=0.5, device='cuda'):
        """Initialize the complete pipeline"""
        print("Initializing Detection + Tracking Pipeline...")
        
        # Initialize YOLO detector
        self.detector = YOLOHumanDetector(
            model_path=yolo_model,
            device=device,
            conf_threshold=conf_threshold
        )
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSORTTracker()
        
        print("Pipeline ready!")
    
    def process_video(self, video_path, output_dir, save_intermediate=True):
        """
        Process video through complete pipeline
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save all results
            save_intermediate: Whether to save intermediate detection results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        # Step 1: YOLO Detection
        print("\nStep 1: Running YOLO Human Detection...")
        detection_start = time.time()
        
        detection_results = self.detector.process_video(
            video_path=video_path,
            output_dir=output_dir / 'detection_results' if save_intermediate else None,
            save_frames=False,
            visualize=False
        )
        
        detection_time = time.time() - detection_start
        print(f"Detection completed in {detection_time:.2f} seconds")
        
        if detection_results is None:
            print("Detection failed!")
            return None
        
        # Step 2: DeepSORT Tracking
        print("\nStep 2: Running DeepSORT Tracking...")
        tracking_start = time.time()
        
        tracking_results = self.tracker.process_video(
            video_path=video_path,
            detection_results=detection_results,
            output_dir=output_dir / 'tracking_results'
        )
        
        tracking_time = time.time() - tracking_start
        print(f"Tracking completed in {tracking_time:.2f} seconds")
        
        # Step 3: Generate Pipeline Summary
        print("\nStep 3: Generating Pipeline Summary...")
        pipeline_summary = self.generate_pipeline_summary(
            detection_results, tracking_results, detection_time, tracking_time
        )
        
        # Save pipeline summary
        summary_path = output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2)
        
        print(f"\nPipeline Summary:")
        print(f"- Total processing time: {detection_time + tracking_time:.2f} seconds")
        print(f"- Detection time: {detection_time:.2f} seconds")
        print(f"- Tracking time: {tracking_time:.2f} seconds")
        print(f"- Total detections: {pipeline_summary['detection_stats']['total_detections']}")
        print(f"- Unique tracks: {pipeline_summary['tracking_stats']['total_tracks']}")
        print(f"- Results saved to: {output_dir}")
        
        return {
            'detection_results': detection_results,
            'tracking_results': tracking_results,
            'pipeline_summary': pipeline_summary
        }
    
    def generate_pipeline_summary(self, detection_results, tracking_results, detection_time, tracking_time):
        """Generate comprehensive pipeline summary"""
        summary = {
            'pipeline_info': {
                'detection_time': detection_time,
                'tracking_time': tracking_time,
                'total_time': detection_time + tracking_time
            },
            'detection_stats': detection_results.get('processing_stats', {}),
            'tracking_stats': {
                'total_tracks': len(tracking_results.get('track_data', {})),
                'avg_track_length': 0
            },
            'video_info': detection_results.get('video_info', {})
        }
        
        # Calculate average track length
        track_data = tracking_results.get('track_data', {})
        if track_data:
            track_lengths = [len(track_info) for track_info in track_data.values()]
            summary['tracking_stats']['avg_track_length'] = sum(track_lengths) / len(track_lengths)
        
        return summary
    
    def process_batch(self, video_directory, output_base_dir, video_extensions=None):
        """Process multiple videos"""
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov']
        
        video_dir = Path(video_directory)
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
        
        if not video_files:
            print(f"No video files found in {video_dir}")
            return
        
        print(f"Found {len(video_files)} videos to process")
        
        batch_results = {}
        output_base = Path(output_base_dir)
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*80}")
            print(f"Processing video {i}/{len(video_files)}: {video_path.name}")
            print(f"{'='*80}")
            
            video_output_dir = output_base / video_path.stem
            results = self.process_video(video_path, video_output_dir)
            
            if results:
                batch_results[video_path.name] = results['pipeline_summary']
        
        # Save batch summary
        batch_summary_path = output_base / 'batch_summary.json'
        with open(batch_summary_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nBatch processing completed!")
        print(f"Processed {len(batch_results)} videos successfully")
        print(f"Batch summary saved to: {batch_summary_path}")
        
        return batch_results

def main():
    parser = argparse.ArgumentParser(description='Detection + Tracking Pipeline')
    parser.add_argument('--video', '-v', help='Input video file')
    parser.add_argument('--batch', '-b', help='Directory containing videos to process')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-intermediate', action='store_true', 
                       help='Save intermediate detection results')
    
    args = parser.parse_args()
    
    if not args.video and not args.batch:
        print("Error: Must specify either --video or --batch")
        return
    
    # Initialize pipeline
    pipeline = DetectionTrackingPipeline(
        yolo_model=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    if args.batch:
        # Process batch of videos
        pipeline.process_batch(args.batch, args.output)
    else:
        # Process single video
        pipeline.process_video(
            video_path=args.video,
            output_dir=args.output,
            save_intermediate=args.save_intermediate
        )

if __name__ == "__main__":
    main()
