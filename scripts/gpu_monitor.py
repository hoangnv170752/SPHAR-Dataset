#!/usr/bin/env python3
"""
GPU monitoring script for YOLO training
"""

import time
import torch
import psutil
import threading
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.monitoring = False
        self.stats = []
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 GPU monitoring started...")
        
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("⏹️ GPU monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self._get_current_stats()
                self.stats.append(stats)
                
                # Print current stats
                if torch.cuda.is_available():
                    print(f"[{stats['timestamp']}] "
                          f"GPU: {stats['gpu_util']:.1f}% | "
                          f"Memory: {stats['gpu_memory_used']:.1f}/{stats['gpu_memory_total']:.1f}GB "
                          f"({stats['gpu_memory_percent']:.1f}%) | "
                          f"CPU: {stats['cpu_percent']:.1f}%")
                else:
                    print(f"[{stats['timestamp']}] CPU: {stats['cpu_percent']:.1f}%")
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
                
            time.sleep(self.interval)
    
    def _get_current_stats(self):
        """Get current system stats"""
        stats = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            stats.update({
                'gpu_memory_used': gpu_memory,
                'gpu_memory_total': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory / gpu_memory_total) * 100,
                'gpu_util': self._get_gpu_utilization(),
            })
        
        return stats
    
    def _get_gpu_utilization(self):
        """Get GPU utilization (simplified)"""
        try:
            # This is a simplified approach
            # For more accurate GPU utilization, you'd need nvidia-ml-py
            return min(100, torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100)
        except:
            return 0
    
    def get_summary(self):
        """Get monitoring summary"""
        if not self.stats:
            return "No monitoring data available"
        
        if torch.cuda.is_available():
            gpu_memory_avg = sum(s['gpu_memory_used'] for s in self.stats) / len(self.stats)
            gpu_memory_max = max(s['gpu_memory_used'] for s in self.stats)
            gpu_util_avg = sum(s['gpu_util'] for s in self.stats) / len(self.stats)
            
            return f"""
📊 Training Monitoring Summary:
  Duration: {len(self.stats) * self.interval} seconds
  GPU Memory Average: {gpu_memory_avg:.2f} GB
  GPU Memory Peak: {gpu_memory_max:.2f} GB
  GPU Utilization Average: {gpu_util_avg:.1f}%
  CPU Average: {sum(s['cpu_percent'] for s in self.stats) / len(self.stats):.1f}%
"""
        else:
            cpu_avg = sum(s['cpu_percent'] for s in self.stats) / len(self.stats)
            return f"""
📊 Training Monitoring Summary:
  Duration: {len(self.stats) * self.interval} seconds
  CPU Average: {cpu_avg:.1f}%
"""

# Usage example
if __name__ == "__main__":
    monitor = GPUMonitor(interval=2)
    
    try:
        monitor.start_monitoring()
        
        # Simulate training
        print("Simulating training for 30 seconds...")
        time.sleep(30)
        
    finally:
        monitor.stop_monitoring()
        print(monitor.get_summary())
