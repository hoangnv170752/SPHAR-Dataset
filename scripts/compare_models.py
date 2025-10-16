#!/usr/bin/env python3
"""
Compare Different Model Versions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from torch.utils.data import DataLoader

# Import models
try:
    from action_recognition_trainer import SlowFastActionModel
except ImportError:
    SlowFastActionModel = None

try:
    from image_action_trainer import OptimizedSlowFastModel
except ImportError:
    OptimizedSlowFastModel = None

try:
    from improved_model_architecture import ImprovedSlowFastModel
except ImportError:
    ImprovedSlowFastModel = None

from image_action_trainer import ImageSequenceDataset

class ModelComparator:
    """Compare different model versions"""
    
    def __init__(self, data_dir, class_mapping_path):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.num_classes = len(self.class_mapping)
        
        # Test dataset
        self.test_dataset = ImageSequenceDataset(
            data_dir, 'test', sequence_length=8, 
            class_mapping=self.class_mapping
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=8, shuffle=False, num_workers=2
        )
        
        print(f"✅ Loaded {len(self.test_dataset)} test samples")
    
    def load_model(self, model_path, model_type='auto'):
        """Load model with automatic type detection"""
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try different model architectures
        models_to_try = []
        
        if model_type == 'auto':
            if 'improved' in str(model_path).lower():
                models_to_try = [ImprovedSlowFastModel, OptimizedSlowFastModel, SlowFastActionModel]
            elif 'ultra_fast' in str(model_path).lower() or 'optimized' in str(model_path).lower():
                models_to_try = [OptimizedSlowFastModel, ImprovedSlowFastModel, SlowFastActionModel]
            else:
                models_to_try = [SlowFastActionModel, OptimizedSlowFastModel, ImprovedSlowFastModel]
        
        # Filter out None models
        models_to_try = [m for m in models_to_try if m is not None]
        
        for ModelClass in models_to_try:
            try:
                if ModelClass == ImprovedSlowFastModel:
                    model = ModelClass(self.num_classes, sequence_length=8, dropout_rate=0.3)
                else:
                    model = ModelClass(self.num_classes, sequence_length=8)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                print(f"✅ Loaded {ModelClass.__name__} from {Path(model_path).name}")
                return model, ModelClass.__name__
                
            except Exception as e:
                continue
        
        print(f"❌ Could not load model from {model_path}")
        return None, None
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        if model is None:
            return None
        
        model.eval()
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(self.num_classes)}
        class_total = {i: 0 for i in range(self.num_classes)}
        
        inference_times = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / data.size(0))  # Per sample
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i].item() == label:
                        class_correct[label] += 1
        
        # Calculate metrics
        overall_accuracy = 100. * correct / total
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        
        class_accuracies = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = 100. * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0
        
        # Model size
        model_size = sum(p.numel() for p in model.parameters())
        
        return {
            'model_name': model_name,
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'avg_inference_time': avg_inference_time,
            'model_size': model_size,
            'total_samples': total
        }
    
    def compare_models(self, model_paths):
        """Compare multiple models"""
        results = []
        
        for model_path in model_paths:
            print(f"\n🔍 Evaluating {Path(model_path).name}...")
            model, model_type = self.load_model(model_path)
            result = self.evaluate_model(model, model_type)
            
            if result:
                results.append(result)
                print(f"   Accuracy: {result['overall_accuracy']:.2f}%")
                print(f"   Inference: {result['avg_inference_time']:.2f}ms")
                print(f"   Parameters: {result['model_size']:,}")
        
        return results
    
    def plot_comparison(self, results, save_path=None):
        """Plot model comparison"""
        if not results:
            print("❌ No results to plot")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = [r['model_name'] for r in results]
        accuracies = [r['overall_accuracy'] for r in results]
        inference_times = [r['avg_inference_time'] for r in results]
        model_sizes = [r['model_size'] / 1e6 for r in results]  # Millions of parameters
        
        # Overall accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Overall Accuracy Comparison', fontweight='bold')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Inference time comparison
        bars2 = ax2.bar(model_names, inference_times, color=['#d62728', '#9467bd', '#8c564b'])
        ax2.set_ylabel('Inference Time (ms)', fontweight='bold')
        ax2.set_title('Inference Speed Comparison', fontweight='bold')
        
        for bar, time in zip(bars2, inference_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Model size comparison
        bars3 = ax3.bar(model_names, model_sizes, color=['#17becf', '#bcbd22', '#ff9896'])
        ax3.set_ylabel('Parameters (Millions)', fontweight='bold')
        ax3.set_title('Model Size Comparison', fontweight='bold')
        
        for bar, size in zip(bars3, model_sizes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{size:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Per-class accuracy heatmap
        class_names = list(self.class_mapping.keys())
        class_acc_matrix = []
        
        for result in results:
            class_accs = [result['class_accuracies'].get(i, 0) for i in range(self.num_classes)]
            class_acc_matrix.append(class_accs)
        
        im = ax4.imshow(class_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.set_yticks(range(len(model_names)))
        ax4.set_yticklabels(model_names)
        ax4.set_title('Per-Class Accuracy Heatmap', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Accuracy (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(class_names)):
                text = ax4.text(j, i, f'{class_acc_matrix[i][j]:.0f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison chart saved to: {save_path}")
        
        plt.show()
    
    def print_summary_table(self, results):
        """Print summary comparison table"""
        if not results:
            return
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Header
        print(f"{'Model':<20} {'Accuracy':<12} {'Speed (ms)':<12} {'Size (M)':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        # Calculate efficiency score (accuracy / (inference_time * model_size))
        for result in results:
            efficiency = result['overall_accuracy'] / (result['avg_inference_time'] * result['model_size'] / 1e6)
            
            print(f"{result['model_name']:<20} "
                  f"{result['overall_accuracy']:<11.2f}% "
                  f"{result['avg_inference_time']:<11.2f}ms "
                  f"{result['model_size']/1e6:<11.1f}M "
                  f"{efficiency:<11.2f}")
        
        print("-" * 80)
        
        # Best in each category
        best_acc = max(results, key=lambda x: x['overall_accuracy'])
        best_speed = min(results, key=lambda x: x['avg_inference_time'])
        best_size = min(results, key=lambda x: x['model_size'])
        
        print(f"\n🏆 Best Accuracy: {best_acc['model_name']} ({best_acc['overall_accuracy']:.2f}%)")
        print(f"⚡ Fastest: {best_speed['model_name']} ({best_speed['avg_inference_time']:.2f}ms)")
        print(f"📦 Smallest: {best_size['model_name']} ({best_size['model_size']/1e6:.1f}M params)")

def main():
    # Model paths to compare
    models_dir = Path(r'D:\SPHAR-Dataset\models')
    model_paths = [
        models_dir / 'ultra_fast_action_model.pt',
        models_dir / 'improved_action_model.pt',
        # Add more model paths as needed
    ]
    
    # Filter existing models
    existing_models = [p for p in model_paths if p.exists()]
    
    if not existing_models:
        print("❌ No models found to compare")
        print("Available models:")
        for model_file in models_dir.glob('*.pt'):
            print(f"   {model_file.name}")
        return
    
    print("🔍 COMPARING ACTION RECOGNITION MODELS")
    print("="*80)
    
    # Setup comparator
    data_dir = r'D:\SPHAR-Dataset\action_recognition_images'
    class_mapping_path = r'D:\SPHAR-Dataset\action_recognition_optimized\class_mapping.json'
    
    comparator = ModelComparator(data_dir, class_mapping_path)
    
    # Compare models
    results = comparator.compare_models(existing_models)
    
    if results:
        # Print summary
        comparator.print_summary_table(results)
        
        # Plot comparison
        charts_dir = Path(data_dir).parent / 'charts_comparison'
        charts_dir.mkdir(exist_ok=True)
        
        comparison_path = charts_dir / 'model_comparison.png'
        comparator.plot_comparison(results, save_path=comparison_path)
        
        print(f"\n📈 Comparison charts saved to: {charts_dir}")
    else:
        print("❌ No valid results to compare")

if __name__ == "__main__":
    main()
