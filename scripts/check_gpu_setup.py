#!/usr/bin/env python3
"""
Script to check GPU setup and provide recommendations
"""

import torch
import platform
import subprocess
import sys

def check_gpu_setup():
    print("="*60)
    print("🔍 GPU SETUP DIAGNOSTIC")
    print("="*60)
    
    # System info
    print(f"🖥️ System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # CUDA check
    print(f"\n📊 CUDA Status:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  CUDA Version (PyTorch): {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"  ✅ GPU Test: Success")
        except Exception as e:
            print(f"  ❌ GPU Test: Failed - {e}")
    else:
        print("  ❌ No CUDA GPU detected")
        
        # Check if NVIDIA GPU exists
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("  💡 NVIDIA GPU detected but CUDA not available in PyTorch")
                print("  🔧 Solution: Install PyTorch with CUDA support")
            else:
                print("  💡 No NVIDIA GPU detected")
        except FileNotFoundError:
            print("  💡 nvidia-smi not found - No NVIDIA GPU or drivers")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not torch.cuda.is_available():
        print("  For GPU Training:")
        print("  1. Install NVIDIA GPU drivers")
        print("  2. Install CUDA Toolkit (11.8 or 12.1)")
        print("  3. Install PyTorch with CUDA:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  4. Or use Google Colab/Kaggle for free GPU")
        
        print("\n  For CPU Training (Current Setup):")
        print("  ✅ Use smaller batch sizes (4-8)")
        print("  ✅ Use smaller image sizes (416x416)")
        print("  ✅ Reduce epochs for testing")
        print("  ✅ Enable mixed precision (AMP)")
        
    else:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 6:
            print("  ⚠️ Low GPU memory - use batch size 4-8")
        elif gpu_memory < 12:
            print("  ✅ Medium GPU memory - use batch size 8-16")
        else:
            print("  🚀 High GPU memory - use batch size 16-32")
    
    print("="*60)

if __name__ == "__main__":
    check_gpu_setup()
