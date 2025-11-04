#!/usr/bin/env python3
"""
Quick GPU Test Script
"""

import torch
import os

def quick_gpu_test():
    print("🚀 Quick GPU Test")
    print("=" * 40)
    
    # Show environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    hip_visible = os.environ.get('HIP_VISIBLE_DEVICES', 'Not set')
    
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"HIP_VISIBLE_DEVICES: {hip_visible}")
    print()
    
    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm available")
        return
    
    # Test default device
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"📊 Total devices visible: {device_count}")
    print(f"🎯 Current device: {current_device}")
    print(f"🔥 Current GPU: {device_name}")
    print()
    
    # Quick tensor test
    try:
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = torch.matmul(x, y)
        
        print(f"✅ Tensor creation: SUCCESS")
        print(f"✅ Matrix multiplication: SUCCESS")
        print(f"📍 Computation device: {z.device}")
        print(f"🎮 GPU used: {torch.cuda.get_device_name(z.device)}")
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")

if __name__ == "__main__":
    quick_gpu_test()