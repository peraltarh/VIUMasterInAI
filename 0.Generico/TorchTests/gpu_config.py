#!/usr/bin/env python3
"""
AMD GPU Device Selection and Configuration Script
"""

import torch
import os

def list_amd_devices():
    """List all available AMD devices"""
    print("=" * 60)
    print("AMD GPU Device Detection")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm devices available")
        return
    
    device_count = torch.cuda.device_count()
    print(f"Total devices found: {device_count}")
    print()
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        properties = torch.cuda.get_device_properties(i)
        
        print(f"Device {i}:")
        print(f"  Name: {device_name}")
        print(f"  Total Memory: {properties.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessor Count: {properties.multi_processor_count}")
        print(f"  Compute Capability: {properties.major}.{properties.minor}")
        print()

def set_gpu_device(device_id=None):
    """Set specific GPU device"""
    if device_id is None:
        # Try to automatically detect the discrete GPU
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i).lower()
            # Look for discrete GPU indicators
            if any(indicator in device_name for indicator in ['rx', '9070', 'xt', 'radeon rx']):
                device_id = i
                print(f"🎯 Auto-detected discrete GPU at device {i}: {torch.cuda.get_device_name(i)}")
                break
        
        if device_id is None:
            print("⚠️  Could not auto-detect discrete GPU, using device 0")
            device_id = 0
    
    # Set the device
    torch.cuda.set_device(device_id)
    current_device = torch.cuda.current_device()
    current_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ Set active device to: {current_device} ({current_name})")
    return device_id

def test_gpu_operations(device_id):
    """Test operations on specific GPU"""
    print("\n" + "=" * 60)
    print(f"Testing GPU Operations on Device {device_id}")
    print("=" * 60)
    
    try:
        device = torch.device(f'cuda:{device_id}')
        
        # Create tensors on specific device
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        print(f"✅ Created tensors on {device}")
        print(f"  Tensor x shape: {x.shape}")
        print(f"  Tensor x device: {x.device}")
        
        # Perform computation
        z = torch.matmul(x, y)
        print(f"✅ Matrix multiplication completed")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        
        print(f"📊 GPU Memory Usage:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved: {memory_reserved:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def configure_environment():
    """Configure environment variables for AMD GPU"""
    print("\n" + "=" * 60)
    print("Environment Configuration")
    print("=" * 60)
    
    # Environment variables that can help with AMD GPU selection
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '1',  # Try to use device 1 (discrete GPU)
        'HIP_VISIBLE_DEVICES': '1',   # AMD equivalent
        'HSA_OVERRIDE_GFX_VERSION': '11.0.0',  # For newer GPUs
        'AMD_SERIALIZE_KERNEL': '1',   # For debugging
    }
    
    print("Recommended environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
        # Optionally set them
        # os.environ[key] = value
    
    print("\nTo set these permanently, add to your system environment or use:")
    print("$env:CUDA_VISIBLE_DEVICES='1'  # PowerShell")
    print("set CUDA_VISIBLE_DEVICES=1     # Command Prompt")

if __name__ == "__main__":
    print("🔍 AMD GPU Configuration Tool")
    
    # List all devices
    list_amd_devices()
    
    # Try to set discrete GPU
    if torch.cuda.is_available():
        device_id = set_gpu_device()
        
        # Test operations
        test_gpu_operations(device_id)
    
    # Show environment configuration
    configure_environment()
    
    print("\n🎉 Configuration complete!")