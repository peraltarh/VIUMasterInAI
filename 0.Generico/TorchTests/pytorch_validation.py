#!/usr/bin/env python3
"""
Comprehensive PyTorch ROCm Installation Validation Script
"""

import sys
import os

def test_pytorch_installation():
    """Test PyTorch installation and GPU support"""
    print("=" * 60)
    print("PyTorch ROCm Installation Validation")
    print("=" * 60)
    
    # Test 1: Python Version
    print(f"✓ Python Version: {sys.version}")
    print(f"✓ Python Executable: {sys.executable}")
    print()
    
    # Test 2: PyTorch Import
    try:
        import torch
        print("✅ PyTorch Import: SUCCESS")
        print(f"✓ PyTorch Version: {torch.__version__}")
    except ImportError as e:
        print("❌ PyTorch Import: FAILED")
        print(f"Error: {e}")
        return False
    
    # Test 3: Check for ROCm/HIP support
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    
    # Check for ROCm/HIP
    has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    print(f"✓ ROCm/HIP Available: {has_rocm}")
    
    if has_rocm:
        print(f"✓ HIP Version: {torch.version.hip}")
    
    # Test 4: Device Detection
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ GPU Device Count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  - GPU {i}: {device_name}")
    else:
        print("ℹ️  No CUDA/ROCm devices detected")
    
    # Test 5: Tensor Operations
    print("\n" + "-" * 40)
    print("Testing Tensor Operations:")
    print("-" * 40)
    
    try:
        # CPU tensor test
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"✅ CPU Tensor: {cpu_tensor}")
        
        # GPU tensor test (if available)
        if torch.cuda.is_available():
            print("🔍 Attempting GPU operations...")
            
            # Set environment variable for better debugging
            os.environ['AMD_SERIALIZE_KERNEL'] = '3'
            
            # Try to get current device info first
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✓ Current GPU Device: {current_device} ({device_name})")
            
            # Test GPU memory allocation first
            try:
                print("🧪 Testing GPU memory allocation...")
                gpu_tensor = cpu_tensor.cuda()
                print(f"✅ GPU Tensor: {gpu_tensor}")
                print(f"✓ GPU Device: {gpu_tensor.device}")
                
                # Force synchronization to catch errors early
                torch.cuda.synchronize()
                print("✅ GPU synchronization successful")
                
                # Simple computation test with error checking
                print("🧪 Testing GPU computation...")
                result = gpu_tensor * 2
                torch.cuda.synchronize()  # Ensure computation completes
                print(f"✅ GPU Computation (x2): {result}")
                
                # Move back to CPU
                cpu_result = result.cpu()
                print(f"✅ Back to CPU: {cpu_result}")
                
                # Test more complex operation
                print("🧪 Testing matrix operations...")
                matrix_a = torch.randn(10, 10).cuda()
                matrix_b = torch.randn(10, 10).cuda()
                matrix_c = torch.matmul(matrix_a, matrix_b)
                torch.cuda.synchronize()
                print(f"✅ Matrix multiplication successful: {matrix_c.shape}")
                
                # Clean up GPU memory
                del gpu_tensor, result, matrix_a, matrix_b, matrix_c
                torch.cuda.empty_cache()
                print("✅ GPU memory cleaned up")
                
            except RuntimeError as gpu_error:
                print(f"❌ GPU operation failed: {gpu_error}")
                print("\n🔧 Troubleshooting suggestions:")
                print("1. Try setting environment variable: AMD_SERIALIZE_KERNEL=3")
                print("2. Check if other applications are using the GPU")
                print("3. Try restarting and running the validation again")
                print("4. Consider updating ROCm drivers")
                
                # Try to continue with CPU only
                print("\n⚠️  Continuing with CPU-only operations...")
                return False
            
        else:
            print("ℹ️  GPU tests skipped (no GPU available)")
            
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        return False
    
    # Test 6: Additional Libraries
    print("\n" + "-" * 40)
    print("Testing Additional Libraries:")
    print("-" * 40)
    
    libraries = [
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
    ]
    
    for name, module in libraries:
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', 'Unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"⚠️  {name}: Not installed")
    
    print("\n" + "=" * 60)
    print("🎉 PyTorch installation validation complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_pytorch_installation()