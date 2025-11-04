#!/usr/bin/env python3
"""
AMD GPU Troubleshooting Script for PyTorch ROCm
"""

import os
import sys
import torch

def set_amd_debug_environment():
    """Set AMD-specific environment variables for debugging"""
    print("🔧 Setting AMD debugging environment variables...")
    
    # AMD debugging variables
    env_vars = {
        'AMD_SERIALIZE_KERNEL': '3',
        'HIP_VISIBLE_DEVICES': '1',  # Use discrete GPU
        'CUDA_VISIBLE_DEVICES': '1',  # Use discrete GPU
        'HSA_ENABLE_SDMA': '0',      # Disable SDMA for stability
        'HIP_FORCE_DEV_KERNARG': '1', # Force device kernel arguments
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✓ {key} = {value}")
    
    print()

def diagnose_gpu_issue():
    """Diagnose common AMD GPU issues with PyTorch"""
    print("🔍 AMD GPU Diagnostic Tool")
    print("=" * 50)
    
    # Set debugging environment
    set_amd_debug_environment()
    
    # Check PyTorch installation
    print("📊 PyTorch Information:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print()
    
    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm devices available")
        return False
    
    # List all devices
    print("🎮 Available Devices:")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {name}")
    print()
    
    # Try progressive GPU tests
    print("🧪 Progressive GPU Testing:")
    print("-" * 30)
    
    try:
        # Test 1: Basic device selection
        print("Test 1: Device selection...")
        device = torch.device('cuda:1')  # Use discrete GPU
        torch.cuda.set_device(1)
        print(f"✅ Successfully selected: {torch.cuda.get_device_name()}")
        
        # Test 2: Simple tensor creation
        print("Test 2: Simple tensor creation...")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"✅ Tensor created: {x}")
        
        # Test 3: Basic arithmetic
        print("Test 3: Basic arithmetic...")
        y = x + 1
        torch.cuda.synchronize()
        print(f"✅ Arithmetic result: {y}")
        
        # Test 4: Small matrix operation
        print("Test 4: Small matrix operation...")
        a = torch.randn(5, 5, device=device)
        b = torch.randn(5, 5, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"✅ Small matrix multiplication: shape {c.shape}")
        
        # Test 5: Larger operation
        print("Test 5: Larger matrix operation...")
        large_a = torch.randn(100, 100, device=device)
        large_b = torch.randn(100, 100, device=device)
        large_c = torch.matmul(large_a, large_b)
        torch.cuda.synchronize()
        print(f"✅ Large matrix multiplication: shape {large_c.shape}")
        
        print("\n🎉 All GPU tests passed!")
        return True
        
    except RuntimeError as e:
        print(f"\n❌ GPU test failed: {e}")
        print("\n🔧 Troubleshooting recommendations:")
        
        if "invalid device function" in str(e):
            print("• Issue: Invalid device function (kernel execution error)")
            print("• Solutions:")
            print("  1. Update ROCm drivers to latest version")
            print("  2. Check GPU compatibility with PyTorch ROCm")
            print("  3. Try different PyTorch ROCm version")
            print("  4. Restart system and try again")
            
        elif "out of memory" in str(e):
            print("• Issue: GPU out of memory")
            print("• Solutions:")
            print("  1. Close other GPU-using applications")
            print("  2. Reduce tensor sizes")
            print("  3. Clear GPU cache: torch.cuda.empty_cache()")
            
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def suggest_fixes():
    """Suggest potential fixes for common issues"""
    print("\n🛠️  Common Fixes for AMD GPU Issues:")
    print("=" * 50)
    
    print("1. Environment Variables (run before Python):")
    print("   set AMD_SERIALIZE_KERNEL=3")
    print("   set HIP_VISIBLE_DEVICES=1")
    print("   set CUDA_VISIBLE_DEVICES=1")
    print()
    
    print("2. PyTorch ROCm Reinstallation:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1")
    print()
    
    print("3. ROCm Driver Update:")
    print("   • Download latest AMD ROCm drivers")
    print("   • Ensure GPU is supported by ROCm")
    print("   • Check Windows ROCm compatibility")
    print()
    
    print("4. Alternative GPU Selection:")
    print("   • Try using integrated GPU: HIP_VISIBLE_DEVICES=0")
    print("   • Or force CPU only for testing")

if __name__ == "__main__":
    success = diagnose_gpu_issue()
    
    if not success:
        suggest_fixes()
        
        # Offer to create a workaround script
        print("\n💡 Creating GPU workaround script...")
        
        workaround_code = '''# GPU Workaround for AMD Issues
import os
import torch

# Set environment variables
os.environ['AMD_SERIALIZE_KERNEL'] = '3'
os.environ['HIP_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Safe GPU device selection
def get_safe_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(1)  # Discrete GPU
            # Test with small operation
            test_tensor = torch.tensor([1.0], device='cuda:1')
            test_result = test_tensor + 1
            torch.cuda.synchronize()
            return torch.device('cuda:1')
        except:
            print("⚠️  GPU failed, using CPU")
            return torch.device('cpu')
    else:
        return torch.device('cpu')

# Use this in your code:
device = get_safe_device()
print(f"Using device: {device}")
'''
        
        with open('gpu_workaround.py', 'w') as f:
            f.write(workaround_code)
        
        print("✅ Created 'gpu_workaround.py' - import this in your projects")
    
    else:
        print("\n✅ GPU is working correctly!")