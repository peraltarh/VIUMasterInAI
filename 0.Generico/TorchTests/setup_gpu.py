#!/usr/bin/env python3
"""
GPU Environment Setup and Test Script
Run this when you start working to ensure your GPU is configured correctly.
"""

import os
import torch

def setup_gpu_environment():
    """Set up environment variables and test GPU"""
    print("🔧 Setting up GPU Environment")
    print("=" * 50)
    
    # Set environment variables for current session
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['HIP_VISIBLE_DEVICES'] = '1'
    os.environ['KERAS_BACKEND'] = 'torch'
    
    print("✅ Environment variables set:")
    print(f"   CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"   KERAS_BACKEND = {os.environ.get('KERAS_BACKEND')}")
    print()
    
    # Test PyTorch GPU access
    print("🧪 Testing PyTorch GPU Access")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA/ROCm available")
        return False
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ GPU Available: {torch.cuda.is_available()}")
    print(f"📊 Visible devices: {device_count}")
    print(f"🎯 Current device: {current_device}")
    print(f"🔥 GPU Name: {device_name}")
    
    # Quick computation test
    try:
        print("\n🚀 Testing GPU Operations")
        print("-" * 30)
        
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        
        print(f"✅ Matrix multiplication successful")
        print(f"📍 Computation device: {z.device}")
        print(f"🎮 GPU used: {torch.cuda.get_device_name(z.device)}")
        print(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        return False

def test_keras_backend():
    """Test Keras with PyTorch backend"""
    try:
        print("\n🧪 Testing Keras with PyTorch Backend")
        print("=" * 50)
        
        import keras
        
        # Check backend
        backend = keras.backend.backend()
        print(f"✅ Keras Backend: {backend}")
        
        if backend.lower() != 'torch':
            print(f"⚠️  Warning: Expected 'torch', got '{backend}'")
            return False
            
        # Test simple model creation
        from keras.models import Sequential
        from keras.layers import Dense, Input
        
        model = Sequential([
            Input(shape=(10,)),  # Modern way - separate Input layer
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        print(f"✅ Keras model created successfully")
        print(f"📊 Model parameters: {model.count_params():,}")
        
        # Test model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print(f"✅ Model compiled successfully")
        
        return True
        
    except ImportError:
        print("❌ Keras not installed. Run: pip install keras")
        return False
    except Exception as e:
        print(f"❌ Keras test failed: {e}")
        return False

def main():
    print("🎯 AMD RX 9070 XT Setup Script")
    print("=" * 50)
    
    pytorch_success = setup_gpu_environment()
    keras_success = test_keras_backend()
    
    if pytorch_success and keras_success:
        print("\n🎉 SUCCESS! Your RX 9070 XT is ready for PyTorch AND Keras!")
        print("=" * 50)
        print("📝 Your GPU is now configured as the default device.")
        print("📝 You can use torch.device('cuda') in your code.")
        print("📝 Keras will use PyTorch backend with GPU acceleration.")
        print("📝 No need to specify cuda:1 anymore!")
    elif pytorch_success:
        print("\n✅ PyTorch GPU setup successful!")
        print("⚠️  Keras setup had issues - check installation.")
    else:
        print("\n❌ Setup failed. Please check your installation.")
    
    return pytorch_success and keras_success

if __name__ == "__main__":
    main()