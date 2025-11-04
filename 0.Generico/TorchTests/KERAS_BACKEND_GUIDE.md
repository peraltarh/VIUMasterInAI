# Keras Backend Configuration Guide

## ✅ **KERAS_BACKEND=torch is now SET PERMANENTLY!**

The following command was executed successfully:
```powershell
setx KERAS_BACKEND "torch"
```

## 🎯 **What This Means:**

### **✅ Permanent Changes:**
- Keras will ALWAYS use PyTorch as its backend
- Works across ALL applications, terminals, and reboots
- Your RX 9070 XT will be used for Keras models automatically
- Consistent behavior in all your AI projects

### **🔄 How to Verify:**
```python
import keras
print("Keras Backend:", keras.backend.backend())
# Should show: torch
```

## 🚀 **Alternative Methods for Setting KERAS_BACKEND:**

### **Method 1: Windows Environment Variables (DONE ✅)**
```powershell
# Already executed - permanent across all sessions
setx KERAS_BACKEND "torch"
```

### **Method 2: In Python Code (Session-based)**
```python
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras  # Must import AFTER setting environment
```

### **Method 3: PowerShell Profile**
```powershell
# Add to your PowerShell profile
$env:KERAS_BACKEND = "torch"
```

### **Method 4: .env File**
```bash
# Create .env file in your project:
KERAS_BACKEND=torch
CUDA_VISIBLE_DEVICES=1
HIP_VISIBLE_DEVICES=1
```

## 🧪 **Testing Your Setup:**

### **Complete Test Script:**
```python
import os
import keras
import torch

print("🧪 Testing Keras + PyTorch + GPU Setup")
print("=" * 50)

# Check backend
print(f"✅ Keras Backend: {keras.backend.backend()}")

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
else:
    print("❌ No GPU available")

# Create simple model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

print(f"✅ Keras model created: {model.count_params()} parameters")
```

## 🎉 **Benefits of PyTorch Backend:**

1. **🔥 GPU Acceleration**: Uses your RX 9070 XT automatically
2. **🚀 Performance**: Often faster than TensorFlow backend
3. **🔗 Consistency**: Same underlying engine as pure PyTorch
4. **📊 Memory**: Better memory management
5. **🛠️ Debugging**: Easier to debug and profile

## 🔄 **Switching Backends (if needed):**

### **Back to TensorFlow:**
```powershell
setx KERAS_BACKEND "tensorflow"
```

### **To JAX:**
```powershell
setx KERAS_BACKEND "jax"
```

### **Check Available Backends:**
```python
import keras
print("Available backends:", keras.backend.list_backend_names())
```

## 🛡️ **Troubleshooting:**

### **If Keras doesn't use PyTorch:**
1. **Restart terminal/VS Code** (environment variables need refresh)
2. **Check environment**: `echo $env:KERAS_BACKEND` (PowerShell)
3. **Set manually**: `$env:KERAS_BACKEND="torch"` before importing

### **If GPU not detected:**
1. **Run setup script**: `python setup_gpu.py`
2. **Check PyTorch**: `torch.cuda.is_available()`
3. **Verify environment**: GPU variables must be set

## 🎯 **Your Complete Setup:**

```
✅ PyTorch 2.8.0a0 with ROCm 6.4.5
✅ AMD Radeon RX 9070 XT as default GPU
✅ Keras with PyTorch backend
✅ All environment variables permanent
✅ Ready for AI/ML projects!
```

**🚀 You're all set! Keras will now use PyTorch + your GPU automatically!**