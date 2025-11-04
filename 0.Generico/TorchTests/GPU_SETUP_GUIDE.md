# AMD GPU Environment Configuration Guide

## Current Status: ✅ PERMANENT SETTINGS APPLIED

The following environment variables have been set permanently:
- `CUDA_VISIBLE_DEVICES=1`
- `HIP_VISIBLE_DEVICES=1`

## What This Means:

### ✅ **Permanent Changes:**
- Your RX 9070 XT (device 1) will be the default GPU for ALL applications
- Works across ALL terminals, applications, and reboots
- PyTorch will automatically use your discrete GPU
- No need to specify `cuda:1` in code anymore

### 🔄 **How to Verify:**
1. **Close and reopen PowerShell/Terminal**
2. **Run:** `py -3.12 -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"`
3. **Should show:** `GPU: AMD Radeon RX 9070 XT`

## Alternative Methods (for reference):

### **Method 1: Windows System Environment (GUI)**
1. Press `Win + R`, type `sysdm.cpl`
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `CUDA_VISIBLE_DEVICES`
5. Variable value: `1`
6. Repeat for `HIP_VISIBLE_DEVICES`

### **Method 2: PowerShell Profile (Session-based)**
```powershell
# Edit your PowerShell profile
notepad $PROFILE

# Add these lines:
$env:CUDA_VISIBLE_DEVICES='1'
$env:HIP_VISIBLE_DEVICES='1'
```

### **Method 3: Batch Script**
Create `set_gpu.bat`:
```batch
@echo off
setx CUDA_VISIBLE_DEVICES "1"
setx HIP_VISIBLE_DEVICES "1"
echo GPU environment configured!
pause
```

## Testing Your Configuration:

### **Simple Test:**
```python
import torch
print(f"Default GPU: {torch.cuda.get_device_name(0)}")
# Should show: AMD Radeon RX 9070 XT
```

### **Complete Test:**
```python
import torch
device = torch.device('cuda')  # No need to specify :1 anymore
x = torch.randn(100, 100, device=device)
print(f"Tensor device: {x.device}")
print(f"GPU name: {torch.cuda.get_device_name(x.device)}")
```

## Reverting Changes (if needed):

### **To Use Integrated GPU Again:**
```powershell
setx CUDA_VISIBLE_DEVICES "0"
setx HIP_VISIBLE_DEVICES "0"
```

### **To Use Both GPUs:**
```powershell
setx CUDA_VISIBLE_DEVICES "0,1"
setx HIP_VISIBLE_DEVICES "0,1"
```

### **To Remove Environment Variables:**
```powershell
reg delete "HKEY_CURRENT_USER\Environment" /v CUDA_VISIBLE_DEVICES /f
reg delete "HKEY_CURRENT_USER\Environment" /v HIP_VISIBLE_DEVICES /f
```

## 🔧 **Troubleshooting: If Environment Variables Don't Load**

If you're in a new session and environment variables aren't working, use this startup script:

### **Quick Setup Script (Recommended):**
```bash
# Run this command to configure and test your GPU:
C:\Users\Rodri\AppData\Local\Programs\Python\Python312\python.exe "c:\Users\Rodri\Documents\VS Workspace\VIUMasterInAI\0.Generico\TorchTests\setup_gpu.py"
```

This script will:
- ✅ Set environment variables for current session
- ✅ Test PyTorch GPU access  
- ✅ Confirm RX 9070 XT is working
- ✅ Show memory usage and device info

### **PowerShell Profile Option:**
```powershell
# Add this to your PowerShell profile for automatic loading:
Add-Content $PROFILE '. "c:\Users\Rodri\Documents\VS Workspace\VIUMasterInAI\0.Generico\TorchTests\load_gpu_env.ps1"'
```

## 🎉 **You're All Set!**

Your AMD Radeon RX 9070 XT is now the default GPU for:
- PyTorch
- TensorFlow (if installed)
- Any CUDA/ROCm application
- All future sessions

**Next time you run PyTorch code, it will automatically use your powerful discrete GPU!**