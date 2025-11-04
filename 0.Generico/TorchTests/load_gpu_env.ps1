# GPU Environment Loader for PowerShell
# Add this to your PowerShell profile for automatic GPU setup

Write-Host "🎯 Loading AMD GPU Environment..." -ForegroundColor Cyan

# Set environment variables
$env:CUDA_VISIBLE_DEVICES = "1"
$env:HIP_VISIBLE_DEVICES = "1"
$env:KERAS_BACKEND = "torch"

Write-Host "✅ GPU Environment Variables Set:" -ForegroundColor Green
Write-Host "   CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Yellow
Write-Host "   HIP_VISIBLE_DEVICES = $env:HIP_VISIBLE_DEVICES" -ForegroundColor Yellow
Write-Host "   KERAS_BACKEND = $env:KERAS_BACKEND" -ForegroundColor Yellow

# Optional: Run PyTorch validation
function Test-PyTorchGPU {
    Write-Host "🧪 Testing PyTorch GPU..." -ForegroundColor Cyan
    python -c "
import torch
if torch.cuda.is_available():
    print('✅ GPU Available:', torch.cuda.get_device_name())
    x = torch.randn(100, 100, device='cuda')
    print('✅ GPU operations working!')
else:
    print('❌ No GPU available')
"
}

# Create alias for easy GPU testing
Set-Alias -Name "test-gpu" -Value Test-PyTorchGPU

Write-Host "🚀 Ready! Use 'test-gpu' to verify PyTorch GPU access" -ForegroundColor Green