"""
Environment Validation Script for Proyecto Práctico - Aprendizaje Por Refuerzo
Validates all requirements for the Jupyter notebook including:
- Modelo 1: TensorFlow + Keras-RL2 + Gym
- Modelo 3: PyTorch + Stable-Baselines3 + Gymnasium + AMD GPU
- Modelo 4: PyTorch + Stable-Baselines3 + Gymnasium + AMD GPU
Checks installed packages, GPU availability, and environment compatibility
"""

import sys
import subprocess
from packaging import version
import importlib.util
import os

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def check_package(package_name, required_version=None):
    """Check if a package is installed and optionally verify version"""
    try:
        # Handle special cases - map package names to their import names
        import_name = package_name
        if package_name == 'keras-rl2':
            import_name = 'rl'
        elif package_name == 'gymnasium[atari]':
            import_name = 'gymnasium'
        elif package_name == 'stable-baselines3':
            import_name = 'stable_baselines3'
        elif package_name == 'shimmy[gym-v0.21]':
            import_name = 'shimmy'
        elif package_name == 'pillow':
            import_name = 'PIL'
        elif package_name == 'ipython':
            import_name = 'IPython'
        elif package_name == 'typing-extensions':
            import_name = 'typing_extensions'
        elif package_name == 'ale-py':
            import_name = 'ale_py'
        elif package_name == 'tensorboard':
            import_name = 'tensorboard'
        
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, None
        
        # Get version
        module = importlib.import_module(import_name)
        installed_version = getattr(module, '__version__', 'unknown')
        
        if required_version and installed_version != 'unknown':
            try:
                if version.parse(installed_version) >= version.parse(required_version):
                    return True, installed_version
                else:
                    return True, f"{installed_version} (requires >={required_version})"
            except:
                return True, installed_version
        
        return True, installed_version
    except Exception as e:
        return False, None

def check_gpu_support():
    """Check for GPU support (PyTorch ROCm and TensorFlow CUDA)"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}GPU Detection & Configuration{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Check PyTorch (AMD ROCm)
    try:
        import torch
        print(f"\n{GREEN}✓{RESET} PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            memory_gb = device_props.total_memory / 1024**3
            print(f"{GREEN}✓{RESET} PyTorch CUDA/ROCm GPU detected: {device_name}")
            print(f"  - GPU Memory: {memory_gb:.1f} GB")
            print(f"  - CUDA Version: {torch.version.cuda}")
            
            # Test tensor creation on GPU
            try:
                x = torch.randn(100, 100).cuda()
                print(f"{GREEN}✓{RESET} GPU tensor operations working")
            except Exception as e:
                print(f"{YELLOW}⚠{RESET} GPU tensor test failed: {e}")
        else:
            print(f"{YELLOW}⚠{RESET} PyTorch: No CUDA/ROCm GPU detected")
            print(f"  - For AMD GPU, install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
    except ImportError:
        print(f"{RED}✗{RESET} PyTorch not installed")
        print(f"  - For AMD GPU, run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
    
    # Check TensorFlow (NVIDIA CUDA - for Modelo 1)
    try:
        import tensorflow as tf
        print(f"\n{GREEN}✓{RESET} TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"{GREEN}✓{RESET} TensorFlow GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            
            # Configure GPU memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"{GREEN}✓{RESET} TensorFlow GPU memory growth configured")
            except RuntimeError:
                print(f"{YELLOW}⚠{RESET} GPU memory growth already configured")
        else:
            print(f"{YELLOW}⚠{RESET} TensorFlow: No GPU detected")
    except ImportError:
        print(f"{YELLOW}⚠{RESET} TensorFlow not installed (needed for Modelo 1)")
    
    # Check Keras backend configuration
    try:
        import keras
        print(f"\n{GREEN}✓{RESET} Keras version: {keras.__version__}")
        backend = keras.backend.backend()
        print(f"  - Current backend: {BLUE}{backend}{RESET}")
        print(f"  - To change backend, set environment variable:")
        print(f"    PowerShell: $env:KERAS_BACKEND='torch'  # or 'tensorflow' or 'jax'")
        print(f"    CMD:        set KERAS_BACKEND=torch")
        print(f"    Python:     import os; os.environ['KERAS_BACKEND'] = 'torch'")
    except ImportError:
        print(f"{RED}✗{RESET} Keras not installed")

def check_atari_environments():
    """Check if Atari environments are available"""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Atari Environment Check{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    # Check Gymnasium with ALE
    try:
        import gymnasium as gym
        import ale_py
        print(f"\n{GREEN}✓{RESET} Gymnasium version: {gym.__version__}")
        print(f"{GREEN}✓{RESET} ALE-Py version: {ale_py.__version__}")
        
        # Test SpaceInvaders environment creation
        try:
            env = gym.make('SpaceInvaders-v0', render_mode=None)
            obs, info = env.reset()
            print(f"{GREEN}✓{RESET} SpaceInvaders-v0 environment working")
            print(f"  - Observation space: {env.observation_space}")
            print(f"  - Action space: {env.action_space}")
            env.close()
        except Exception as e:
            print(f"{RED}✗{RESET} SpaceInvaders-v0 failed: {e}")
            
        # Check available ROMs
        try:
            from ale_py import roms
            available_roms = [r for r in dir(roms) if not r.startswith('_')]
            print(f"  - Available ROMs: {len(available_roms)}")
            if 'SpaceInvaders' in available_roms:
                print(f"{GREEN}✓{RESET} SpaceInvaders ROM available")
            else:
                print(f"{YELLOW}⚠{RESET} SpaceInvaders ROM not found in available ROMs")
        except Exception as e:
            print(f"{YELLOW}⚠{RESET} Could not list ROMs: {e}")
            
    except ImportError as e:
        print(f"{RED}✗{RESET} Atari environment setup incomplete: {e}")
        
    # Check old Gym (Modelo 1)
    try:
        # Try to import legacy gym separately from gymnasium
        import sys
        if 'gym' in sys.modules:
            # If already imported as part of gymnasium, remove it
            old_gym = sys.modules['gym']
        
        # Try importing the actual legacy gym package
        import gym as legacy_gym
        if hasattr(legacy_gym, '__version__'):
            gym_version = legacy_gym.__version__
        else:
            # Fallback for newer gymnasium that shadows 'gym'
            gym_version = "1.2.3 (gymnasium)"
            
        print(f"\n{GREEN}✓{RESET} Legacy Gym version: {gym_version}")
        
        # Test compatibility patch
        import numpy as np
        np.bool = bool  # Compatibility patch
        print(f"{GREEN}✓{RESET} NumPy bool compatibility patch applied")
        
        try:
            env = legacy_gym.make('SpaceInvaders-v0')
            env.reset()
            print(f"{GREEN}✓{RESET} Legacy Gym SpaceInvaders-v0 working")
            env.close()
        except Exception as e:
            print(f"{RED}✗{RESET} Legacy Gym SpaceInvaders-v0 failed: {e}")
            
    except ImportError:
        print(f"{YELLOW}⚠{RESET} Legacy Gym not installed (needed for Modelo 1)")
        print(f"  - Install with: py -m pip install gym==0.17.3")

def check_stable_baselines3():
    """Check Stable-Baselines3 functionality"""
    print(f"\n{MAGENTA}{'='*60}{RESET}")
    print(f"{MAGENTA}Stable-Baselines3 Check{RESET}")
    print(f"{MAGENTA}{'='*60}{RESET}")
    
    try:
        import stable_baselines3 as sb3
        print(f"\n{GREEN}✓{RESET} Stable-Baselines3 version: {sb3.__version__}")
        
        # Check key algorithms
        from stable_baselines3 import DQN, PPO
        print(f"{GREEN}✓{RESET} DQN algorithm available")
        print(f"{GREEN}✓{RESET} PPO algorithm available")
        
        # Check environment utilities
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
        print(f"{GREEN}✓{RESET} Atari environment utilities available")
        
        # Test environment creation
        try:
            env = make_atari_env('SpaceInvaders-v0', n_envs=1, seed=123)
            env = VecFrameStack(env, n_stack=4)
            print(f"{GREEN}✓{RESET} Stable-Baselines3 Atari environment creation working")
            env.close()
        except Exception as e:
            print(f"{RED}✗{RESET} SB3 environment creation failed: {e}")
        
        # Check callbacks
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        print(f"{GREEN}✓{RESET} Training callbacks available")
        
        # Check evaluation
        from stable_baselines3.common.evaluation import evaluate_policy
        print(f"{GREEN}✓{RESET} Policy evaluation utilities available")
        
    except ImportError as e:
        print(f"{RED}✗{RESET} Stable-Baselines3 not properly installed: {e}")
def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Proyecto Práctico - Environment Validation{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    
    # Define required packages for all models in the notebook
    packages = {
        'Core System Packages': [
            ('setuptools', '65.5.0'),
            ('wheel', None),
            ('pip', None),
            ('typing-extensions', '4.5.0'),
        ],
        'Scientific Computing': [
            ('numpy', '1.23.5'),
            ('pandas', '1.5.3'),
            ('matplotlib', '3.7.5'),
            ('pillow', None),
            ('openpyxl', None),
        ],
        'Deep Learning Frameworks': [
            ('torch', None),           # PyTorch for AMD (Modelo 3 & 4)
            ('tensorflow', '2.12.1'),  # TensorFlow for Modelo 1
            ('keras', None),           # Keras 3.x supports multiple backends
        ],
        'RL Environments': [
            ('gymnasium[atari]', None),        # Modern Atari for Modelo 3 & 4
            ('gym', '0.17.3'),                # Legacy Gym for Modelo 1
            ('ale-py', None),                  # Atari Learning Environment
            ('shimmy[gym-v0.21]', None),       # Compatibility wrapper
        ],
        'RL Algorithms & Libraries': [
            ('keras-rl2', '1.0.5'),           # For Modelo 1 (TensorFlow/Keras)
            ('stable-baselines3', None),       # For Modelo 3 & 4 (PyTorch)
        ],
        'Visualization & Monitoring': [
            ('tensorboard', None),             # TensorBoard for all models
            ('imageio', None),                 # Video recording in SB3
        ],
        'Jupyter/Notebook Environment': [
            ('ipykernel', None),
            ('notebook', None),
            ('ipython', '8.12.3'),
            ('jupyterlab', None),
        ],
    }
    
    installed = []
    missing = []
    version_mismatch = []
    
    for category, pkg_list in packages.items():
        print(f"\n{BLUE}{category}:{RESET}")
        for package, req_version in pkg_list:
            is_installed, installed_version = check_package(package, req_version)
            
            if is_installed:
                status = f"{GREEN}✓{RESET}"
                version_str = f"({installed_version})" if installed_version else ""
                print(f"  {status} {package:25} {version_str}")
                installed.append(package)
                
                # Check version mismatch
                if req_version and installed_version and installed_version != req_version:
                    if "requires" not in str(installed_version):
                        version_mismatch.append((package, installed_version, req_version))
            else:
                status = f"{RED}✗{RESET}"
                req_str = f"(requires {req_version})" if req_version else ""
                print(f"  {status} {package:25} {RED}NOT INSTALLED{RESET} {req_str}")
                missing.append((package, req_version))
    
    # Specialized checks
    check_gpu_support()
    check_atari_environments()
    check_stable_baselines3()
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Installation Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}Installed:{RESET} {len(installed)} packages")
    print(f"{RED}Missing:{RESET} {len(missing)} packages")
    
    if missing:
        print(f"\n{YELLOW}Missing Packages - Installation Commands:{RESET}")
        
        # Group by installation method
        regular_packages = []
        special_packages = []
        
        for package, req_version in missing:
            if package in ['gymnasium[atari]', 'shimmy[gym-v0.21]']:
                special_packages.append((package, req_version))
            else:
                regular_packages.append((package, req_version))
        
        if regular_packages:
            print(f"\n{CYAN}# Standard packages:{RESET}")
            for package, req_version in regular_packages:
                if req_version:
                    print(f"py -m pip install {package}=={req_version}")
                else:
                    print(f"py -m pip install {package}")
        
        if special_packages:
            print(f"\n{CYAN}# Special packages:{RESET}")
            for package, req_version in special_packages:
                if package == 'gymnasium[atari]':
                    print(f"py -m pip install 'gymnasium[atari]'")
                elif package == 'shimmy[gym-v0.21]':
                    print(f"py -m pip install 'shimmy[gym-v0.21]'")
        
        # PyTorch AMD installation if missing
        if any(pkg == 'torch' for pkg, _ in missing):
            print(f"\n{CYAN}# PyTorch with AMD ROCm support:{RESET}")
            print(f"py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0")
        
        # TensorFlow if missing
        if any(pkg == 'tensorflow' for pkg, _ in missing):
            print(f"\n{CYAN}# TensorFlow (for Modelo 1):{RESET}")
            print(f"py -m pip install tensorflow==2.12.1")
        
        # Keras if missing
        if any(pkg == 'keras' for pkg, _ in missing):
            print(f"\n{CYAN}# Keras 3.x (multi-backend support):{RESET}")
            print(f"py -m pip install keras>=3.0.0")
        
        # Stable-Baselines3 if missing
        if any(pkg == 'stable-baselines3' for pkg, _ in missing):
            print(f"\n{CYAN}# Stable-Baselines3 (for Modelo 3 & 4):{RESET}")
            print(f"py -m pip install stable-baselines3[extra]")
        
        # Keras-RL2 if missing
        if any(pkg == 'keras-rl2' for pkg, _ in missing):
            print(f"\n{CYAN}# Keras-RL2 (for Modelo 1):{RESET}")
            print(f"py -m pip install keras-rl2==1.0.5")
    
    if version_mismatch:
        print(f"\n{YELLOW}Version Recommendations:{RESET}")
        for package, installed_ver, required_ver in version_mismatch:
            print(f"  {package}: installed {installed_ver}, recommended {required_ver}")
    
    # Configuration guidance
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Configuration Guidance{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    
    print(f"\n{MAGENTA}For Modelo 1 (TensorFlow + Keras-RL2 + Legacy Gym):{RESET}")
    print(f"  - Use TensorFlow backend")
    print(f"  - Apply numpy compatibility: np.bool = bool")
    print(f"  - Use gym v0.17.3 with SpaceInvaders-v0")
    
    print(f"\n{MAGENTA}For Modelo 3 & 4 (PyTorch + Stable-Baselines3 + Gymnasium):{RESET}")
    print(f"  - Set Keras backend: $env:KERAS_BACKEND='torch'")
    print(f"  - Use Gymnasium with ALE: SpaceInvaders-v0")
    print(f"  - Enable AMD GPU: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
    
    print(f"\n{BLUE}Environment Variables:{RESET}")
    print(f"  PowerShell: $env:KERAS_BACKEND='torch'")
    print(f"  Python:     import os; os.environ['KERAS_BACKEND'] = 'torch'")
    
    print(f"\n{GREEN}Next Steps:{RESET}")
    print(f"  1. Install missing packages using commands above")
    print(f"  2. Restart Jupyter kernel after installations")
    print(f"  3. Run this validation script again to verify")
    print(f"  4. Open Proyecto_práctico.ipynb and test environment setup cells")
    
    print()

if __name__ == "__main__":
    main()
