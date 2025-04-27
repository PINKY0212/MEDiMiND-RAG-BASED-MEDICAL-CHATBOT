import torch
import sys
import platform
import numpy as np
from typing import Dict, Any

def get_system_info() -> Dict[str, Any]:
    """Get system information including Python version, OS, and PyTorch details."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else "N/A",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

def check_cuda_compatibility() -> Dict[str, Any]:
    """Check CUDA compatibility and return detailed information."""
    info = get_system_info()
    
    # Check NumPy version compatibility
    numpy_major_version = int(info["numpy_version"].split('.')[0])
    if numpy_major_version >= 2:
        info["numpy_warning"] = (
            "Warning: You are using NumPy 2.x which may cause compatibility issues with some PyTorch modules. "
            "Consider downgrading to NumPy 1.x with: pip install 'numpy<2'"
        )
    else:
        info["numpy_warning"] = None
    
    # Check if CUDA is available
    if not info["cuda_available"]:
        info["compatibility_status"] = "CUDA is not available on this system"
        return info
    
    # Check if PyTorch was built with CUDA support
    if not torch.cuda.is_available():
        info["compatibility_status"] = "PyTorch was not built with CUDA support"
        return info
    
    # Check if there are any CUDA devices
    if info["device_count"] == 0:
        info["compatibility_status"] = "No CUDA devices found"
        return info
    
    # If all checks pass
    info["compatibility_status"] = "CUDA is available and working properly"
    return info

def print_cuda_info() -> None:
    """Print detailed CUDA and system information."""
    info = check_cuda_compatibility()
    
    print("\n=== System Information ===")
    print(f"Python Version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"NumPy Version: {info['numpy_version']}")
    
    if info.get('numpy_warning'):
        print("\n=== Warning ===")
        print(info['numpy_warning'])
    
    print("\n=== CUDA Information ===")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"CuDNN Version: {info['cudnn_version']}")
        print(f"Number of CUDA Devices: {info['device_count']}")
        print(f"Current CUDA Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
    
    print("\n=== Status ===")
    print(f"Compatibility Status: {info['compatibility_status']}")

if __name__ == "__main__":
    print_cuda_info() 