#!/usr/bin/env python3
"""
RT-DETR Setup Script (Updated for PyTorch 2.9+)
Sets up RT-DETR for vehicle detection with existing PyTorch installation.
"""

import os
import subprocess
import sys


def run_command(command, description, check=True):
    """Execute a shell command with error handling."""
    print(f"\n[INFO] {description}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if check and result.returncode != 0:
        print(f"[ERROR] {description} failed:")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def check_gpu():
    """Check if NVIDIA GPU is available."""
    print("\n" + "="*60)
    print("Checking GPU availability...")
    print("="*60)
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("[WARNING] No NVIDIA GPU detected or nvidia-smi not available")
        print("RT-DETR will run on CPU (slower)")


def verify_pytorch():
    """Verify PyTorch installation."""
    print("\n" + "="*60)
    print("Verifying PyTorch installation...")
    print("="*60)
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} found")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("[ERROR] PyTorch not found!")
        print("Please install PyTorch first: pip install torch torchvision")
        return False


def clone_rtdetr():
    """Clone RT-DETR repository."""
    print("\n" + "="*60)
    print("Setting up RT-DETR repository...")
    print("="*60)
    
    if os.path.exists("RT-DETR"):
        print("[INFO] RT-DETR directory already exists")
        response = input("Remove and re-clone? (y/n): ").strip().lower()
        if response == 'y':
            run_command("rm -rf RT-DETR", "Removing existing RT-DETR directory")
        else:
            print("[INFO] Using existing RT-DETR directory")
            return True
    
    success = run_command(
        "git clone https://github.com/lyuwenyu/RT-DETR.git",
        "Cloning RT-DETR repository"
    )
    
    if not success:
        return False
    
    if not os.path.exists("RT-DETR/rtdetr_pytorch"):
        print("[ERROR] RT-DETR PyTorch directory not found!")
        return False
    
    print("[OK] RT-DETR cloned successfully")
    return True


def install_rtdetr_dependencies():
    """Install RT-DETR dependencies."""
    print("\n" + "="*60)
    print("Installing RT-DETR dependencies...")
    print("="*60)
    
    # Change to RT-DETR directory
    original_dir = os.getcwd()
    os.chdir("RT-DETR/rtdetr_pytorch")
    
    try:
        # Install additional dependencies that might be needed
        deps = [
            "onnx",
            "onnxruntime", 
            "onnx-simplifier",
        ]
        
        for dep in deps:
            run_command(
                f"pip install -q {dep}",
                f"Installing {dep}",
                check=False  # Don't fail if optional deps fail
            )
        
        print("[OK] Dependencies installed")
        return True
        
    finally:
        os.chdir(original_dir)


def download_model():
    """Download pretrained RT-DETR model."""
    print("\n" + "="*60)
    print("Downloading pretrained model...")
    print("="*60)
    
    checkpoint_dir = "RT-DETR/rtdetr_pytorch/checkpoints"
    checkpoint_file = f"{checkpoint_dir}/rtdetr_r50vd_coco.pth"
    
    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if os.path.exists(checkpoint_file):
        print(f"[INFO] Model already exists: {checkpoint_file}")
        return True
    
    # Download model
    model_url = "https://github.com/lyuwenyu/storage/releases/download/v0.0.1/rtdetr_r50vd_coco.pth"
    
    success = run_command(
        f"wget -q --show-progress -O {checkpoint_file} {model_url}",
        "Downloading RT-DETR model weights (~100MB)"
    )
    
    if success and os.path.exists(checkpoint_file):
        print(f"[OK] Model downloaded: {checkpoint_file}")
        return True
    else:
        print("[ERROR] Model download failed!")
        print(f"Please download manually from: {model_url}")
        return False


def verify_setup():
    """Verify RT-DETR setup."""
    print("\n" + "="*60)
    print("Verifying RT-DETR setup...")
    print("="*60)
    
    checks = {
        "RT-DETR directory": "RT-DETR/rtdetr_pytorch",
        "Config file": "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        "Checkpoint": "RT-DETR/rtdetr_pytorch/checkpoints/rtdetr_r50vd_coco.pth",
        "Source code": "RT-DETR/rtdetr_pytorch/src/core.py",
    }
    
    all_good = True
    for name, path in checks.items():
        if os.path.exists(path):
            print(f"[OK] {name}")
        else:
            print(f"[ERROR] {name} - Not found: {path}")
            all_good = False
    
    return all_good


def create_test_script():
    """Create a simple test script."""
    print("\n" + "="*60)
    print("Creating test script...")
    print("="*60)
    
    test_script = """#!/usr/bin/env python3
'''Quick test to verify RT-DETR setup'''
import sys
import os

# Add RT-DETR to path
sys.path.insert(0, 'RT-DETR/rtdetr_pytorch')

try:
    from src.core import YAMLConfig
    print("[OK] RT-DETR imports successful!")
    
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    
    config_path = 'RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
    checkpoint_path = 'RT-DETR/rtdetr_pytorch/checkpoints/rtdetr_r50vd_coco.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"[OK] Model checkpoint found")
        print("\\n🎉 RT-DETR setup is complete and working!")
        print("\\nTo run vehicle detection:")
        print("  python vehicle_detection_simple.py IMG_20250813_161947.jpg")
    else:
        print("[ERROR] Model checkpoint not found")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] Error: {e}")
    sys.exit(1)
"""
    
    with open("test_rtdetr.py", "w") as f:
        f.write(test_script)
    
    os.chmod("test_rtdetr.py", 0o755)
    print("[OK] Created test_rtdetr.py")


def main():
    """Main setup function."""
    print("="*60)
    print("RT-DETR Setup for Vehicle Detection")
    print("="*60)
    
    # Step 1: Check GPU
    check_gpu()
    
    # Step 2: Verify PyTorch
    if not verify_pytorch():
        print("\n[ERROR] Setup failed: PyTorch not found")
        sys.exit(1)
    
    # Step 3: Clone RT-DETR
    if not clone_rtdetr():
        print("\n[ERROR] Setup failed: Could not clone RT-DETR")
        sys.exit(1)
    
    # Step 4: Install dependencies
    if not install_rtdetr_dependencies():
        print("\n[WARN]  Some dependencies failed but continuing...")
    
    # Step 5: Download model
    if not download_model():
        print("\n[ERROR] Setup failed: Could not download model")
        sys.exit(1)
    
    # Step 6: Verify everything
    if not verify_setup():
        print("\n[ERROR] Setup verification failed")
        sys.exit(1)
    
    # Step 7: Create test script
    create_test_script()
    
    # Success!
    print("\n" + "="*60)
    print("[OK] RT-DETR Setup Complete!")
    print("="*60)
    print("\nTest the setup:")
    print("  python test_rtdetr.py")
    print("\nRun vehicle detection:")
    print("  python vehicle_detection_simple.py IMG_20250813_161947.jpg")
    print("="*60)


if __name__ == "__main__":
    main()
