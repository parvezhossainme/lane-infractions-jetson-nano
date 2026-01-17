#!/usr/bin/env python3
"""
RT-DETR Setup Checker
Verifies that RT-DETR is properly installed and configured.
"""

import os
import sys


def check_setup():
    """Check RT-DETR setup status."""
    
    print("=" * 60)
    print("RT-DETR Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # 1. Check virtual environment
    print("\n1. Virtual Environment:")
    if os.path.exists('venv'):
        print("   ✅ Virtual environment exists")
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if in_venv:
            print(f"   ✅ Currently activated: {sys.prefix}")
        else:
            print(f"   ⚠️  Not activated. Run: source venv/bin/activate")
    else:
        print("   ❌ Virtual environment not found")
        all_good = False
    
    # 2. Check PyTorch
    print("\n2. PyTorch:")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ⚠️  CUDA not available (CPU mode)")
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("      Run: pip install -r requirements.txt")
        all_good = False
    
    # 3. Check OpenCV
    print("\n3. OpenCV:")
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__} installed")
    except ImportError:
        print("   ❌ OpenCV not installed")
        all_good = False
    
    # 4. Check RT-DETR repository
    print("\n4. RT-DETR Repository:")
    if os.path.exists('RT-DETR'):
        print("   ✅ RT-DETR repository cloned")
        
        # Check specific directories
        if os.path.exists('RT-DETR/rtdetr_pytorch'):
            print("   ✅ PyTorch implementation found")
        else:
            print("   ❌ RT-DETR/rtdetr_pytorch not found")
            all_good = False
        
        # Check config
        if os.path.exists('RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'):
            print("   ✅ Config file exists")
        else:
            print("   ❌ Config file not found")
            all_good = False
        
        # Check checkpoint
        if os.path.exists('RT-DETR/rtdetr_pytorch/checkpoints/rtdetr_r50vd_coco.pth'):
            print("   ✅ Pretrained model downloaded")
        else:
            print("   ⚠️  Pretrained model not found")
            print("      This will be downloaded when you run setup_rtdetr.py")
    else:
        print("   ❌ RT-DETR repository not cloned")
        print("      Run: python setup_rtdetr.py")
        all_good = False
    
    # 5. Check image files
    print("\n5. Test Images:")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir('.') if f.lower().endswith(ext)])
    
    if image_files:
        print(f"   ✅ Found {len(image_files)} image(s):")
        for img in image_files[:5]:  # Show first 5
            print(f"      - {img}")
        if len(image_files) > 5:
            print(f"      ... and {len(image_files) - 5} more")
    else:
        print("   ⚠️  No image files found in current directory")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to go.")
        print("\nQuick Start:")
        print("  Lane Detection:")
        print("    python lane_detection_save.py IMG_20250813_161947.jpg")
        print("\n  Vehicle Detection (requires RT-DETR setup):")
        print("    cd RT-DETR/rtdetr_pytorch")
        print("    python ../../vehicle_detection.py ../../IMG_20250813_161947.jpg")
    else:
        print("❌ Some issues found. Please fix them first.")
        print("\nTo set up everything:")
        print("  1. Activate venv: source venv/bin/activate")
        print("  2. Setup RT-DETR: python setup_rtdetr.py")
    print("=" * 60)
    
    return all_good


if __name__ == "__main__":
    check_setup()
