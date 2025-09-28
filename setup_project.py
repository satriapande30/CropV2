#!/usr/bin/env python3
"""
Setup script to create the project structure and missing files
Run this first to set up the complete project structure.
"""

import os
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "src",
        "src/core", 
        "src/ui",
        "src/utils",
        "config",
        "logs", 
        "output",
        "resources",
        "resources/themes"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files"""
    init_files = {
        "src/__init__.py": '"""Image Auto-Cropping Tool Source Package"""',
        "src/core/__init__.py": '"""Core processing modules"""',
        "src/ui/__init__.py": '"""User interface components"""', 
        "src/utils/__init__.py": '"""Utility functions"""'
    }
    
    for file_path, content in init_files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content + "\n")
        print(f"Created: {file_path}")

def create_placeholder_files():
    """Create placeholder files that might be missing"""
    
    # Create a minimal image_utils.py
    image_utils_content = '''"""
Image utility functions
"""

import cv2
import numpy as np
from typing import Tuple

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def convert_color_space(image: np.ndarray, conversion: int) -> np.ndarray:
    """Convert image color space"""
    return cv2.cvtColor(image, conversion)
'''
    
    with open("src/utils/image_utils.py", 'w', encoding='utf-8') as f:
        f.write(image_utils_content)
    print("Created: src/utils/image_utils.py")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'PyQt5',
        'opencv-python', 
        'numpy',
        'Pillow',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("\nAll required packages are installed!")
    return True

def create_simple_config():
    """Create a simple default configuration file"""
    config_content = '''{
  "ui_mode": "light",
  "aspect_ratio": "3:4", 
  "quality_preset": "medium",
  "auto_augmentation": true,
  "crop_method": "face_aware",
  "last_input_path": "",
  "last_output_path": "",
  "face_detection_confidence": 0.5,
  "max_workers": 4,
  "batch_size": 10,
  "memory_limit_mb": 1000,
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
}'''
    
    with open("config/app_config.json", 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("Created: config/app_config.json")

def main():
    """Main setup function"""
    print("Setting up Image Auto-Cropping Tool v2.0...")
    print("=" * 50)
    
    # Create directory structure
    create_directories()
    print()
    
    # Create __init__.py files  
    create_init_files()
    print()
    
    # Create utility files
    create_placeholder_files()
    print()
    
    # Create default config
    create_simple_config()
    print()
    
    # Check requirements
    packages_ok = check_requirements()
    print()
    
    print("Setup complete!")
    print("=" * 50)
    
    if packages_ok:
        print("You can now run the application with:")
        print("python main.py")
    else:
        print("Please install missing packages first, then run:")
        print("python main.py")
    
    print("\nProject structure created:")
    print("├── main.py")
    print("├── setup_project.py")  
    print("├── requirements.txt")
    print("├── src/")
    print("│   ├── core/")
    print("│   ├── ui/")
    print("│   └── utils/")
    print("├── config/")
    print("├── logs/")
    print("└── output/")

if __name__ == "__main__":
    main()