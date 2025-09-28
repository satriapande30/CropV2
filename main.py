#!/usr/bin/env python3
"""
Complete Image Auto-Cropping & Augmentation Tool v2.0
Run this file for the full featured application
"""

import sys
import os
from pathlib import Path

# Add source directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

# Check PyQt5 availability
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt
except ImportError:
    print("ERROR: PyQt5 is not installed.")
    print("Please install it with: pip install PyQt5")
    input("Press Enter to exit...")
    sys.exit(1)

# Check OpenCV availability
try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV or NumPy is not installed.")
    print("Please install them with: pip install opencv-python numpy")
    input("Press Enter to exit...")
    sys.exit(1)

def ensure_directories():
    """Ensure necessary directories exist"""
    directories = ["src", "src/core", "src/ui", "config", "logs", "output"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_minimal_config_if_missing():
    """Create minimal config manager if missing"""
    config_file = Path("src/core/config_manager.py")
    
    if not config_file.exists():
        config_code = '''"""
Configuration Manager
"""

import json
import os
from pathlib import Path

class ConfigManager:
    """Simple configuration management"""
    
    def __init__(self, config_file="config/app_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.default_config = {
            "ui_mode": "light",
            "aspect_ratio": "3:4",
            "quality_preset": "medium",
            "last_input_path": "",
            "last_output_path": ""
        }
        
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                config = self.default_config.copy()
                config.update(loaded)
                return config
        except:
            pass
        return self.default_config.copy()
    
    def save_config(self, config=None):
        """Save configuration"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except:
            pass
    
    def get(self, key, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set config value"""
        self.config[key] = value
    
    def update(self, updates):
        """Update config"""
        self.config.update(updates)
'''
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_code)

def create_init_files():
    """Create __init__.py files"""
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py", 
        "src/ui/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(init_file).exists():
            Path(init_file).touch()

def main():
    """Main application entry point"""
    print("Starting Image Auto-Cropping Tool v2.0 - Complete Edition")
    print("=" * 60)
    
    # Setup
    ensure_directories()
    create_init_files()
    create_minimal_config_if_missing()
    
    # Import complete application components
    try:
        from core.config_manager import ConfigManager
        from ui.main_window import MainWindow
        print("✓ All components loaded successfully")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Some components may be missing. Using fallback imports...")
        
        # Try to use the complete main window we just created
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from src.ui.main_window import MainWindow
            from src.core.config_manager import ConfigManager
            print("✓ Fallback imports successful")
            
        except ImportError as e2:
            print(f"✗ Fallback import failed: {e2}")
            print("Please ensure all files are properly created.")
            
            # Show error dialog
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Import Error", 
                f"Failed to import required components:\n\n{str(e2)}\n\n"
                "Please run setup_project.py first or check file structure.")
            return
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Image Auto-Cropping Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Image Processing Tools")
    
    # High DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    try:
        # Create configuration manager
        config_manager = ConfigManager()
        print("✓ Configuration manager initialized")
        
        # Create and show main window
        window = MainWindow(config_manager)
        window.show()
        
        print("✓ Application window opened")
        print("\nApplication ready! The complete image processing tool is now running.")
        print("Features available:")
        print("• Batch image cropping with aspect ratio control")
        print("• Multiple quality settings")
        print("• Real-time preview")
        print("• Progress tracking")
        print("• Professional UI with dark/light themes")
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"✗ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error dialog
        QMessageBox.critical(None, "Application Error", 
            f"Failed to start application:\n\n{str(e)}\n\n"
            "Please check the console output for more details.")

if __name__ == "__main__":
    main()