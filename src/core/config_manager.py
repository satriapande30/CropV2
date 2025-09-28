"""
Minimal Configuration Manager
"""

import json
import os
from pathlib import Path

class ConfigManager:
    """Basic configuration management"""
    
    def __init__(self, config_file="config/app_config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.default_config = {
            "ui_mode": "light",
            "aspect_ratio": "3:4",
            "quality_preset": "medium", 
            "auto_augmentation": True,
            "crop_method": "face_aware",
            "last_input_path": "",
            "last_output_path": "",
            "max_workers": 4
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
        except Exception as e:
            print(f"Config load error: {e}")
        
        return self.default_config.copy()
    
    def save_config(self, config=None):
        """Save configuration"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Config save error: {e}")
    
    def get(self, key, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set config value"""
        self.config[key] = value
    
    def update(self, updates):
        """Update config"""
        self.config.update(updates)
