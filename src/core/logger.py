"""
Minimal Logging System
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Setup basic logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('ImageCropTool')
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(f'ImageCropTool.{name}')
