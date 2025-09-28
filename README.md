# Image Auto-Cropping & Augmentation Tool v2.0

A professional-grade PyQt5 application for intelligent image cropping with advanced face detection, automatic orientation correction, and comprehensive augmentation capabilities.

## Key Improvements & Bug Fixes

### Fixed Issues from Original Version:
1. **No More Image Stretching** - Images are properly cropped maintaining aspect ratio without distortion
2. **Intelligent Auto-Augmentation** - Automatically detects and corrects orientation issues (rotation, upside-down images)
3. **Professional Project Structure** - Modular, maintainable codebase with proper separation of concerns
4. **Enhanced Face Detection** - Improved accuracy with MediaPipe integration and OpenCV fallbacks
5. **Better UI/UX** - Modern interface with real-time preview, batch analysis, and comprehensive progress tracking

## Features

### Core Processing
- **Intelligent Face Detection**: MediaPipe + OpenCV with landmark-based alignment
- **Smart Cropping Methods**: Face-aware, Smart crop (edge detection), Center crop
- **Aspect Ratio Support**: 1:1, 3:4, 4:3, 16:9, 9:16 with proper handling
- **No Stretching**: Maintains image proportions with optional padding
- **High-Quality Output**: Configurable compression levels

### Auto-Augmentation System
- **Automatic Orientation Detection**: Detects rotated or upside-down images
- **Face-Based Correction**: Uses facial landmarks for precise alignment
- **Edge-Based Analysis**: Fallback method when no face is detected
- **Batch Consistency**: Analyzes entire datasets for uniform corrections

### Manual Augmentation Options
- Rotation: 90°, 180°, 270°
- Flipping: Horizontal, Vertical
- Custom angle rotation for fine adjustments

### Advanced UI Features
- **Real-time Preview**: See crop areas before processing
- **Batch Analysis**: Consistency checking across image sets
- **Processing History**: Track all operations with thumbnails
- **Progress Tracking**: Detailed progress with file-by-file status
- **Time Estimation**: Predict processing duration
- **Theme Support**: Light and dark modes
- **Memory Monitoring**: Real-time resource usage display

### Professional Capabilities
- **Multi-threading**: Parallel processing for speed
- **Error Handling**: Comprehensive logging and recovery
- **Configuration Management**: Save/load processing presets
- **Export Options**: HTML reports, processing logs, preview images
- **Batch Processing**: Handle thousands of images efficiently

## Installation

### Requirements
- Python 3.7 or higher
- See `requirements.txt` for complete dependencies

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd image-crop-tool

# Install dependencies
pip install -r requirements.txt

# Optional: Install MediaPipe for better face detection
pip install mediapipe

# Run the application
python main.py
```

### For Enhanced Face Detection
```bash
pip install mediapipe
```
Note: MediaPipe may not be available on all systems. The application will fall back to enhanced OpenCV detection automatically.

## Project Structure

```
image-crop-tool/
├── main.py                    # Application entry point
├── requirements.txt           # Dependencies
├── README.md                 # This file
│
├── src/                      # Source code
│   ├── core/                 # Core processing modules
│   │   ├── __init__.py
│   │   ├── logger.py         # Logging system
│   │   ├── config_manager.py # Configuration management
│   │   ├── face_detector.py  # Face detection (MediaPipe + OpenCV)
│   │   ├── image_cropper.py  # Intelligent cropping
│   │   ├── image_augmenter.py# Auto-augmentation system
│   │   └── image_processor.py# Batch processing engine
│   │
│   ├── ui/                   # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py    # Main application window
│   │   ├── preview_dialog.py # Preview and history dialog
│   │   ├── settings_dialog.py# Advanced settings
│   │   └── batch_analysis_dialog.py # Batch analysis
│   │
│   └── utils/                # Utility modules
│       ├── __init__.py
│       └── image_utils.py    # Image processing utilities
│
├── config/                   # Configuration files
│   └── app_config.json      # User preferences
│
├── logs/                     # Application logs
│   └── app_YYYYMMDD_HHMMSS.log
│
├── output/                   # Default output directory
│
└── resources/                # Application resources
    └── themes/              # UI themes
        ├── light.qss
        └── dark.qss
```

## Usage Guide

### Basic Workflow
1. **Select Input Folder**: Choose folder containing your images
2. **Set Output Folder**: Choose destination for processed images
3. **Configure Settings**: 
   - Aspect ratio (1:1, 3:4, 4:3, 16:9, 9:16)
   - Crop method (Face Aware, Smart, Center)
   - Quality preset (Low, Medium, High)
4. **Enable Auto-Augmentation**: For automatic orientation correction
5. **Preview Sample**: Test settings on a sample image
6. **Start Processing**: Begin batch processing

### Crop Methods Explained

#### Face Aware Cropping
- Detects faces using MediaPipe or OpenCV
- Positions eyes in the upper third (rule of thirds)
- Maintains natural head positioning
- Automatically rotates for face alignment
- Falls back to smart crop if no face found

#### Smart Cropping
- Uses edge detection to find interesting areas
- Analyzes gradient magnitude across the image
- Selects regions with highest detail/contrast
- Good for landscapes, objects, non-portrait images

#### Center Cropping
- Simple center-based cropping
- Maintains aspect ratio without analysis
- Fastest method for uniform datasets
- Good for pre-aligned images

### Auto-Augmentation Features

The auto-augmentation system automatically detects and corrects:

1. **Rotated Images**: Uses face landmarks or edge analysis to detect rotation
2. **Upside-down Photos**: Detects when faces or content is inverted
3. **Tilted Faces**: Corrects face rotation based on eye alignment
4. **Batch Consistency**: Ensures uniform orientation across image sets

### Advanced Settings

Access advanced settings through the "Advanced Settings" button:

- **Face Detection Confidence**: Adjust detection sensitivity
- **Crop Padding**: Control whitespace around detected faces
- **Threading Workers**: Optimize for your CPU
- **Memory Limits**: Prevent system overload
- **Output Formats**: Choose between PNG, JPEG, WebP

### Preview System

The preview system offers three views:

1. **Sample Preview**: Test settings on individual images
2. **Processing History**: Track all processed images with thumbnails
3. **Batch Preview**: Grid view of multiple processed images

## Performance Optimization

### Threading Configuration
- Default: 4 workers for balanced performance
- Increase for high-core CPUs
- Decrease if experiencing memory issues

### Memory Management
- Automatic memory monitoring
- Configurable memory limits
- Batch size adjustment for large datasets

### Speed Tips
1. Use "Center Crop" for fastest processing
2. Reduce output size for quicker processing
3. Disable auto-augmentation if not needed
4. Process in smaller batches for very large datasets

## Configuration Management

### Saving Configurations
- Automatically saves UI settings
- Export custom presets
- Load configurations for different projects

### Configuration Options
```json
{
  "aspect_ratio": "3:4",
  "crop_method": "face_aware",
  "quality_preset": "medium",
  "auto_augmentation": true,
  "face_detection_confidence": 0.5,
  "max_workers": 4,
  "memory_limit_mb": 1000,
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
}
```

## Error Handling & Logging

### Comprehensive Logging
- All operations logged with timestamps
- Error details for troubleshooting
- Processing statistics and performance metrics
- Separate log files for each session

### Error Recovery
- Graceful handling of corrupted images
- Automatic fallbacks for processing failures
- Detailed error reporting with suggested solutions

## Development & Maintenance

### Code Structure
- **Modular Design**: Easy to extend and maintain
- **Clean Separation**: UI, processing, and configuration separated
- **Type Hints**: Full type annotation for better IDE support
- **Documentation**: Comprehensive docstrings and comments

### Testing
```bash
# Run tests (if pytest installed)
pytest tests/

# Code formatting
black src/

# Code linting
flake8 src/
```

### Adding New Features
1. **New Crop Methods**: Extend `ImageCropper` class
2. **Additional Augmentations**: Add methods to `ImageAugmenter`
3. **UI Components**: Create new dialogs in `ui/` directory
4. **Configuration Options**: Update `ConfigManager`

## Troubleshooting

### Common Issues

#### MediaPipe Installation Issues
```bash
# If MediaPipe fails to install, use OpenCV only
pip install -r requirements.txt --ignore-installed mediapipe
```
The application automatically falls back to enhanced OpenCV detection.

#### Memory Issues
- Reduce batch size in advanced settings
- Lower memory limit threshold
- Process images in smaller groups
- Use "Low" quality preset for testing

#### Face Detection Issues
- Increase face detection confidence
- Try different crop methods
- Ensure images have clear, frontal faces
- Check image quality and lighting

#### Processing Errors
- Check logs in `logs/` directory
- Verify image file formats are supported
- Ensure sufficient disk space
- Check file permissions

### Performance Issues
- Reduce number of worker threads
- Use center crop for speed
- Lower output resolution
- Disable auto-augmentation

## API Reference (For Developers)

### Core Classes

#### FaceDetector
```python
from src.core.face_detector import FaceDetector

detector = FaceDetector(confidence_threshold=0.5)
faces = detector.detect_faces(image)
primary_face = detector.get_primary_face(image)
```

#### ImageCropper
```python
from src.core.image_cropper import ImageCropper

cropper = ImageCropper()
cropped = cropper.crop_image(
    image=image,
    aspect_ratio='3:4',
    crop_method='face_aware'
)
```

#### ImageAugmenter
```python
from src.core.image_augmenter import ImageAugmenter

augmenter = ImageAugmenter()
issues = augmenter.detect_image_issues(image)
corrected, applied = augmenter.auto_correct_image(image)
```

## Contributing

### Guidelines
1. Follow existing code style and structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Use type hints throughout
5. Follow Git commit message conventions

### Pull Request Process
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

MIT License - See LICENSE file for details.

## Changelog

### v2.0 (Current)
- **Fixed**: Image stretching bug - images now maintain proper aspect ratios
- **Fixed**: Improved augmentation system with automatic orientation detection
- **Added**: Professional project structure with modular design
- **Added**: Real-time preview system with crop area visualization
- **Added**: Batch analysis and consistency checking
- **Added**: Enhanced face detection with MediaPipe integration
- **Added**: Comprehensive error handling and logging
- **Added**: Theme support (light/dark modes)
- **Added**: Memory monitoring and performance optimization
- **Added**: Export capabilities (reports, previews, configurations)
- **Improved**: UI/UX with modern design and better workflow
- **Improved**: Processing speed with optimized threading

### v1.0 (Previous)
- Basic cropping functionality
- Simple face detection
- Basic augmentation options
- Single-file structure

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Open an issue on GitHub with detailed information
4. Include relevant log files and system information

## Acknowledgments

- **MediaPipe**: Google's ML solutions for face detection
- **OpenCV**: Computer vision library for image processing  
- **PyQt5**: GUI framework for the application interface
- **NumPy**: Numerical computing for image operations

---

**Built for professional image processing workflows with attention to quality, performance, and maintainability.**# CropV2
#   C r o p V 2  
 # CropV2
