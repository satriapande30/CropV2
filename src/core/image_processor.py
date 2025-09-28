"""
Batch image processing with threading and progress tracking
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtCore import QThread, pyqtSignal
import time

from .face_detector import FaceDetector
from .image_cropper import ImageCropper
from .image_augmenter import ImageAugmenter
from .logger import get_logger

logger = get_logger("ImageProcessor")

class ProcessingStats:
    """Track processing statistics"""
    
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.successful_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        self.start_time = None
        self.end_time = None
        self.errors = []
    
    def start_processing(self, total_files: int):
        """Start processing timer"""
        self.total_files = total_files
        self.start_time = time.time()
        logger.info(f"Starting processing of {total_files} files")
    
    def file_processed(self, success: bool, error_msg: str = None):
        """Record file processing result"""
        self.processed_files += 1
        if success:
            self.successful_files += 1
        else:
            self.failed_files += 1
            if error_msg:
                self.errors.append(error_msg)
    
    def file_skipped(self, reason: str):
        """Record skipped file"""
        self.skipped_files += 1
        self.processed_files += 1
        logger.debug(f"File skipped: {reason}")
    
    def finish_processing(self):
        """End processing timer"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Processing complete in {duration:.2f}s: "
                   f"{self.successful_files}/{self.total_files} successful")
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'success_rate': self.successful_files / max(1, self.total_files) * 100,
            'duration_seconds': duration,
            'files_per_second': self.processed_files / max(1, duration),
            'errors': self.errors
        }

class ImageProcessor(QThread):
    """Background image processing with threading and progress tracking"""
    
    # Qt signals for UI updates
    progress_updated = pyqtSignal(int, str)  # progress percentage, current file
    file_processed = pyqtSignal(str, bool, str)  # filename, success, message
    processing_complete = pyqtSignal(dict)  # processing stats
    preview_ready = pyqtSignal(str, np.ndarray, np.ndarray)  # filename, original, processed
    batch_analysis_ready = pyqtSignal(dict)  # batch analysis results
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        
        # Initialize processing components
        self.face_detector = FaceDetector(
            confidence_threshold=config_manager.get('face_detection_confidence', 0.5)
        )
        self.image_cropper = ImageCropper(self.face_detector)
        self.image_augmenter = ImageAugmenter(self.face_detector)
        
        # Processing parameters
        self.input_folder = ""
        self.output_folder = ""
        self.aspect_ratio = "3:4"
        self.quality_preset = "medium"
        self.target_size = None
        self.augmentations = []
        self.crop_method = "face_aware"
        self.auto_augmentation = True
        
        # Control flags
        self.is_cancelled = False
        self.is_paused = False
        
        # Statistics
        self.stats = ProcessingStats()
        
        # Supported formats
        self.supported_formats = config_manager.get('supported_formats', 
            ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
        
        # Threading settings
        self.max_workers = config_manager.get('max_workers', 4)
        self.batch_size = config_manager.get('batch_size', 10)
    
    def setup_processing(self, input_folder: str, output_folder: str, 
                        aspect_ratio: str = '3:4', target_size: Optional[Tuple[int, int]] = None,
                        quality_preset: str = 'medium', augmentations: List[str] = None,
                        crop_method: str = 'face_aware', auto_augmentation: bool = True):
        """Setup processing parameters"""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size
        self.quality_preset = quality_preset
        self.augmentations = augmentations or []
        self.crop_method = crop_method
        self.auto_augmentation = auto_augmentation
        
        self.is_cancelled = False
        self.is_paused = False
        
        logger.info(f"Processing setup: {input_folder} -> {output_folder}")
        logger.info(f"Settings: {aspect_ratio}, {crop_method}, auto_aug: {auto_augmentation}")
    
    def cancel_processing(self):
        """Cancel the current processing"""
        self.is_cancelled = True
        logger.info("Processing cancellation requested")
    
    def pause_processing(self):
        """Pause processing"""
        self.is_paused = True
        logger.info("Processing paused")
    
    def resume_processing(self):
        """Resume processing"""
        self.is_paused = False
        logger.info("Processing resumed")
    
    def run(self):
        """Main processing loop"""
        try:
            # Get list of image files
            image_files = self._get_image_files()
            
            if not image_files:
                self.processing_complete.emit({'error': 'No image files found'})
                return
            
            # Initialize statistics
            self.stats = ProcessingStats()
            self.stats.start_processing(len(image_files))
            
            # Create output directory
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Analyze batch if auto-augmentation is enabled
            if self.auto_augmentation:
                self._analyze_batch(image_files)
            
            # Process files
            if self.max_workers > 1:
                self._process_files_threaded(image_files)
            else:
                self._process_files_sequential(image_files)
            
            # Finish processing
            self.stats.finish_processing()
            self.processing_complete.emit(self.stats.get_summary())
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.processing_complete.emit({'error': str(e)})
    
    def _get_image_files(self) -> List[Path]:
        """Get list of supported image files from input folder"""
        image_files = []
        input_path = Path(self.input_folder)
        
        if not input_path.exists():
            logger.error(f"Input folder does not exist: {self.input_folder}")
            return []
        
        # Recursively find image files
        for ext in self.supported_formats:
            # Add both lowercase and uppercase extensions
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        logger.info(f"Found {len(image_files)} image files")
        return image_files
    
    def _analyze_batch(self, image_files: List[Path]):
        """Analyze batch of images for consistency issues"""
        try:
            # Sample up to 20 images for analysis to avoid slowdown
            sample_files = image_files[:20] if len(image_files) > 20 else image_files
            sample_images = []
            
            for file_path in sample_files:
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        sample_images.append(image)
                except Exception as e:
                    logger.warning(f"Could not load sample image {file_path}: {e}")
            
            if sample_images:
                analysis = self.image_augmenter.analyze_batch_consistency(sample_images)
                self.batch_analysis_ready.emit(analysis)
                logger.info(f"Batch analysis: {analysis['recommendations']}")
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
    
    def _process_files_sequential(self, image_files: List[Path]):
        """Process files sequentially"""
        for i, file_path in enumerate(image_files):
            if self.is_cancelled:
                break
            
            # Handle pause
            while self.is_paused and not self.is_cancelled:
                time.sleep(0.1)
            
            if self.is_cancelled:
                break
            
            # Process single file
            self._process_single_file(file_path)
            
            # Update progress
            progress = int((i + 1) / len(image_files) * 100)
            self.progress_updated.emit(progress, file_path.name)
    
    def _process_files_threaded(self, image_files: List[Path]):
        """Process files using thread pool"""
        completed_count = 0
        total_files = len(image_files)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path 
                for file_path in image_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                if self.is_cancelled:
                    # Cancel remaining tasks
                    for remaining_future in future_to_file:
                        remaining_future.cancel()
                    break
                
                file_path = future_to_file[future]
                completed_count += 1
                
                # Update progress
                progress = int(completed_count / total_files * 100)
                self.progress_updated.emit(progress, file_path.name)
                
                # Handle pause
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)
    
    def _process_single_file(self, file_path: Path) -> bool:
        """Process a single image file"""
        try:
            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            original_image = image.copy()  # Keep for preview
            
            # Auto-correction if enabled
            if self.auto_augmentation:
                corrected_image, corrections = self.image_augmenter.auto_correct_image(image)
                if corrections:
                    logger.info(f"Auto-corrections for {file_path.name}: {corrections}")
                    image = corrected_image
            
            # Crop image
            cropped_image = self.image_cropper.crop_image(
                image=image,
                aspect_ratio=self.aspect_ratio,
                target_size=self.target_size,
                crop_method=self.crop_method
            )
            
            # Save main processed image
            output_filename = f"cropped_{file_path.stem}.png"
            output_path = Path(self.output_folder) / output_filename
            
            # Get compression settings
            compression_params = self._get_compression_params()
            success = cv2.imwrite(str(output_path), cropped_image, compression_params)
            
            if not success:
                raise RuntimeError("Failed to save processed image")
            
            # Apply augmentations if requested
            if self.augmentations:
                augmented_images = self.image_augmenter.apply_augmentations(
                    cropped_image, self.augmentations, auto_correct=False
                )
                
                for aug_name, aug_image in augmented_images:
                    aug_filename = f"cropped_{file_path.stem}_{aug_name}.png"
                    aug_path = Path(self.output_folder) / aug_filename
                    cv2.imwrite(str(aug_path), aug_image, compression_params)
            
            # Record success
            self.stats.file_processed(True)
            self.file_processed.emit(file_path.name, True, "Successfully processed")
            
            # Emit preview if needed
            self.preview_ready.emit(file_path.name, original_image, cropped_image)
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            self.stats.file_processed(False, error_msg)
            self.file_processed.emit(file_path.name, False, error_msg)
            return False
    
    def _get_compression_params(self) -> List[int]:
        """Get compression parameters based on quality preset"""
        quality_settings = {
            'low': [cv2.IMWRITE_PNG_COMPRESSION, 9],      # High compression
            'medium': [cv2.IMWRITE_PNG_COMPRESSION, 6],   # Medium compression  
            'high': [cv2.IMWRITE_PNG_COMPRESSION, 1]      # Low compression
        }
        
        return quality_settings.get(self.quality_preset, quality_settings['medium'])
    
    def get_processing_preview(self, sample_file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a preview of how a single file will be processed
        
        Returns:
            Tuple of (original_image, preview_with_crop_area, crop_info)
        """
        try:
            # Load image
            image = cv2.imread(sample_file_path)
            if image is None:
                raise ValueError(f"Could not load image: {sample_file_path}")
            
            # Get crop preview
            preview_image, crop_info = self.image_cropper.get_crop_preview(
                image, self.aspect_ratio, self.crop_method
            )
            
            return image, preview_image, crop_info
            
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            raise
    
    def estimate_processing_time(self, sample_size: int = 5) -> Dict[str, float]:
        """
        Estimate total processing time based on sample processing
        
        Returns:
            Dictionary with time estimates
        """
        try:
            image_files = self._get_image_files()
            if not image_files:
                return {'error': 'No files found'}
            
            # Process a few sample files to estimate time
            sample_files = image_files[:min(sample_size, len(image_files))]
            total_sample_time = 0
            
            for file_path in sample_files:
                start_time = time.time()
                try:
                    # Quick processing test
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        self.image_cropper.crop_image(image, self.aspect_ratio, crop_method='center')
                    process_time = time.time() - start_time
                    total_sample_time += process_time
                except Exception:
                    continue
            
            if total_sample_time == 0:
                return {'error': 'Could not process samples'}
            
            # Calculate estimates
            avg_time_per_file = total_sample_time / len(sample_files)
            estimated_total_time = avg_time_per_file * len(image_files)
            
            # Account for threading
            if self.max_workers > 1:
                threading_efficiency = min(self.max_workers * 0.8, self.max_workers)  # 80% efficiency
                estimated_total_time /= threading_efficiency
            
            return {
                'total_files': len(image_files),
                'sample_files': len(sample_files),
                'avg_time_per_file': avg_time_per_file,
                'estimated_total_seconds': estimated_total_time,
                'estimated_total_minutes': estimated_total_time / 60,
                'threading_workers': self.max_workers
            }
            
        except Exception as e:
            logger.error(f"Time estimation failed: {e}")
            return {'error': str(e)}