"""
Intelligent image augmentation with automatic orientation detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from .face_detector import FaceDetector
from .logger import get_logger

logger = get_logger("ImageAugmenter")

class ImageAugmenter:
    """Intelligent image augmentation with auto-correction capabilities"""
    
    def __init__(self, face_detector: FaceDetector = None):
        self.face_detector = face_detector or FaceDetector()
        
        # Augmentation methods
        self.augmentation_methods = {
            'rotate_90': self.rotate_90,
            'rotate_180': self.rotate_180,
            'rotate_270': self.rotate_270,
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical,
            'auto_rotate': self.auto_rotate_correct,
            'auto_flip': self.auto_flip_correct
        }
    
    def detect_image_issues(self, image: np.ndarray) -> Dict[str, bool]:
        """
        Detect common image orientation issues
        
        Returns:
            Dictionary with detected issues: {'needs_rotation': bool, 'needs_flip': bool, 'rotation_angle': float}
        """
        issues = {
            'needs_rotation': False,
            'needs_flip': False,
            'rotation_angle': 0.0,
            'is_upside_down': False
        }
        
        # Check face orientation
        face_info = self.face_detector.get_primary_face(image)
        
        if face_info:
            rotation = face_info.get('rotation', 0.0)
            
            # Check if significant rotation is needed
            if abs(rotation) > 5.0:  # More than 5 degrees
                issues['needs_rotation'] = True
                issues['rotation_angle'] = -rotation  # Correct rotation
                logger.info(f"Face rotation detected: {rotation:.1f} degrees")
            
            # Check if image might be upside down
            landmarks = face_info['landmarks']
            if 'eye_center' in landmarks and 'mouth' in landmarks:
                eye_y = landmarks['eye_center'][1]
                mouth_y = landmarks['mouth'][1]
                
                # If mouth is above eyes, image might be upside down
                if mouth_y < eye_y:
                    issues['is_upside_down'] = True
                    issues['needs_rotation'] = True
                    issues['rotation_angle'] = 180.0
                    logger.info("Upside down orientation detected")
        
        # Use edge detection to detect orientation
        else:
            issues.update(self._detect_orientation_by_edges(image))
        
        return issues
    
    def _detect_orientation_by_edges(self, image: np.ndarray) -> Dict[str, bool]:
        """Detect orientation using edge analysis when no face is found"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:  # Use first 20 lines
                angle = np.degrees(theta)
                # Convert to -90 to 90 range
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            
            if angles:
                # Find dominant angle
                hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
                dominant_angle_idx = np.argmax(hist)
                dominant_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
                
                if abs(dominant_angle) > 5:
                    return {
                        'needs_rotation': True,
                        'rotation_angle': -dominant_angle,
                        'is_upside_down': abs(dominant_angle) > 90
                    }
        
        return {'needs_rotation': False, 'rotation_angle': 0.0, 'is_upside_down': False}
    
    def auto_correct_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Automatically correct common image issues
        
        Returns:
            Tuple of (corrected_image, list_of_applied_corrections)
        """
        corrected = image.copy()
        applied_corrections = []
        
        # Detect issues
        issues = self.detect_image_issues(image)
        
        # Apply auto-rotation if needed
        if issues['needs_rotation']:
            angle = issues['rotation_angle']
            if abs(angle - 180) < 5:  # Close to 180 degrees
                corrected = self.rotate_180(corrected)
                applied_corrections.append('rotate_180')
            elif abs(angle - 90) < 5:  # Close to 90 degrees
                corrected = self.rotate_90(corrected)
                applied_corrections.append('rotate_90')
            elif abs(angle + 90) < 5:  # Close to -90 degrees
                corrected = self.rotate_270(corrected)
                applied_corrections.append('rotate_270')
            else:
                # Custom angle rotation
                corrected = self._rotate_custom_angle(corrected, angle)
                applied_corrections.append(f'rotate_{angle:.1f}')
        
        logger.info(f"Auto-corrections applied: {applied_corrections}")
        return corrected, applied_corrections
    
    def apply_augmentations(self, image: np.ndarray, augmentations: List[str], 
                          auto_correct: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Apply selected augmentations to image
        
        Args:
            image: Input image
            augmentations: List of augmentation names
            auto_correct: Whether to apply auto-correction first
        
        Returns:
            List of (augmentation_name, augmented_image) tuples
        """
        results = []
        base_image = image.copy()
        
        # Apply auto-correction first if requested
        if auto_correct and ('auto_rotate' in augmentations or 'auto_flip' in augmentations):
            corrected_image, corrections = self.auto_correct_image(image)
            if corrections:
                results.append(('auto_corrected', corrected_image))
                base_image = corrected_image
        
        # Apply other augmentations
        for aug_name in augmentations:
            if aug_name in ['auto_rotate', 'auto_flip']:
                continue  # Already handled above
                
            if aug_name in self.augmentation_methods:
                try:
                    augmented = self.augmentation_methods[aug_name](base_image.copy())
                    results.append((aug_name, augmented))
                    logger.debug(f"Applied augmentation: {aug_name}")
                except Exception as e:
                    logger.error(f"Failed to apply augmentation {aug_name}: {e}")
        
        return results
    
    # Basic augmentation methods
    def rotate_90(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 90 degrees clockwise"""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees"""
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def rotate_270(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 270 degrees clockwise (90 counter-clockwise)"""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically"""
        return cv2.flip(image, 0)
    
    def auto_rotate_correct(self, image: np.ndarray) -> np.ndarray:
        """Auto-correct rotation based on face or edge detection"""
        corrected, _ = self.auto_correct_image(image)
        return corrected
    
    def auto_flip_correct(self, image: np.ndarray) -> np.ndarray:
        """Auto-correct flip (currently same as auto-rotate)"""
        return self.auto_rotate_correct(image)
    
    def _rotate_custom_angle(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by custom angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2
        
        # Perform rotation with black background
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        
        return rotated
    
    def analyze_batch_consistency(self, images: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze a batch of images for consistency issues
        
        Returns:
            Analysis report with recommendations
        """
        analysis = {
            'total_images': len(images),
            'faces_detected': 0,
            'rotation_issues': 0,
            'orientation_issues': 0,
            'recommendations': []
        }
        
        rotation_angles = []
        
        for i, image in enumerate(images):
            issues = self.detect_image_issues(image)
            
            if self.face_detector.get_primary_face(image):
                analysis['faces_detected'] += 1
            
            if issues['needs_rotation']:
                analysis['rotation_issues'] += 1
                rotation_angles.append(issues['rotation_angle'])
            
            if issues['is_upside_down']:
                analysis['orientation_issues'] += 1
        
        # Generate recommendations
        if analysis['rotation_issues'] > len(images) * 0.3:  # More than 30% have rotation issues
            analysis['recommendations'].append('Enable auto-rotation correction')
        
        if analysis['orientation_issues'] > 0:
            analysis['recommendations'].append('Some images may be upside down')
        
        if len(rotation_angles) > 0:
            avg_rotation = np.mean(rotation_angles)
            if abs(avg_rotation) > 2:
                analysis['recommendations'].append(f'Average rotation needed: {avg_rotation:.1f}°')
        
        face_detection_rate = analysis['faces_detected'] / analysis['total_images'] * 100
        analysis['face_detection_rate'] = face_detection_rate
        
        if face_detection_rate < 50:
            analysis['recommendations'].append('Low face detection rate - consider using smart crop instead')
        
        return analysis
    
    def get_augmentation_preview(self, image: np.ndarray, augmentations: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate preview thumbnails for augmentations
        
        Returns:
            Dictionary mapping augmentation names to preview images
        """
        previews = {}
        
        # Create thumbnail size for previews
        h, w = image.shape[:2]
        thumb_size = (200, int(200 * h / w)) if w > h else (int(200 * w / h), 200)
        
        for aug_name in augmentations:
            if aug_name in self.augmentation_methods:
                try:
                    augmented = self.augmentation_methods[aug_name](image.copy())
                    thumbnail = cv2.resize(augmented, thumb_size, interpolation=cv2.INTER_AREA)
                    previews[aug_name] = thumbnail
                except Exception as e:
                    logger.error(f"Failed to generate preview for {aug_name}: {e}")
        
        return previews