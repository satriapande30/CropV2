"""
Intelligent image cropping with face awareness and proper aspect ratio handling
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from .face_detector import FaceDetector
from .logger import get_logger

logger = get_logger("ImageCropper")

class ImageCropper:
    """Intelligent image cropping with face alignment and proper aspect ratio handling"""
    
    def __init__(self, face_detector: FaceDetector = None):
        self.face_detector = face_detector or FaceDetector()
        
        # Standard aspect ratios
        self.aspect_ratios = {
            '1:1': (1, 1),
            '3:4': (3, 4),
            '4:3': (4, 3),
            '16:9': (16, 9),
            '9:16': (9, 16)
        }
    
    def crop_image(self, image: np.ndarray, aspect_ratio: str = '3:4', 
                   target_size: Optional[Tuple[int, int]] = None,
                   crop_method: str = 'face_aware',
                   padding: float = 0.1) -> np.ndarray:
        """
        Crop image with specified aspect ratio and method
        
        Args:
            image: Input image as numpy array
            aspect_ratio: Target aspect ratio ('1:1', '3:4', etc.)
            target_size: Optional target size (width, height). If None, preserves original resolution
            crop_method: 'face_aware', 'center', 'smart'
            padding: Padding around detected face (as fraction of face size)
        
        Returns:
            Cropped image without stretching
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        h, w = image.shape[:2]
        logger.info(f"Processing image: {w}x{h}, method: {crop_method}")
        
        # Get aspect ratio
        aspect_w, aspect_h = self.aspect_ratios.get(aspect_ratio, (3, 4))
        target_aspect = aspect_w / aspect_h
        
        # Determine crop method
        if crop_method == 'face_aware':
            cropped = self._crop_face_aware(image, target_aspect, padding)
        elif crop_method == 'smart':
            cropped = self._crop_smart(image, target_aspect)
        else:  # center crop
            cropped = self._crop_center(image, target_aspect)
        
        # Resize if target size specified (maintaining aspect ratio)
        if target_size:
            cropped = self._resize_maintain_aspect(cropped, target_size)
        
        return cropped
    
    def _crop_face_aware(self, image: np.ndarray, target_aspect: float, 
                        padding: float) -> np.ndarray:
        """Crop image with face detection and alignment"""
        h, w = image.shape[:2]
        
        # Detect primary face
        face_info = self.face_detector.get_primary_face(image)
        
        if face_info:
            logger.info("Face detected, using face-aware cropping")
            return self._crop_with_face(image, face_info, target_aspect, padding)
        else:
            logger.info("No face detected, falling back to smart crop")
            return self._crop_smart(image, target_aspect)
    
    def _crop_with_face(self, image: np.ndarray, face_info: Dict, 
                       target_aspect: float, padding: float) -> np.ndarray:
        """Crop image with face positioning and eye alignment"""
        h, w = image.shape[:2]
        
        # Get face information
        bbox = face_info['bbox']
        landmarks = face_info['landmarks']
        rotation = face_info.get('rotation', 0)
        
        # Rotate image if face is significantly tilted
        if abs(rotation) > 2.0:  # More than 2 degrees
            logger.info(f"Rotating image by {-rotation:.1f} degrees for face alignment")
            image = self._rotate_image(image, -rotation)
            h, w = image.shape[:2]
            
            # Recalculate face position after rotation
            face_info = self.face_detector.get_primary_face(image)
            if face_info:
                bbox = face_info['bbox']
                landmarks = face_info['landmarks']
        
        # Calculate face center and eye center
        face_center_x = bbox['x'] + bbox['width'] // 2
        face_center_y = bbox['y'] + bbox['height'] // 2
        eye_center = landmarks.get('eye_center', (face_center_x, face_center_y))
        
        # Calculate crop dimensions that maintain aspect ratio
        if target_aspect >= 1:  # Landscape or square
            crop_height = min(h, int(w / target_aspect))
            crop_width = int(crop_height * target_aspect)
        else:  # Portrait
            crop_width = min(w, int(h * target_aspect))
            crop_height = int(crop_width / target_aspect)
        
        # Position crop area to place eyes in upper third (rule of thirds)
        target_eye_y = crop_height // 3
        
        # Calculate crop coordinates
        crop_x = max(0, min(w - crop_width, eye_center[0] - crop_width // 2))
        crop_y = max(0, min(h - crop_height, eye_center[1] - target_eye_y))
        
        # Apply padding if face is too close to edges
        face_bbox_padded = self._add_padding_to_bbox(bbox, padding)
        
        # Ensure face with padding fits in crop
        min_x = max(0, face_bbox_padded['x'] + face_bbox_padded['width'] - crop_width)
        max_x = min(w - crop_width, face_bbox_padded['x'])
        min_y = max(0, face_bbox_padded['y'] + face_bbox_padded['height'] - crop_height)
        max_y = min(h - crop_height, face_bbox_padded['y'])
        
        crop_x = max(min_x, min(max_x, crop_x))
        crop_y = max(min_y, min(max_y, crop_y))
        
        # Extract crop
        cropped = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
        logger.info(f"Face-aware crop: {crop_width}x{crop_height} at ({crop_x}, {crop_y})")
        return cropped
    
    def _crop_smart(self, image: np.ndarray, target_aspect: float) -> np.ndarray:
        """Smart cropping using edge detection and interest points"""
        h, w = image.shape[:2]
        
        # Calculate crop dimensions
        if target_aspect >= 1:  # Landscape or square
            crop_height = min(h, int(w / target_aspect))
            crop_width = int(crop_height * target_aspect)
        else:  # Portrait
            crop_width = min(w, int(h * target_aspect))
            crop_height = int(crop_width / target_aspect)
        
        # Use saliency detection for smart positioning
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple saliency using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Find region with highest average saliency
            best_score = 0
            best_x, best_y = w // 2 - crop_width // 2, h // 2 - crop_height // 2
            
            # Test multiple positions
            step = min(crop_width // 4, crop_height // 4, 50)
            for y in range(0, h - crop_height + 1, step):
                for x in range(0, w - crop_width + 1, step):
                    roi = magnitude[y:y + crop_height, x:x + crop_width]
                    score = np.mean(roi)
                    if score > best_score:
                        best_score = score
                        best_x, best_y = x, y
            
            crop_x, crop_y = best_x, best_y
            
        except Exception as e:
            logger.warning(f"Smart crop failed, using center crop: {e}")
            crop_x = w // 2 - crop_width // 2
            crop_y = h // 2 - crop_height // 2
        
        # Extract crop
        cropped = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        logger.info(f"Smart crop: {crop_width}x{crop_height} at ({crop_x}, {crop_y})")
        return cropped
    
    def _crop_center(self, image: np.ndarray, target_aspect: float) -> np.ndarray:
        """Simple center crop maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate crop dimensions
        if target_aspect >= 1:  # Landscape or square
            crop_height = min(h, int(w / target_aspect))
            crop_width = int(crop_height * target_aspect)
        else:  # Portrait
            crop_width = min(w, int(h * target_aspect))
            crop_height = int(crop_width / target_aspect)
        
        # Center the crop
        crop_x = (w - crop_width) // 2
        crop_y = (h - crop_height) // 2
        
        cropped = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        logger.info(f"Center crop: {crop_width}x{crop_height} at ({crop_x}, {crop_y})")
        return cropped
    
    def _resize_maintain_aspect(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image maintaining aspect ratio without stretching"""
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calculate aspect ratios
        current_aspect = w / h
        target_aspect = target_w / target_h
        
        if abs(current_aspect - target_aspect) < 0.01:  # Already correct aspect ratio
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # If aspects don't match, pad with black bars instead of stretching
        if current_aspect > target_aspect:  # Image is wider
            new_width = target_w
            new_height = int(target_w / current_aspect)
        else:  # Image is taller
            new_height = target_h
            new_width = int(target_h * current_aspect)
        
        # Resize to fit
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_h - new_height) // 2
        x_offset = (target_w - new_width) // 2
        
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle in degrees"""
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
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LANCZOS4, 
                                borderValue=(0, 0, 0))
        
        return rotated
    
    def _add_padding_to_bbox(self, bbox: Dict, padding: float) -> Dict:
        """Add padding around bounding box"""
        pad_w = int(bbox['width'] * padding)
        pad_h = int(bbox['height'] * padding)
        
        return {
            'x': max(0, bbox['x'] - pad_w),
            'y': max(0, bbox['y'] - pad_h),
            'width': bbox['width'] + 2 * pad_w,
            'height': bbox['height'] + 2 * pad_h
        }
    
    def get_crop_preview(self, image: np.ndarray, aspect_ratio: str = '3:4',
                        crop_method: str = 'face_aware') -> Tuple[np.ndarray, Dict]:
        """
        Get a preview of how the image will be cropped
        
        Returns:
            Tuple of (preview_image, crop_info)
        """
        h, w = image.shape[:2]
        
        # Get aspect ratio
        aspect_w, aspect_h = self.aspect_ratios.get(aspect_ratio, (3, 4))
        target_aspect = aspect_w / aspect_h
        
        # Create preview image (copy of original)
        preview = image.copy()
        
        # Calculate crop area
        if crop_method == 'face_aware':
            face_info = self.face_detector.get_primary_face(image)
            if face_info:
                # Show detected face
                bbox = face_info['bbox']
                cv2.rectangle(preview, 
                            (bbox['x'], bbox['y']), 
                            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                            (0, 255, 0), 2)  # Green rectangle for face
                
                # Show eye positions
                landmarks = face_info['landmarks']
                if 'left_eye' in landmarks:
                    cv2.circle(preview, landmarks['left_eye'], 3, (0, 255, 0), -1)
                if 'right_eye' in landmarks:
                    cv2.circle(preview, landmarks['right_eye'], 3, (0, 255, 0), -1)
        
        # Calculate and show crop area
        if target_aspect >= 1:  # Landscape or square
            crop_height = min(h, int(w / target_aspect))
            crop_width = int(crop_height * target_aspect)
        else:  # Portrait
            crop_width = min(w, int(h * target_aspect))
            crop_height = int(crop_width / target_aspect)
        
        # Get crop position (simplified for preview)
        if crop_method == 'face_aware' and face_info:
            eye_center = face_info['landmarks'].get('eye_center', (w//2, h//2))
            crop_x = max(0, min(w - crop_width, eye_center[0] - crop_width // 2))
            crop_y = max(0, min(h - crop_height, eye_center[1] - crop_height // 3))
        else:
            crop_x = (w - crop_width) // 2
            crop_y = (h - crop_height) // 2
        
        # Draw crop area
        cv2.rectangle(preview, 
                     (crop_x, crop_y), 
                     (crop_x + crop_width, crop_y + crop_height),
                     (0, 0, 255), 2)  # Red rectangle for crop area
        
        crop_info = {
            'crop_x': crop_x,
            'crop_y': crop_y,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'face_detected': face_info is not None if crop_method == 'face_aware' else False
        }
        
        return preview, crop_info