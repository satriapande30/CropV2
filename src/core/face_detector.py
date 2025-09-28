"""
Enhanced face detection with MediaPipe and OpenCV fallback
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from .logger import get_logger

logger = get_logger("FaceDetector")

# Try to import MediaPipe, fall back to OpenCV if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe loaded successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.info("MediaPipe not available, using enhanced OpenCV")
    mp = None

class FaceDetector:
    """Enhanced face detection using MediaPipe or OpenCV"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self._init_mediapipe()
        else:
            self._init_opencv()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe components"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=self.confidence_threshold
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,  # Allow multiple faces
            refine_landmarks=True,
            min_detection_confidence=self.confidence_threshold
        )
    
    def _init_opencv(self):
        """Initialize OpenCV cascades"""
        try:
            # Load multiple cascade classifiers for better detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade_alt = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            
            logger.info("OpenCV cascades loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading OpenCV cascades: {e}")
            # Minimal fallback
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade_alt = None
            self.eye_cascade = None
            self.profile_cascade = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in image and return their information
        
        Returns:
            List of face dictionaries with keys: 'bbox', 'landmarks', 'confidence'
        """
        if self.use_mediapipe:
            return self._detect_with_mediapipe(image)
        else:
            return self._detect_with_opencv(image)
    
    def get_primary_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Get the primary (largest/most prominent) face from image
        
        Returns:
            Dictionary with face information or None if no face found
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Return the largest face (by area)
        primary_face = max(faces, key=lambda f: f['bbox']['width'] * f['bbox']['height'])
        return primary_face
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """MediaPipe face detection"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        faces = []
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            
            for landmarks in results.multi_face_landmarks:
                # Extract key landmarks
                left_eye = landmarks.landmark[33]   # Left eye center
                right_eye = landmarks.landmark[263] # Right eye center
                nose_tip = landmarks.landmark[1]    # Nose tip
                mouth_center = landmarks.landmark[13] # Mouth center
                
                # Convert to pixel coordinates
                left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))
                nose_pos = (int(nose_tip.x * w), int(nose_tip.y * h))
                mouth_pos = (int(mouth_center.x * w), int(mouth_center.y * h))
                
                # Calculate bounding box
                x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
                
                bbox = {
                    'x': min(x_coords),
                    'y': min(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords)
                }
                
                face_info = {
                    'bbox': bbox,
                    'landmarks': {
                        'left_eye': left_eye_pos,
                        'right_eye': right_eye_pos,
                        'nose': nose_pos,
                        'mouth': mouth_pos,
                        'eye_center': (
                            (left_eye_pos[0] + right_eye_pos[0]) // 2,
                            (left_eye_pos[1] + right_eye_pos[1]) // 2
                        )
                    },
                    'confidence': 1.0,  # MediaPipe doesn't provide confidence scores
                    'rotation': self._calculate_face_rotation(left_eye_pos, right_eye_pos)
                }
                
                faces.append(face_info)
        
        return faces
    
    def _detect_with_opencv(self, image: np.ndarray) -> List[Dict]:
        """Enhanced OpenCV face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast
        
        all_faces = []
        
        # Multiple detection methods with different parameters
        detection_methods = [
            (self.face_cascade, 1.05, 5, (30, 30)),
            (self.face_cascade, 1.1, 3, (50, 50)),
            (self.face_cascade, 1.2, 7, (40, 40)),
        ]
        
        # Add alternative cascades if available
        if self.face_cascade_alt:
            detection_methods.append((self.face_cascade_alt, 1.1, 5, (30, 30)))
        if self.profile_cascade:
            detection_methods.append((self.profile_cascade, 1.1, 5, (30, 30)))
        
        # Collect all detections
        for cascade, scale_factor, min_neighbors, min_size in detection_methods:
            if cascade is None:
                continue
                
            try:
                detected_faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                all_faces.extend(detected_faces)
            except Exception as e:
                logger.warning(f"Face detection method failed: {e}")
                continue
        
        # Remove duplicate detections
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        # Convert to standard format
        faces = []
        for x, y, w, h in unique_faces:
            # Detect eyes for better landmark estimation
            left_eye_pos, right_eye_pos = self._detect_eyes_in_face(gray, x, y, w, h)
            
            bbox = {'x': x, 'y': y, 'width': w, 'height': h}
            
            # Estimate other landmarks
            eye_center = (
                (left_eye_pos[0] + right_eye_pos[0]) // 2,
                (left_eye_pos[1] + right_eye_pos[1]) // 2
            )
            
            nose_pos = (x + w // 2, y + int(h * 0.6))
            mouth_pos = (x + w // 2, y + int(h * 0.8))
            
            face_info = {
                'bbox': bbox,
                'landmarks': {
                    'left_eye': left_eye_pos,
                    'right_eye': right_eye_pos,
                    'nose': nose_pos,
                    'mouth': mouth_pos,
                    'eye_center': eye_center
                },
                'confidence': 0.8,  # Estimated confidence
                'rotation': self._calculate_face_rotation(left_eye_pos, right_eye_pos)
            }
            
            faces.append(face_info)
        
        return faces
    
    def _remove_duplicate_faces(self, faces: List, overlap_threshold: float = 0.3) -> List:
        """Remove overlapping face detections using Non-Maximum Suppression"""
        if len(faces) <= 1:
            return faces
        
        # Convert to (x, y, x2, y2) format
        boxes = np.array([(x, y, x+w, y+h) for x, y, w, h in faces])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the largest remaining box
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            remaining_indices = indices[1:]
            current_box = boxes[current]
            remaining_boxes = boxes[remaining_indices]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas[current] + areas[remaining_indices] - intersection
            
            iou = intersection / union
            
            # Keep boxes with low IoU
            indices = remaining_indices[iou <= overlap_threshold]
        
        return [faces[i] for i in keep]
    
    def _detect_eyes_in_face(self, gray: np.ndarray, face_x: int, face_y: int, 
                           face_w: int, face_h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Detect eyes within face region"""
        face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
        
        if self.eye_cascade is not None:
            try:
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(10, 10)
                )
                
                # Filter eyes by position (should be in upper part of face)
                valid_eyes = [eye for eye in eyes if eye[1] < face_h * 0.6]
                
                if len(valid_eyes) >= 2:
                    # Sort by x-coordinate
                    valid_eyes = sorted(valid_eyes, key=lambda e: e[0])
                    left_eye = valid_eyes[0]
                    right_eye = valid_eyes[-1]
                    
                    # Convert to global coordinates
                    left_eye_pos = (
                        face_x + left_eye[0] + left_eye[2] // 2,
                        face_y + left_eye[1] + left_eye[3] // 2
                    )
                    right_eye_pos = (
                        face_x + right_eye[0] + right_eye[2] // 2,
                        face_y + right_eye[1] + right_eye[3] // 2
                    )
                    
                    return left_eye_pos, right_eye_pos
                    
            except Exception as e:
                logger.debug(f"Eye detection failed: {e}")
        
        # Fallback: estimate eye positions based on face proportions
        eye_y = face_y + int(face_h * 0.35)
        left_eye_x = face_x + int(face_w * 0.3)
        right_eye_x = face_x + int(face_w * 0.7)
        
        return (left_eye_x, eye_y), (right_eye_x, eye_y)
    
    def _calculate_face_rotation(self, left_eye: Tuple[int, int], 
                               right_eye: Tuple[int, int]) -> float:
        """Calculate face rotation angle in degrees"""
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        
        angle = np.degrees(np.arctan(dy / dx))
        return angle