"""
Face Detection and Preprocessing Utilities
Uses MediaPipe for efficient face detection on edge devices
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List


class FaceDetector:
    """Face detector using MediaPipe"""
    
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initialize face detector
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            model_selection: 0 for short-range (2m), 1 for full-range (5m)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        print(f"✅ FaceDetector initialized (confidence={min_detection_confidence})")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face detection dictionaries with bbox and confidence
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to pixel coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Get confidence
                confidence = detection.score[0]
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'detection': detection
                })
        
        return faces
    
    def draw_faces(self, image: np.ndarray, faces: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: Input image
            faces: List of face detections
            
        Returns:
            Image with drawn bounding boxes
        """
        output = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{confidence:.2f}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from image with padding
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            padding: Padding ratio around face
            
        Returns:
            Cropped face image or None if invalid
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        return face
    
    def close(self):
        """Release resources"""
        self.face_detection.close()


class FacePreprocessor:
    """Preprocess face images for model inference"""
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
        
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model
        
        Args:
            face: Input face image (BGR)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Resize
        face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize
        if self.normalize:
            face = face.astype(np.float32) / 255.0
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def preprocess_for_int8(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face for INT8 quantized model
        
        Args:
            face: Input face image (BGR)
            
        Returns:
            Preprocessed image in uint8 format
        """
        # Resize
        face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Keep as uint8 (0-255 range)
        face = face.astype(np.uint8)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face


def test_face_detection():
    """Test face detection with webcam"""
    print("Testing Face Detection...")
    print("Press 'q' to quit")
    
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw faces
        output = detector.draw_faces(frame, faces)
        
        # Show count
        cv2.putText(output, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("✅ Test complete!")


if __name__ == '__main__':
    test_face_detection()

