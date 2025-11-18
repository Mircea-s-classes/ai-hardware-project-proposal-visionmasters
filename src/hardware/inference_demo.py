"""
Real-Time Inference Demo for Raspberry Pi + Coral Edge TPU
This script will be used when hardware is available
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

# Note: These imports will work on Raspberry Pi with Coral
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    import tflite_runtime.interpreter as tflite
    CORAL_AVAILABLE = True
except ImportError:
    print("⚠️  PyCoral not available. Install on Raspberry Pi:")
    print("   pip install pycoral tflite-runtime")
    CORAL_AVAILABLE = False
    import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.face_detection import FaceDetector, FacePreprocessor


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to Emote Mapping
EMOTION_EMOTES = {
    'Angry': ('angry.png', 'angry.mp3'),
    'Disgust': ('disgust.png', 'disgust.mp3'),
    'Fear': ('fear.png', 'fear.mp3'),
    'Happy': ('happy.png', 'happy.mp3'),
    'Sad': ('sad.png', 'sad.mp3'),
    'Surprise': ('surprise.png', 'surprise.mp3'),
    'Neutral': ('neutral.png', 'neutral.mp3')
}


class EmotionRecognizer:
    """Facial expression recognition with Edge TPU"""
    
    def __init__(self, model_path, use_edgetpu=True):
        """
        Initialize emotion recognizer
        
        Args:
            model_path: Path to TFLite model
            use_edgetpu: Whether to use Edge TPU acceleration
        """
        self.use_edgetpu = use_edgetpu and CORAL_AVAILABLE
        
        print(f"\n📦 Loading model: {model_path}")
        print(f"   Using Edge TPU: {self.use_edgetpu}")
        
        if self.use_edgetpu:
            # Load with Edge TPU
            self.interpreter = edgetpu.make_interpreter(str(model_path))
        else:
            # Load with TFLite Runtime or TensorFlow
            if CORAL_AVAILABLE:
                self.interpreter = tflite.Interpreter(model_path=str(model_path))
            else:
                self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Input dtype: {self.input_dtype}")
    
    def predict(self, face_image):
        """
        Predict emotion from face image
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Tuple of (emotion_label, confidence, all_probabilities)
        """
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], face_image)
        
        # Run inference
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Dequantize if INT8
        if self.output_details[0]['dtype'] == np.uint8:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)
        
        # Apply softmax if needed
        if np.max(output) > 1.0 or np.min(output) < 0.0:
            output = self._softmax(output)
        
        # Get prediction
        emotion_idx = np.argmax(output)
        emotion_label = EMOTION_LABELS[emotion_idx]
        confidence = output[emotion_idx]
        
        return emotion_label, confidence, output, inference_time
    
    @staticmethod
    def _softmax(x):
        """Apply softmax to get probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class EmoteDisplay:
    """Display Clash Royale emotes"""
    
    def __init__(self, emotes_dir):
        """
        Initialize emote display
        
        Args:
            emotes_dir: Directory containing emote images
        """
        self.emotes_dir = Path(emotes_dir)
        self.emote_cache = {}
        
        # Pre-load emote images
        for emotion, (image_file, _) in EMOTION_EMOTES.items():
            image_path = self.emotes_dir / 'images' / image_file
            if image_path.exists():
                emote = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                self.emote_cache[emotion] = emote
            else:
                print(f"⚠️  Emote not found: {image_path}")
        
        print(f"✅ Loaded {len(self.emote_cache)} emotes")
    
    def get_emote(self, emotion):
        """Get emote image for emotion"""
        return self.emote_cache.get(emotion)
    
    def overlay_emote(self, frame, emote, position=(10, 10), size=(100, 100)):
        """Overlay emote on frame"""
        if emote is None:
            return frame
        
        # Resize emote
        emote = cv2.resize(emote, size)
        
        x, y = position
        h, w = emote.shape[:2]
        
        # Check if emote has alpha channel
        if emote.shape[2] == 4:
            # Extract alpha channel
            alpha = emote[:, :, 3] / 255.0
            
            # Overlay with transparency
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * emote[:, :, c] +
                    (1 - alpha) * frame[y:y+h, x:x+w, c]
                )
        else:
            # No alpha, just paste
            frame[y:y+h, x:x+w] = emote
        
        return frame


def main():
    """Main demo application"""
    parser = argparse.ArgumentParser(description='Real-time emotion recognition demo')
    parser.add_argument('--model', type=str, default='models/model_int8_edgetpu.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--emotes-dir', type=str, default='data/emotes',
                       help='Path to emotes directory')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--no-edgetpu', action='store_true',
                       help='Disable Edge TPU acceleration')
    parser.add_argument('--display-fps', action='store_true',
                       help='Display FPS counter')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Real-Time Facial Expression Recognition")
    print("="*60)
    
    # Setup paths
    model_path = project_root / args.model
    emotes_dir = project_root / args.emotes_dir
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nPlease train and quantize the model first:")
        print("1. python src/model/train_baseline.py")
        print("2. python src/model/quantize_model.py")
        print("3. edgetpu_compiler models/model_int8.tflite")
        sys.exit(1)
    
    # Initialize components
    print("\n🚀 Initializing components...")
    face_detector = FaceDetector(min_detection_confidence=0.7)
    preprocessor = FacePreprocessor(target_size=(224, 224), normalize=False)
    recognizer = EmotionRecognizer(model_path, use_edgetpu=not args.no_edgetpu)
    
    # Load emotes if available
    emote_display = None
    if emotes_dir.exists():
        emote_display = EmoteDisplay(emotes_dir)
    else:
        print(f"⚠️  Emotes directory not found: {emotes_dir}")
    
    # Open camera
    print(f"\n📷 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully")
    print("\n" + "="*60)
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*60 + "\n")
    
    # Performance tracking
    fps_history = []
    last_emotion = None
    last_confidence = 0.0
    
    try:
        while True:
            frame_start = time.perf_counter()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                # Get first face
                face = faces[0]
                x, y, w, h = face['bbox']
                
                # Draw face box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Crop and preprocess
                face_img = face_detector.crop_face(frame, face['bbox'])
                if face_img is not None:
                    # Preprocess for model
                    if recognizer.input_dtype == np.uint8:
                        face_input = preprocessor.preprocess_for_int8(face_img)
                    else:
                        face_input = preprocessor.preprocess(face_img)
                    
                    # Predict emotion
                    emotion, confidence, probs, inference_time = recognizer.predict(face_input)
                    
                    # Update tracking
                    last_emotion = emotion
                    last_confidence = confidence
                    
                    # Display emotion
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(display_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display inference time
                    cv2.putText(display_frame, f"Inference: {inference_time:.1f}ms",
                               (10, display_frame.shape[0] - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display emote
                    if emote_display:
                        emote = emote_display.get_emote(emotion)
                        if emote is not None:
                            display_frame = emote_display.overlay_emote(
                                display_frame, emote,
                                position=(display_frame.shape[1] - 120, 10),
                                size=(100, 100)
                            )
            else:
                # No face detected
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_time = time.perf_counter() - frame_start
            fps = 1.0 / frame_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            if args.display_fps:
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                           (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Emotion Recognition Demo', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        
        # Print statistics
        if fps_history:
            print("\n" + "="*60)
            print("Session Statistics:")
            print("="*60)
            print(f"Average FPS: {np.mean(fps_history):.2f}")
            print(f"Min FPS: {np.min(fps_history):.2f}")
            print(f"Max FPS: {np.max(fps_history):.2f}")
            if last_emotion:
                print(f"Last emotion: {last_emotion} ({last_confidence:.2f})")
            print("="*60 + "\n")


if __name__ == '__main__':
    main()

