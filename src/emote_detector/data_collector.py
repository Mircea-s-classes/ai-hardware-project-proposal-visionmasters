"""
Data Collection Tool for Training Custom Poses
Collect samples of yourself doing different poses/gestures
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier


class PoseDataCollector:
    """Tool for collecting training data for pose classification"""
    
    def __init__(self, data_dir="pose_data"):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.detector = HolisticDetector(model_complexity=1)
        self.classifier = PoseClassifier()
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection settings
        self.current_pose = 0
        self.collected_samples = 0
        self.samples_per_pose = 100  # Collect 100 samples per pose
        self.auto_collect = False
        self.collection_delay = 5  # Collect every N frames
        self.frame_counter = 0
        
        # Pose labels - CUSTOMIZE THESE FOR YOUR PROJECT!
        self.pose_labels = {
            0: "Laughing",    # Hands on waist or hips
            1: "Yawning",     # Hand(s) covering mouth
            2: "Crying",      # Hands covering face/eyes
            3: "Taunting",    # Fists near face, taunting pose
        }
        
        # Collected data
        self.collected_data = []
        self.collected_labels = []
        
        self._print_instructions()
    
    def _print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("📸 Pose Data Collector")
        print("="*60)
        print("\nCollect training data for your custom poses!")
        print("\nControls:")
        for idx, name in self.pose_labels.items():
            print(f"  '{idx}' - Select pose: {name}")
        print(f"  'a' - Toggle auto-collection (ON/OFF)")
        print(f"  's' - Save collected data")
        print(f"  't' - Train model with collected data")
        print(f"  'l' - Load previously saved data")
        print(f"  'c' - Clear current collection")
        print(f"  'q' - Quit")
        print("\n💡 Tips:")
        print(f"   - Collect at least {self.samples_per_pose} samples per pose")
        print(f"   - Vary your position slightly while collecting")
        print(f"   - Make sure your full upper body is visible")
        print("="*60 + "\n")
    
    def collect_data(self):
        """Main data collection loop"""
        print("📷 Initializing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Trying camera 1...")
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        print("✅ Camera ready!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            self.frame_counter += 1
            
            # Detect landmarks
            results = self.detector.detect(frame)
            
            # Draw landmarks
            frame = self.detector.draw_landmarks(frame, results)
            
            # Get pose data
            landmark_data = self.detector.get_landmark_data(results)
            pose_landmarks = landmark_data.get('pose')
            
            # Auto-collect if enabled
            if (self.auto_collect and 
                pose_landmarks is not None and 
                self.frame_counter % self.collection_delay == 0):
                
                # Extract features and store
                features = self.classifier.extract_features(pose_landmarks)
                self.collected_data.append(features)
                self.collected_labels.append(self.current_pose)
                self.collected_samples += 1
                
                print(f"  ✓ Sample {self.collected_samples} for {self.pose_labels[self.current_pose]}")
                
                # Auto-advance to next pose if enough samples
                if self.collected_samples >= self.samples_per_pose:
                    print(f"\n✅ Completed {self.samples_per_pose} samples for {self.pose_labels[self.current_pose]}")
                    self.collected_samples = 0
                    self.current_pose = (self.current_pose + 1) % len(self.pose_labels)
                    
                    if self.current_pose == 0:
                        print("\n🎉 All poses collected! Press 't' to train or keep collecting.")
                        self.auto_collect = False
                    else:
                        print(f"📍 Now collecting: {self.pose_labels[self.current_pose]}")
            
            # Draw UI
            self._draw_ui(frame, pose_landmarks)
            
            cv2.imshow('Pose Data Collector', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('9'):
                pose_idx = key - ord('0')
                if pose_idx in self.pose_labels:
                    self.current_pose = pose_idx
                    self.collected_samples = 0
                    print(f"📍 Switched to: {self.pose_labels[self.current_pose]}")
            elif key == ord('a'):
                self.auto_collect = not self.auto_collect
                status = "ON 🟢" if self.auto_collect else "OFF 🔴"
                print(f"Auto-collection: {status}")
            elif key == ord('s'):
                self._save_data()
            elif key == ord('l'):
                self._load_data()
            elif key == ord('t'):
                self._train_model()
            elif key == ord('c'):
                self.collected_data = []
                self.collected_labels = []
                self.collected_samples = 0
                print("🗑️ Cleared all collected data")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
    
    def _draw_ui(self, frame, pose_landmarks):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (0, 0), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (400, 150), (255, 255, 255), 1)
        
        # Current pose
        cv2.putText(frame, f"Pose: {self.pose_labels[self.current_pose]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Sample count
        total = len(self.collected_data)
        cv2.putText(frame, f"Samples: {self.collected_samples}/{self.samples_per_pose} (Total: {total})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Auto-collect status
        status = "ON" if self.auto_collect else "OFF"
        color = (0, 255, 0) if self.auto_collect else (0, 0, 255)
        cv2.putText(frame, f"Auto-collect: {status}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Detection status
        if pose_landmarks is not None:
            cv2.putText(frame, "Pose: DETECTED", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Pose: NOT DETECTED", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Instructions at bottom
        cv2.putText(frame, "0-3: Select pose | a: Auto-collect | s: Save | t: Train | q: Quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visual indicator when collecting
        if self.auto_collect and pose_landmarks is not None:
            cv2.circle(frame, (w - 30, 30), 15, (0, 255, 0), -1)
    
    def _save_data(self):
        """Save collected data to files"""
        if len(self.collected_data) == 0:
            print("⚠️ No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save arrays
        features_file = self.data_dir / f"pose_features_{timestamp}.npy"
        labels_file = self.data_dir / f"pose_labels_{timestamp}.npy"
        
        np.save(features_file, np.array(self.collected_data))
        np.save(labels_file, np.array(self.collected_labels))
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "num_samples": len(self.collected_data),
            "pose_labels": self.pose_labels,
            "samples_per_pose": self.samples_per_pose,
            "collection_date": datetime.now().isoformat()
        }
        
        metadata_file = self.data_dir / f"pose_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as "latest" for easy loading
        np.save(self.data_dir / "pose_features_latest.npy", np.array(self.collected_data))
        np.save(self.data_dir / "pose_labels_latest.npy", np.array(self.collected_labels))
        
        print(f"\n✅ Data saved!")
        print(f"   Samples: {len(self.collected_data)}")
        print(f"   Location: {self.data_dir}")
    
    def _load_data(self):
        """Load previously saved data"""
        features_file = self.data_dir / "pose_features_latest.npy"
        labels_file = self.data_dir / "pose_labels_latest.npy"
        
        if not features_file.exists() or not labels_file.exists():
            print("⚠️ No saved data found!")
            return
        
        try:
            self.collected_data = np.load(features_file).tolist()
            self.collected_labels = np.load(labels_file).tolist()
            print(f"✅ Loaded {len(self.collected_data)} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _train_model(self):
        """Train the classifier with collected data"""
        if len(self.collected_data) < 20:
            print(f"⚠️ Need at least 20 samples to train (have {len(self.collected_data)})")
            return
        
        # Auto-save data before training
        print("\n💾 Auto-saving data before training...")
        self._save_data()
        
        X = np.array(self.collected_data)
        y = np.array(self.collected_labels)
        
        # Update classifier labels
        self.classifier.set_pose_labels(self.pose_labels)
        
        # Train
        print(f"\n🚀 Training with {len(X)} samples...")
        accuracy = self.classifier.train_model(X, y)
        
        print(f"\n✅ Model trained and saved!")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n📊 To generate charts, run: python train_model.py")


if __name__ == "__main__":
    collector = PoseDataCollector()
    collector.collect_data()

