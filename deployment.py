"""
Real-Time Tomato Detection and Ripeness Classification System
Combines YOLOv8 for object detection and TensorFlow for ripeness classification
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
import tensorflow as tf 
# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# YOLO Configuration
YOLO_MODEL_PATH = r'C:\Users\harsh\OneDrive\Desktop\Agrolens\tomato_trained_model.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.7

# TensorFlow Lite Configuration
TFLITE_MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\Agrolens\tomato_disease_model.tflite"
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ['Ripe', 'Unripe']

# Display Configuration
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FPS_DISPLAY = True

# Color scheme for bounding boxes (BGR format)
COLOR_RIPE = (0, 255, 0)      # Green
COLOR_UNRIPE = (0, 165, 255)  # Orange
COLOR_NOT_TOMATO = (0, 0, 255)  # Red

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

class TomatoDetectionSystem:
    """Integrated system for tomato detection and ripeness classification"""
    
    def __init__(self):
        """Initialize both YOLO and TensorFlow Lite models"""
        print("=" * 60)
        print("Initializing Tomato Detection & Classification System")
        print("=" * 60)
        
        # Set device for YOLO
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Load TensorFlow Lite model
        self._load_tflite_model()
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
        print("\n✓ System initialized successfully!")
        print("=" * 60)
    
    def _load_yolo_model(self):
        """Load YOLOv8 model for tomato detection"""
        print("\n[YOLO] Loading YOLOv8 model...")
        
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
        
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Model loaded successfully from {YOLO_MODEL_PATH}")
        print(f"[YOLO] Available classes: {list(self.yolo_model.names.values())}")
    
    def _load_tflite_model(self):
        """Load TensorFlow Lite model for ripeness classification"""
        print("\n[TFLite] Loading classification model...")
        
        if not os.path.exists(TFLITE_MODEL_PATH):
            raise FileNotFoundError(f"TFLite model not found at: {TFLITE_MODEL_PATH}")
        
        # Initialize TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"[TFLite] Model loaded successfully from {TFLITE_MODEL_PATH}")
        print(f"[TFLite] Input shape: {self.input_details[0]['shape']}")
        print(f"[TFLite] Output shape: {self.output_details[0]['shape']}")
        print(f"[TFLite] Classes: {CLASS_LABELS}")
    
    # ========================================================================
    # YOLO DETECTION METHODS
    # ========================================================================
    
    def detect_tomatoes(self, frame):
        """
        Detect tomatoes in frame using YOLOv8
        
        Args:
            frame: Input frame from camera
            
        Returns:
            list of dict: Detected tomatoes with bounding boxes and confidence
        """
        # Run YOLO detection
        results = self.yolo_model(frame, device=self.device, verbose=False)
        
        detections = []
        
        # Process YOLO results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    # Only process detections above confidence threshold
                    if confidence >= YOLO_CONFIDENCE_THRESHOLD:
                        # Get bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'is_tomato': 'tomato' in class_name.lower()
                        }
                        
                        detections.append(detection)
        
        return detections
    
    # ========================================================================
    # TENSORFLOW LITE CLASSIFICATION METHODS
    # ========================================================================
    
    def classify_ripeness(self, cropped_image):
        """
        Classify tomato ripeness using TensorFlow Lite model
        
        Args:
            cropped_image: Cropped tomato region from frame
            
        Returns:
            tuple: (predicted_class, confidence, probabilities)
        """
        try:
            # Preprocess image for TFLite model
            img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # Normalize to [0, 1]
            
            # Check input data type and convert accordingly
            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.float32:
                img = img.astype(np.float32)
            elif input_dtype == np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Get predicted class and confidence
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx]
            predicted_class = CLASS_LABELS[class_idx]
            
            return predicted_class, confidence, predictions
            
        except Exception as e:
            print(f"[TFLite] Error during classification: {e}")
            return None, 0.0, None
    
    # ========================================================================
    # INTEGRATED PROCESSING PIPELINE
    # ========================================================================
    
    def process_frame(self, frame):
        """
        Main processing pipeline: Detect tomatoes and classify ripeness
        
        Args:
            frame: Input frame from camera
            
        Returns:
            frame: Annotated frame with bounding boxes and labels
        """
        start_time = time.time()
        
        # Step 1: Detect tomatoes using YOLO
        detections = self.detect_tomatoes(frame)
        
        # Step 2: Process each detection
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Ensure valid bounding box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if detection['is_tomato']:
                # Crop tomato region
                cropped_tomato = frame[y1:y2, x1:x2]
                
                # Check if crop is valid
                if cropped_tomato.size > 0:
                    # Step 3: Classify ripeness using TensorFlow
                    ripeness, confidence, probs = self.classify_ripeness(cropped_tomato)
                    
                    if ripeness:
                        # Determine label and color based on ripeness
                        if ripeness == 'Ripe':
                            label = f"Ripe Tomato {confidence:.2%}"
                            color = COLOR_RIPE
                        else:
                            label = f"Unripe Tomato {confidence:.2%}"
                            color = COLOR_UNRIPE
                        
                        # Draw bounding box and label
                        self._draw_detection(frame, x1, y1, x2, y2, label, color)
                        
                        # Display probabilities (optional)
                        if probs is not None:
                            prob_text = f"R:{probs[0]:.2f} U:{probs[1]:.2f}"
                            cv2.putText(frame, prob_text, (x1, y2 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # Classification failed
                        label = f"Tomato (Classification Error)"
                        self._draw_detection(frame, x1, y1, x2, y2, label, COLOR_UNRIPE)
                else:
                    # Invalid crop
                    label = "Tomato (Invalid Crop)"
                    self._draw_detection(frame, x1, y1, x2, y2, label, COLOR_UNRIPE)
            else:
                # Not a tomato
                label = f"Not a Tomato"
                self._draw_detection(frame, x1, y1, x2, y2, label, COLOR_NOT_TOMATO)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.total_time += processing_time
        self.frame_count += 1
        
        # Display FPS
        if FPS_DISPLAY:
            fps = 1.0 / processing_time if processing_time > 0 else 0
            avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f} | Avg: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _draw_detection(self, frame, x1, y1, x2, y2, label, color):
        """Draw bounding box and label on frame"""
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1, label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y1_label - label_size[1] - 10),
                     (x1 + label_size[0], y1_label + baseline - 10),
                     color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1_label - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ============================================================================
# CAMERA CAPTURE AND MAIN LOOP
# ============================================================================

def run_realtime_detection(camera_index=0):
    """
    Run real-time tomato detection and classification
    
    Args:
        camera_index: Camera device index (0 for default webcam)
    """
    print("\n" + "=" * 60)
    print("Starting Real-Time Detection System")
    print("=" * 60)
    
    # Initialize detection system
    system = TomatoDetectionSystem()
    
    # Open camera
    print(f"\nOpening camera (index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera opened successfully")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("=" * 60 + "\n")
    
    frame_number = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Process frame through detection and classification pipeline
            processed_frame = system.process_frame(frame)
            
            # Display processed frame
            cv2.imshow('Tomato Detection & Ripeness Classification', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{frame_number:04d}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")
            
            frame_number += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Display statistics
        print("\n" + "=" * 60)
        print("Session Statistics:")
        print(f"  Total frames processed: {system.frame_count}")
        print(f"  Average FPS: {system.frame_count / system.total_time:.2f}")
        print(f"  Total time: {system.total_time:.2f} seconds")
        print("=" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run real-time detection with default webcam (index 0)
    # Change camera_index if using external camera (1, 2, etc.)
    run_realtime_detection(camera_index=0)
    
    # For testing with a video file instead of webcam:
    # Uncomment the following lines and comment out the line above
    # video_path = "path/to/your/video.mp4"
    # run_realtime_detection(camera_index=video_path)