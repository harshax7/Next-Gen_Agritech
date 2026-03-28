"""
Real-Time Tomato Detection and Ripeness Classification System
YOLO → TFLite pipeline (no color definitions)
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import tensorflow as tf

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_MODEL_PATH = r'./tomato_trained_model.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5

TFLITE_MODEL_PATH = "./tomato_disease_model.tflite"
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ['Ripe', 'Unripe']

# ============================================================================
# SYSTEM CLASS
# ============================================================================

class TomatoDetectionSystem:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self._load_yolo_model()
        self._load_tflite_model()
        self.frame_count = 0
        self.total_time = 0

    def _load_yolo_model(self):
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Loaded | Classes: {list(self.yolo_model.names.values())}")

    def _load_tflite_model(self):
        if not os.path.exists(TFLITE_MODEL_PATH):
            raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL_PATH}")
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[TFLite] Loaded | Input: {self.input_details[0]['shape']} | Classes: {CLASS_LABELS}")

    # -------------------------------------------------------------------------
    # STEP 1: Send raw frame directly to YOLO
    # -------------------------------------------------------------------------
    def detect_with_yolo(self, frame):
        """
        Send the raw frame directly to YOLO.
        Returns only detections that are tomatoes and above confidence threshold.
        """
        results = self.yolo_model(frame, device=self.device, verbose=False)

        tomato_detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]

                if confidence < YOLO_CONFIDENCE_THRESHOLD:
                    continue  # Skip low-confidence detections

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                if 'tomato' in class_name.lower():
                    width  = x2 - x1
                    height = y2 - y1

                    '''# Filter 1: must be reasonably sized
                    # too small = noise, too large = probably not a tomato
                    if width < 20 or height < 20:
                        continue
                    if width > 400 or height > 400:
                        continue

                    # Filter 2: must be roughly circular (tomatoes are round)
                    # aspect ratio close to 1.0 means width ≈ height
                    aspect_ratio = width / height
                    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                        continue'''

                    tomato_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'yolo_confidence': confidence,
                        'class': class_name
                    })
                else:
                    # Not a tomato — draw and skip TFLite
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Not Tomato ({confidence:.0%})",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return tomato_detections

    # -------------------------------------------------------------------------
    # STEP 2: Send YOLO-confirmed tomato crops to TFLite
    # -------------------------------------------------------------------------
    def classify_with_tflite(self, cropped_tomato):
        """
        Send cropped tomato region (from YOLO bbox) to TFLite for Ripe/Unripe classification.
        """
        try:
            img = cv2.cvtColor(cropped_tomato, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0

            input_dtype = self.input_details[0]['dtype']
            if input_dtype == np.float32:
                img = img.astype(np.float32)
            elif input_dtype == np.uint8:
                img = (img * 255).astype(np.uint8)

            img = np.expand_dims(img, axis=0)  # Add batch dimension

            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()

            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            class_idx = np.argmax(predictions)

            return CLASS_LABELS[class_idx], float(predictions[class_idx]), predictions

        except Exception as e:
            print(f"[TFLite] Classification error: {e}")
            return None, 0.0, None

    # ------------------------------------------------------------------------
    # MAIN PIPELINE: frame → YOLO → TFLite → annotated frame
    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        start = time.time()

        # STEP 1: Send frame directly to YOLO
        tomato_detections = self.detect_with_yolo(frame)

        # STEP 2: For each confirmed tomato, send crop to TFLite
        for det in tomato_detections:
            x1, y1, x2, y2 = det['bbox']

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # STEP 2: TFLite ripeness classification
            ripeness, conf, probs = self.classify_with_tflite(crop)

            if ripeness:
                label = f"{ripeness} ({conf:.0%})"
                box_color = (0, 255, 0) if ripeness == 'Ripe' else (0, 165, 255)
            else:
                label = "Tomato (Error)"
                box_color = (128, 128, 128)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw label background + text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw, y1), box_color, cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show raw probabilities below box
            if probs is not None:
                prob_text = f"Ripe:{probs[0]:.2f}  Unripe:{probs[1]:.2f}"
                cv2.putText(frame, prob_text, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1)

        # FPS overlay
        elapsed = time.time() - start
        self.total_time += elapsed
        self.frame_count += 1
        fps = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = self.frame_count / self.total_time

        cv2.putText(frame, f"FPS: {fps:.1f}  Avg: {avg_fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tomatoes detected: {len(tomato_detections)}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


# ============================================================================
# MAIN LOOP
# ============================================================================

def run(camera_index=1):
    system = TomatoDetectionSystem()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Running — press 'q' to quit, 's' to save frame")
    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # ──────────────────────────────────────────
            # Raw frame sent directly to YOLO here ↓
            processed = system.process_frame(frame)
            # ──────────────────────────────────────────

            cv2.imshow('Tomato Detection | Ripeness Classification', processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"saved_frame_{frame_num:04d}.jpg"
                cv2.imwrite(fname, processed)
                print(f"Saved: {fname}")

            frame_num += 1

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFrames: {system.frame_count} | Avg FPS: {system.frame_count/system.total_time:.2f}")


if __name__ == "__main__":
    run(camera_index=1)