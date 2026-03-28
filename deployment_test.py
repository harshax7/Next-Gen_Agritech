"""
Real-Time Tomato Detection and Ripeness Classification System
YOLO ONNX → TFLite pipeline
"""

import os
import cv2
import numpy as np
import time
import ast

try:
    from tflite_runtime.interpreter import Interpreter  # Lightweight (RPi)
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # Fallback (PC)

import onnxruntime as ort

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_ONNX_PATH            = "./tomato_trained_model.onnx"
YOLO_CONF_THRESHOLD       = 0.5
YOLO_IOU_THRESHOLD        = 0.45

TFLITE_MODEL_PATH         = "./tomato_disease_model.tflite"
IMAGE_SIZE                = (224, 224)
CLASS_LABELS              = ['Ripe', 'Unripe']

# Auto-detected at runtime from the ONNX model — do not edit manually
YOLO_INPUT_SIZE           = None

# ============================================================================
# YOLO ONNX HELPER FUNCTIONS
# ============================================================================

def letterbox(img, new_shape=(640, 640)):
    """Resize + pad image to square while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = new_shape[0] - nh
    pad_w = new_shape[1] - nw
    top, left = pad_h // 2, pad_w // 2

    img_padded = cv2.copyMakeBorder(
        img_resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_padded, scale, (left, top)


def preprocess_yolo(frame, input_size):
    """Convert BGR frame to YOLO ONNX input tensor."""
    img, scale, pad = letterbox(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # Add batch dim
    return img, scale, pad


def postprocess_yolo(outputs, scale, pad, orig_shape, conf_thresh, iou_thresh):
    """
    Parse raw YOLO ONNX output into bounding boxes.
    YOLOv8 output shape: [1, num_classes+4, num_anchors]
    """
    predictions = outputs[0]  # shape: (1, 4+nc, 8400)

    if predictions.ndim == 3:
        predictions = predictions[0]        # (4+nc, 8400)
    predictions = predictions.T             # (8400, 4+nc)

    boxes_xywh   = predictions[:, :4]
    class_scores = predictions[:, 4:]

    confidences = np.max(class_scores, axis=1)
    class_ids   = np.argmax(class_scores, axis=1)

    mask = confidences >= conf_thresh
    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert cx,cy,w,h → x1,y1,x2,y2 (in letterbox space)
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Remove letterbox padding and scale back to original frame coords
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Clamp to frame
    h, w = orig_shape[:2]
    x1 = np.clip(x1, 0, w).astype(int)
    y1 = np.clip(y1, 0, h).astype(int)
    x2 = np.clip(x2, 0, w).astype(int)
    y2 = np.clip(y2, 0, h).astype(int)

    # NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(float)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_thresh,
        iou_thresh
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'bbox':       [x1[i], y1[i], x2[i], y2[i]],
                'confidence': float(confidences[i]),
                'class_id':   int(class_ids[i])
            })
    return detections


# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class TomatoDetectionSystem:

    def __init__(self):
        self._load_yolo_onnx()
        self._load_tflite()
        self.frame_count = 0
        self.total_time  = 0.0

    # -------------------------------------------------------------------------
    def _load_yolo_onnx(self):
        if not os.path.exists(YOLO_ONNX_PATH):
            raise FileNotFoundError(f"YOLO ONNX model not found: {YOLO_ONNX_PATH}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.yolo_session = ort.InferenceSession(
            YOLO_ONNX_PATH,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name

        # Auto-detect input size from model (avoids dimension mismatch errors)
        global YOLO_INPUT_SIZE
        input_shape      = self.yolo_session.get_inputs()[0].shape
        YOLO_INPUT_SIZE  = int(input_shape[2])
        print(f"[YOLO ONNX] Loaded: {YOLO_ONNX_PATH}")
        print(f"[YOLO ONNX] Auto-detected input size: {YOLO_INPUT_SIZE}")

        # Read class names from ONNX metadata if available
        meta = self.yolo_session.get_modelmeta().custom_metadata_map
        if 'names' in meta:
            self.yolo_class_names = ast.literal_eval(meta['names'])
        else:
            self.yolo_class_names = {0: 'tomato'}
        print(f"[YOLO ONNX] Classes: {self.yolo_class_names}")

    # -------------------------------------------------------------------------
    def _load_tflite(self):
        if not os.path.exists(TFLITE_MODEL_PATH):
            raise FileNotFoundError(f"TFLite model not found: {TFLITE_MODEL_PATH}")

        self.interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[TFLite] Loaded | Input shape: {self.input_details[0]['shape']} | Classes: {CLASS_LABELS}")

    # -------------------------------------------------------------------------
    def detect_with_yolo(self, frame):
        """Run YOLO ONNX inference, return tomato detections."""
        input_tensor, scale, pad = preprocess_yolo(frame, YOLO_INPUT_SIZE)

        outputs = self.yolo_session.run(None, {self.yolo_input_name: input_tensor})
        all_detections = postprocess_yolo(
            outputs, scale, pad, frame.shape,
            YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
        )

        tomato_detections = []
        for det in all_detections:
            class_name = self.yolo_class_names.get(det['class_id'], 'unknown')
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            if 'tomato' in class_name.lower():
                det['class'] = class_name
                tomato_detections.append(det)
            else:
                # Draw non-tomato detections in red and skip TFLite
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Not Tomato ({conf:.0%})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return tomato_detections

    # -------------------------------------------------------------------------
    def classify_with_tflite(self, crop):
        """Classify cropped tomato region as Ripe or Unripe."""
        try:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0

            dtype = self.input_details[0]['dtype']
            if dtype == np.float32:
                img = img.astype(np.float32)
            elif dtype == np.uint8:
                img = (img * 255).astype(np.uint8)

            img = np.expand_dims(img, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()

            probs     = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            class_idx = int(np.argmax(probs))
            return CLASS_LABELS[class_idx], float(probs[class_idx]), probs

        except Exception as e:
            print(f"[TFLite] Classification error: {e}")
            return None, 0.0, None

    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        start = time.time()

        # STEP 1: Detect tomatoes via YOLO ONNX
        tomato_detections = self.detect_with_yolo(frame)

        # STEP 2: Classify each confirmed tomato crop via TFLite
        for det in tomato_detections:
            x1, y1, x2, y2 = det['bbox']

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ripeness, conf, probs = self.classify_with_tflite(crop)

            if ripeness:
                label     = f"{ripeness} ({conf:.0%})"
                box_color = (0, 255, 0) if ripeness == 'Ripe' else (0, 165, 255)
            else:
                label     = "Tomato (Error)"
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
        self.total_time  += elapsed
        self.frame_count += 1
        fps     = 1.0 / elapsed if elapsed > 0 else 0
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

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
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

            processed = system.process_frame(frame)
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
        if system.frame_count > 0:
            print(f"\nFrames: {system.frame_count} | Avg FPS: {system.frame_count/system.total_time:.2f}")


if __name__ == "__main__":
    run(camera_index=1)