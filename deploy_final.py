
print("STARTING SCRIPT...")
import cv2
import numpy as np

# =========================
# LOAD INTERPRETER
# =========================
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using TensorFlow")
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("Using tflite-runtime")

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "model.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

print("Model input:", input_shape)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()

    # =========================
    # PREPROCESS
    # =========================
    img = cv2.resize(frame, (width, height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # =========================
    # INFERENCE
    # =========================
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    print("Output shape:", output.shape)  # DEBUG

    # =========================
    # FIX SHAPE
    # =========================
    output = output.T   # (2100, 5)

    # =========================
    # POSTPROCESS
    # =========================
    for det in output:
        x, y, w, h, conf = det

        if conf > CONF_THRESHOLD:
            x1 = int((x - w / 2) * orig.shape[1])
            y1 = int((y - h / 2) * orig.shape[0])
            x2 = int((x + w / 2) * orig.shape[1])
            y2 = int((y + h / 2) * orig.shape[0])

            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig, f"Tomato {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow("Tomato Detection", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()