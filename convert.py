from ultralytics import YOLO

# Load your trained model
model = YOLO("tomato_trained_model.pt")

# Export to TFLite (optimized for Pi)
model.export(
    format="tflite",
    imgsz=320   # smaller = faster
)

print("✅ Conversion complete!")