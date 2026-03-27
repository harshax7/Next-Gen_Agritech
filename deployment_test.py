from ultralytics import YOLO

model = YOLO("tomato_trained_model.pt")
model.info()