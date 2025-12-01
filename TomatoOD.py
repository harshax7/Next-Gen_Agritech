from ultralytics import YOLO
import torch
import os
from pathlib import Path

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define paths
base_model_path = r'C:\Users\harsh\OneDrive\Desktop\Agrolens\yolov8n.pt'  # Base YOLOv8 model
trained_model_path = r'C:\Users\harsh\OneDrive\Desktop\Agrolens\tomato_trained_model.pt'  # Your trained model

# Initialize model variable
model = None

# Check if trained model already exists
if os.path.exists(trained_model_path):
    print("Trained model found! Loading existing model...")
    model = YOLO(trained_model_path)
    print("Model loaded successfully!")
    
else:
    print("No trained model found. Starting training...")
    
    # Load base model and train
    model = YOLO('yolov8n.pt')  # This will download if not present
    
    results = model.train(
        data=r'YOLOV8\tomato.yaml',  # Your dataset config
        epochs=50,
        batch=8,
        device=device,
        name='tomato_detection'  # Custom run name
    )
    
    print("Training completed!")
    
    # Save the trained model with custom name
    model.save(trained_model_path)
    print(f"Model saved as {trained_model_path}")

# Function to predict if an image contains tomato
def predict_tomato(image_path, confidence_threshold=0.5):
    """
    Predict if an image contains tomato
    
    Args:
        image_path (str): Path to the image file
        confidence_threshold (float): Minimum confidence for detection
    
    Returns:
        dict: Prediction results with tomato detection status
    """
    # Run prediction
    results = model(image_path, device=device)
    
    # Initialize result dictionary
    prediction_result = {
        'is_tomato_detected': False,
        'detections': [],
        'highest_confidence': 0.0
    }
    
    # Process results
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Check if it's a tomato detection above threshold
                if confidence >= confidence_threshold:
                    prediction_result['detections'].append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': box.xywh[0].tolist()  # x, y, width, height
                    })
                    
                    # Update highest confidence
                    if confidence > prediction_result['highest_confidence']:
                        prediction_result['highest_confidence'] = confidence
                    
                    # Assuming 'tomato' is in your class names
                    if 'tomato' in class_name.lower():
                        prediction_result['is_tomato_detected'] = True
    
    return prediction_result

# Example usage for single image prediction
def test_single_image(image_path):
    """Test the model on a single image"""
    print(f"\nPredicting on image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Make prediction
    result = predict_tomato(image_path)
    
    # Display results
    print(f"Tomato detected: {result['is_tomato_detected']}")
    print(f"Highest confidence: {result['highest_confidence']:.3f}")
    print(f"Total detections: {len(result['detections'])}")
    
    if result['detections']:
        print("\nDetection details:")
        for i, detection in enumerate(result['detections']):
            print(f"  {i+1}. Class: {detection['class']}, Confidence: {detection['confidence']:.3f}")
    
    return result

# Main execution
if __name__ == "__main__":
    # Example usage:
    # Replace with your actual image path
    test_image_path = r'C:\Users\harsh\OneDrive\Desktop\Agrolens\testing\test1.jpg'
    
    # Uncomment the line below to test with your image
    result = test_single_image(r"C:\Users\harsh\OneDrive\Desktop\Agrolens\testing\test1.jpg")
    
    # You can also use the model directly for visualization:
    # results = model(test_image_path)
    # results[0].show()  # Display the image with detections
    # results[0].save('output_with_detections.jpg')  # Save the annotated image
    
    print("Model is ready for predictions!")
    print(f"Available classes: {list(model.names.values())}")