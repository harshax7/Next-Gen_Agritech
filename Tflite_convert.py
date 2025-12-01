import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('best_disease_detection_model.h5')

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('tomato_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)