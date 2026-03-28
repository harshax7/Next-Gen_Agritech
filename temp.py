import onnxruntime as ort

session = ort.InferenceSession("tomato_trained_model.onnx")
input_shape = session.get_inputs()[0].shape

print("Input shape:", input_shape)