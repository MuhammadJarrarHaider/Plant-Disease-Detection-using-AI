from tensorflow.keras.models import load_model
import cv2
import numpy as np
import json
import os

# Path to the fine-tuned model and class indices
model_path = "D:/AI/project/plant_disease_model_finetuned.h5"
class_indices_path = "D:/AI/project/class_indices.json"

# Load the fine-tuned model
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()
model = load_model(model_path)

# Load class indices
if not os.path.exists(class_indices_path):
    print(f"Class indices file not found at {class_indices_path}")
    exit()
with open(class_indices_path, "r") as json_file:
    class_indices = json.load(json_file)

# Invert class indices to map index to class name
index_to_class = {v: k for k, v in class_indices.items()}

# Path to the test image
test_image_path = "D:/AI/project/00e7c4b2-3005-4558-9cfa-235e356cb7a8___RS_Erly.B 7844.jpg"
if not os.path.exists(test_image_path):
    print(f"Test image not found at {test_image_path}")
    exit()

# Preprocess the test image
img = cv2.imread(test_image_path)
img_resized = cv2.resize(img, (224, 224)) / 255.0  # Resize and normalize
img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)  # Get the index of the predicted class
predicted_class = index_to_class[predicted_class_index]
confidence = prediction[0][predicted_class_index]  # Confidence of the prediction

# Display the results
print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
