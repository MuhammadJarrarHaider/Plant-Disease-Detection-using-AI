import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

dataset_path = "D:/Artificial Intelligence/project/PlantVillage"

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=30,  # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip horizontally
    fill_mode="nearest",  # Fill empty pixels
    validation_split=0.2  # Split into 80% training and 20% validation
)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=32,
    class_mode="categorical",
    subset="training"  # Training subset
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"  # Validation subset
)

# Define the ResNet50 model
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = True  # Unfreeze the base model for fine-tuning

# Fine-tune specific layers (optional: here, unfreeze only the last 50 layers)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduce dimensionality
    Dense(128, activation="relu"),  # Fully connected layer
    Dropout(0.3),  # Prevent overfitting
    Dense(len(train_data.class_indices), activation="softmax")  # Output layer
])

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,  # Fewer epochs for fine-tuning
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=val_data.samples // val_data.batch_size
)

# Evaluate the model
val_data.reset()
predictions = model.predict(val_data, steps=val_data.samples // val_data.batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes[:len(y_pred)]

# Generate evaluation metrics
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Save the fine-tuned model
fine_tuned_model_path = "D:/Artificial Intelligence/project/plant_disease_model_finetuned.h5"
model.save(fine_tuned_model_path)
print(f"Fine-tuned model saved at {fine_tuned_model_path}")
