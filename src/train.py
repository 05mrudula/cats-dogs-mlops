# ==========================================
# CNN Model Training
# ==========================================

# Import JSON for saving training history and evaluation metrics
import json

# Import Path for creating directories and handling file paths
from pathlib import Path

# Import TensorFlow for building and training the CNN
import tensorflow as tf

# Import the prepared training, validation, and test datasets
from preprocess import (
    train_dataset,
    validation_dataset,
    test_dataset
)


# ==========================================
# Output Directory Configuration
# ==========================================

# Define the directory where trained models will be stored
MODEL_DIR = Path("models")

# Define the directory where training reports will be stored
REPORT_DIR = Path("reports")

# Create the directories if they do not already exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Model Configuration
# ==========================================

# Define the dimensions of the input images
IMAGE_SIZE = (224, 224)

# Define the number of colour channels in an RGB image
CHANNELS = 3

# Define the number of training epochs
EPOCHS = 10


# ==========================================
# CNN Model Definition
# ==========================================

# Create a Sequential CNN model
model = tf.keras.Sequential([

    # Explicitly define the input shape for the CNN
    tf.keras.Input(
        shape=(*IMAGE_SIZE, CHANNELS)
    ),

    # First convolutional layer extracts basic image features
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduce the spatial dimensions of the feature maps
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolutional layer learns more complex features
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Further reduce the spatial dimensions
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolutional layer learns higher-level image features
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    # Reduce the feature-map dimensions again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert the extracted feature maps into a one-dimensional vector
    tf.keras.layers.Flatten(),

    # Fully connected layer used for classification
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout helps reduce overfitting during training
    tf.keras.layers.Dropout(
        0.5
    ),

    # Binary output layer for cat/dog classification
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# Model Compilation
# ==========================================

# Configure the model for binary classification
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Display Model Architecture
# ==========================================

# Print the model architecture and parameter counts
model.summary()


# ==========================================
# Training Configuration
# ==========================================

# Save the best model based on validation loss
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_DIR / "baseline_cnn.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

# Stop training early if validation performance stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min",
    restore_best_weights=True,
    verbose=1
)


# ==========================================
# Model Training
# ==========================================

# Train the CNN using the prepared training and validation datasets
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint,
        early_stopping
    ]
)


# ==========================================
# Save Training History
# ==========================================

# Convert TensorFlow training history into a regular Python dictionary
training_history = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

# Save the training history as a JSON file
with open(
    REPORT_DIR / "training_history.json",
    "w"
) as file:

    json.dump(
        training_history,
        file,
        indent=4
    )


# ==========================================
# Load Best Model
# ==========================================

# Load the best-performing model saved during training
best_model = tf.keras.models.load_model(
    MODEL_DIR / "baseline_cnn.keras"
)


# ==========================================
# Test Set Evaluation
# ==========================================

# Evaluate the best model using the unseen test dataset
test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)

# Display the final test results
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# ==========================================
# Save Test Metrics
# ==========================================

# Store the final evaluation metrics in a dictionary
test_metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

# Save the test metrics as a JSON file
with open(
    REPORT_DIR / "test_metrics.json",
    "w"
) as file:

    json.dump(
        test_metrics,
        file,
        indent=4
    )


# ==========================================
# Training Completion Message
# ==========================================

# Confirm that the model and experiment results were saved successfully
print("Baseline model saved to:", MODEL_DIR / "baseline_cnn.keras")
print("Training history saved to:", REPORT_DIR / "training_history.json")
print("Test metrics saved to:", REPORT_DIR / "test_metrics.json")