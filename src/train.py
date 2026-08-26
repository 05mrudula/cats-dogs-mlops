# ==========================================
# CNN Model Training with MLflow Tracking
# ==========================================

# Import JSON for saving training history and evaluation metrics
import json

# Import Path for creating directories and handling file paths
from pathlib import Path

# Import TensorFlow for building and training the CNN
import tensorflow as tf

# Import MLflow for experiment tracking
import mlflow
import mlflow.tensorflow

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

# Define the batch size
BATCH_SIZE = 32

# Define the optimizer
OPTIMIZER = "adam"

# Define the loss function
LOSS_FUNCTION = "binary_crossentropy"


# ==========================================
# MLflow Configuration
# ==========================================

# Define the MLflow experiment name
mlflow.set_experiment("cats-dogs-baseline-cnn")


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
    optimizer=OPTIMIZER,
    loss=LOSS_FUNCTION,
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
# Start MLflow Run
# ==========================================

with mlflow.start_run() as run:

    # Display the MLflow run ID
    print("\nMLflow Run ID:", run.info.run_id)

    # ------------------------------------------
    # Log Model Parameters
    # ------------------------------------------

    mlflow.log_params({
        "image_width": IMAGE_SIZE[0],
        "image_height": IMAGE_SIZE[1],
        "channels": CHANNELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "loss_function": LOSS_FUNCTION,
        "dropout": 0.5,
        "conv_layers": 3
    })

    # ------------------------------------------
    # Model Training
    # ------------------------------------------

    print("\nStarting model training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint,
            early_stopping
        ]
    )

    # ------------------------------------------
    # Log Training Metrics
    # ------------------------------------------

    # Log all metrics collected during training
    for epoch in range(len(history.history["loss"])):

        mlflow.log_metrics(
            {
                "train_loss": float(
                    history.history["loss"][epoch]
                ),
                "train_accuracy": float(
                    history.history["accuracy"][epoch]
                ),
                "validation_loss": float(
                    history.history["val_loss"][epoch]
                ),
                "validation_accuracy": float(
                    history.history["val_accuracy"][epoch]
                )
            },
            step=epoch
        )

    # ------------------------------------------
    # Save Training History
    # ------------------------------------------

    # Convert TensorFlow training history into
    # a regular Python dictionary
    training_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    # Save the training history as a JSON file
    training_history_path = REPORT_DIR / "training_history.json"

    with open(
        training_history_path,
        "w"
    ) as file:

        json.dump(
            training_history,
            file,
            indent=4
        )

    # ------------------------------------------
    # Log Training History to MLflow
    # ------------------------------------------

    mlflow.log_artifact(
        str(training_history_path),
        artifact_path="reports"
    )

    # ------------------------------------------
    # Log Model to MLflow
    # ------------------------------------------

    # Load the best-performing model saved by
    # ModelCheckpoint
    best_model = tf.keras.models.load_model(
        MODEL_DIR / "baseline_cnn.keras"
    )

    # Log the trained model as an MLflow artifact
    mlflow.tensorflow.log_model(
        best_model,
        name="baseline_cnn"
    )

    # ------------------------------------------
    # Test Set Evaluation
    # ------------------------------------------

    print("\nEvaluating model on test dataset...\n")

    test_loss, test_accuracy = best_model.evaluate(
        test_dataset,
        verbose=1
    )

    # Display the final test results
    print("\nTest loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # ------------------------------------------
    # Log Test Metrics to MLflow
    # ------------------------------------------

    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    })

    # ------------------------------------------
    # Save Test Metrics
    # ------------------------------------------

    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    test_metrics_path = REPORT_DIR / "test_metrics.json"

    with open(
        test_metrics_path,
        "w"
    ) as file:

        json.dump(
            test_metrics,
            file,
            indent=4
        )

    # ------------------------------------------
    # Log Test Metrics File to MLflow
    # ------------------------------------------

    mlflow.log_artifact(
        str(test_metrics_path),
        artifact_path="reports"
    )

    # ------------------------------------------
    # Log Model File to MLflow
    # ------------------------------------------

    mlflow.log_artifact(
        str(MODEL_DIR / "baseline_cnn.keras"),
        artifact_path="models"
    )

    # ------------------------------------------
    # Display MLflow Run Information
    # ------------------------------------------

    print("\n==========================================")
    print("MLflow Training Run Completed")
    print("==========================================")

    print("Experiment: cats-dogs-baseline-cnn")
    print("Run ID:", run.info.run_id)

    print("\nModel saved to:")
    print(MODEL_DIR / "baseline_cnn.keras")

    print("\nTraining history saved to:")
    print(training_history_path)

    print("\nTest metrics saved to:")
    print(test_metrics_path)

    print("\nTest accuracy:", test_accuracy)

    print("\nMLflow tracking data saved locally.")