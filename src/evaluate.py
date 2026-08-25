# ==========================================
# Model Evaluation and Performance Analysis
# ==========================================

# Import JSON for saving evaluation results
import json

# Import Path for handling file and directory paths
from pathlib import Path

# Import TensorFlow for loading and evaluating the trained model
import tensorflow as tf

# Import NumPy for numerical operations
import numpy as np

# Import Matplotlib for creating evaluation plots
import matplotlib.pyplot as plt

# Import classification metrics from scikit-learn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Import the test dataset and training history
from preprocess import test_dataset


# ==========================================
# Directory Configuration
# ==========================================

# Define the location of the trained baseline model
MODEL_PATH = Path("models/baseline_cnn.keras")

# Define the directory where evaluation results will be stored
REPORT_DIR = Path("reports")

# Create the reports directory if it does not already exist
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load Trained Model
# ==========================================

# Load the best-performing CNN model saved during training
model = tf.keras.models.load_model(MODEL_PATH)

# Display a confirmation message
print("Baseline CNN model loaded successfully.")


# ==========================================
# Evaluate Model on Test Dataset
# ==========================================

# Evaluate the trained model using unseen test data
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

# Display the test performance
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# Collect Predictions
# ==========================================

# Lists are used to store the actual labels and model predictions
true_labels = []
predicted_labels = []


# Iterate through the complete test dataset
for images, labels in test_dataset:

    # Generate prediction probabilities for the test images
    predictions = model.predict(
        images,
        verbose=0
    )

    # Convert sigmoid probabilities into binary class labels
    predictions = (predictions >= 0.5).astype(int).flatten()

    # Store the actual labels
    true_labels.extend(
        labels.numpy().astype(int)
    )

    # Store the predicted labels
    predicted_labels.extend(
        predictions
    )


# Convert lists into NumPy arrays for metric calculation
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ==========================================
# Classification Report
# ==========================================

# Generate precision, recall, and F1-score for each class
report = classification_report(
    true_labels,
    predicted_labels,
    target_names=["Cat", "Dog"],
    output_dict=True
)

# Display the classification report in the terminal
print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["Cat", "Dog"]
    )
)


# ==========================================
# Save Classification Report
# ==========================================

# Save the classification report as a JSON file
with open(
    REPORT_DIR / "classification_report.json",
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ==========================================
# Generate Confusion Matrix
# ==========================================

# Calculate the confusion matrix using actual and predicted labels
cm = confusion_matrix(
    true_labels,
    predicted_labels
)

# Display the confusion matrix in the terminal
print("Confusion Matrix:")
print(cm)


# ==========================================
# Plot Confusion Matrix
# ==========================================

# Create a visual representation of the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

# Plot the confusion matrix
display.plot()

# Add a descriptive title
plt.title("Cats vs Dogs - Baseline CNN Confusion Matrix")

# Save the confusion matrix figure
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    bbox_inches="tight"
)

# Display the plot
plt.show()

# Close the figure to release memory
plt.close()


# ==========================================
# Save Evaluation Metrics
# ==========================================

# Store the main evaluation metrics in a dictionary
evaluation_metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "cat_precision": float(report["Cat"]["precision"]),
    "cat_recall": float(report["Cat"]["recall"]),
    "cat_f1_score": float(report["Cat"]["f1-score"]),
    "dog_precision": float(report["Dog"]["precision"]),
    "dog_recall": float(report["Dog"]["recall"]),
    "dog_f1_score": float(report["Dog"]["f1-score"])
}

# Save the metrics as a JSON file
with open(
    REPORT_DIR / "evaluation_metrics.json",
    "w"
) as file:

    json.dump(
        evaluation_metrics,
        file,
        indent=4
    )


# ==========================================
# Evaluation Completion Message
# ==========================================

# Confirm that all evaluation artifacts have been generated
print("\nEvaluation completed successfully.")

print(
    "Classification report saved to:",
    REPORT_DIR / "classification_report.json"
)

print(
    "Evaluation metrics saved to:",
    REPORT_DIR / "evaluation_metrics.json"
)

print(
    "Confusion matrix saved to:",
    REPORT_DIR / "confusion_matrix.png"
)