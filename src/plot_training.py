# ==========================================
# Training Performance Visualization
# ==========================================

# Import JSON for reading the saved training history
import json

# Import Path for handling file paths
from pathlib import Path

# Import Matplotlib for creating plots
import matplotlib.pyplot as plt


# ==========================================
# File Configuration
# ==========================================

# Define the location of the saved training history
HISTORY_PATH = Path("reports/training_history.json")

# Define the directory where plots will be saved
REPORT_DIR = Path("reports")

# Create the reports directory if it does not already exist
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load Training History
# ==========================================

# Open the training history JSON file
with open(HISTORY_PATH, "r") as file:

    # Convert the JSON data into a Python dictionary
    history = json.load(file)


# ==========================================
# Extract Training Metrics
# ==========================================

# Extract training accuracy recorded for each epoch
training_accuracy = history["accuracy"]

# Extract validation accuracy recorded for each epoch
validation_accuracy = history["val_accuracy"]

# Extract training loss recorded for each epoch
training_loss = history["loss"]

# Extract validation loss recorded for each epoch
validation_loss = history["val_loss"]

# Create epoch numbers starting from 1
epochs = range(1, len(training_accuracy) + 1)


# ==========================================
# Plot Training and Validation Accuracy
# ==========================================

# Create a new figure for the accuracy plot
plt.figure()

# Plot training accuracy across epochs
plt.plot(
    epochs,
    training_accuracy,
    label="Training Accuracy"
)

# Plot validation accuracy across epochs
plt.plot(
    epochs,
    validation_accuracy,
    label="Validation Accuracy"
)

# Add a descriptive title
plt.title("Training and Validation Accuracy")

# Label the x-axis
plt.xlabel("Epoch")

# Label the y-axis
plt.ylabel("Accuracy")

# Display the legend
plt.legend()

# Add a grid to make the graph easier to read
plt.grid(True)

# Save the accuracy plot
plt.savefig(
    REPORT_DIR / "training_validation_accuracy.png",
    bbox_inches="tight"
)

# Display the plot
plt.show()

# Close the figure to release memory
plt.close()


# ==========================================
# Plot Training and Validation Loss
# ==========================================

# Create a new figure for the loss plot
plt.figure()

# Plot training loss across epochs
plt.plot(
    epochs,
    training_loss,
    label="Training Loss"
)

# Plot validation loss across epochs
plt.plot(
    epochs,
    validation_loss,
    label="Validation Loss"
)

# Add a descriptive title
plt.title("Training and Validation Loss")

# Label the x-axis
plt.xlabel("Epoch")

# Label the y-axis
plt.ylabel("Loss")

# Display the legend
plt.legend()

# Add a grid to make the graph easier to read
plt.grid(True)

# Save the loss plot
plt.savefig(
    REPORT_DIR / "training_validation_loss.png",
    bbox_inches="tight"
)

# Display the plot
plt.show()

# Close the figure to release memory
plt.close()


# ==========================================
# Completion Message
# ==========================================

# Confirm that both plots were successfully generated
print("Training accuracy plot saved to:")
print(REPORT_DIR / "training_validation_accuracy.png")

print("\nTraining loss plot saved to:")
print(REPORT_DIR / "training_validation_loss.png")