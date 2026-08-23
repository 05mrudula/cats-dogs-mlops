# ==========================================
# Dataset Loading and Preprocessing
# ==========================================

# Import Path for handling file and directory paths
from pathlib import Path

# Import train_test_split for creating dataset splits
from sklearn.model_selection import train_test_split

# Import TensorFlow for image preprocessing
import tensorflow as tf


# ==========================================
# Dataset Configuration
# ==========================================

# Define the location of the Cats vs Dogs dataset
DATA_DIR = Path("data/raw/PetImages")
CAT_DIR = DATA_DIR / "Cat"
DOG_DIR = DATA_DIR / "Dog"

# Define the target image dimensions required by the CNN
IMAGE_SIZE = (224, 224)


# ==========================================
# Image Discovery
# ==========================================

# Collect all JPEG image paths from the Cat and Dog folders
cat_images = list(CAT_DIR.glob("*.jpg"))
dog_images = list(DOG_DIR.glob("*.jpg"))

# Display the number of images available in each class
print("Number of cat images:", len(cat_images))
print("Number of dog images:", len(dog_images))


# ==========================================
# Label Preparation
# ==========================================

# Assign labels: 0 for cats and 1 for dogs
cat_labels = [0] * len(cat_images)
dog_labels = [1] * len(dog_images)

# Combine image paths and their corresponding labels
image_paths = cat_images + dog_images
labels = cat_labels + dog_labels


# ==========================================
# Dataset Splitting
# ==========================================

# Split the dataset into 80% training data and 20% temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Split the temporary data equally into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Display the final dataset split sizes
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ==========================================
# Image Preprocessing
# ==========================================

# Load and preprocess a single image
def load_and_preprocess_image(image_path):
    # Load the image, resize it, and ensure it has RGB channels
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert the image into a numerical array
    image = tf.keras.utils.img_to_array(image)

    # Normalize pixel values from 0-255 to 0-1
    image = image / 255.0

    return image


# ==========================================
# Preprocessing Verification
# ==========================================

# Test preprocessing using the first image in the dataset
sample_image = load_and_preprocess_image(image_paths[0])

# Display the shape and pixel value range of the processed image
print("Processed image shape:", sample_image.shape)
print("Pixel value range:", sample_image.min(), "to", sample_image.max())


# ==========================================
# Data Augmentation
# ==========================================

# Define augmentation techniques for the training images
data_augmentation = tf.keras.Sequential([
    # Randomly flip images horizontally
    tf.keras.layers.RandomFlip("horizontal"),

    # Randomly rotate images slightly
    tf.keras.layers.RandomRotation(0.1),

    # Randomly zoom into images
    tf.keras.layers.RandomZoom(0.1),
])

# ==========================================
# Augmentation Verification
# ==========================================

# Add a batch dimension because Keras augmentation layers expect batches
sample_batch = tf.expand_dims(sample_image, axis=0)

# Apply data augmentation to the sample image
augmented_image = data_augmentation(sample_batch, training=True)

# Display the shape of the augmented image
print("Augmented image shape:", augmented_image.shape)