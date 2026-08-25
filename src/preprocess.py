# ==========================================
# Dataset Loading and Preprocessing
# ==========================================

# Import Path for handling file and directory paths
from pathlib import Path

# Import train_test_split for creating dataset splits
from sklearn.model_selection import train_test_split

# Import TensorFlow for image preprocessing and data pipelines
import tensorflow as tf


# ==========================================
# Dataset Configuration
# ==========================================

# Define the location of the Cats vs Dogs dataset
DATA_DIR = Path("data/raw/PetImages")

# Define the directories containing cat and dog images
CAT_DIR = DATA_DIR / "Cat"
DOG_DIR = DATA_DIR / "Dog"

# Define the target image dimensions required by the CNN
IMAGE_SIZE = (224, 224)

# Define the number of images processed together in one batch
BATCH_SIZE = 32


# ==========================================
# Image Discovery
# ==========================================

# Collect all JPG image paths from the Cat directory
cat_images = list(CAT_DIR.glob("*.jpg"))

# Collect all JPG image paths from the Dog directory
dog_images = list(DOG_DIR.glob("*.jpg"))

# Display the number of images available in each class
print("Number of cat images:", len(cat_images))
print("Number of dog images:", len(dog_images))


# ==========================================
# Label Preparation
# ==========================================

# Assign label 0 to cat images
cat_labels = [0] * len(cat_images)

# Assign label 1 to dog images
dog_labels = [1] * len(dog_images)

# Combine image paths from both classes
image_paths = cat_images + dog_images

# Combine the corresponding labels
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
# This produces an overall 80/10/10 dataset split
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

def load_and_preprocess_image(image_path, label):
    """
    Load, resize, normalize, and return a single image.

    TensorFlow operations are used so that this function
    can run correctly inside a tf.data data pipeline.
    """

    # Read the image file as raw binary data
    image = tf.io.read_file(image_path)

    # Decode the image while automatically detecting its format
    # This allows the pipeline to handle different image formats
    # even when the file extension is misleading.
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Explicitly define the image shape for TensorFlow
    image.set_shape([None, None, 3])

    # Resize the image to the dimensions required by the CNN
    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Convert pixel values from 0-255 to the range 0-1
    image = tf.cast(image, tf.float32) / 255.0

    # Return the processed image and its corresponding label
    return image, label


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
    tf.keras.layers.RandomZoom(0.1)
])


# ==========================================
# TensorFlow Dataset Creation
# ==========================================

def create_dataset(image_paths, labels, training=False):
    """
    Create a TensorFlow data pipeline from image paths and labels.
    """

    # Convert Path objects into strings for TensorFlow
    image_paths = [str(path) for path in image_paths]

    # Create a TensorFlow dataset from image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    # Load and preprocess each image
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply augmentation only to the training dataset
    if training:

        # Randomly transform training images to improve
        # the model's ability to generalize to unseen data
        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(image, training=True),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Shuffle the training samples to reduce ordering bias
        dataset = dataset.shuffle(
            buffer_size=1000
        )

    # Skip individual images that cannot be decoded or processed
    # so that one corrupt image does not stop the entire pipeline
    dataset = dataset.ignore_errors()

    # Group images into batches for efficient processing
    dataset = dataset.batch(BATCH_SIZE)

    # Prefetch batches to improve data pipeline performance
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ==========================================
# Create Training Dataset
# ==========================================

# Create the training dataset with augmentation enabled
train_dataset = create_dataset(
    X_train,
    y_train,
    training=True
)


# ==========================================
# Create Validation Dataset
# ==========================================

# Create the validation dataset without augmentation
validation_dataset = create_dataset(
    X_val,
    y_val,
    training=False
)


# ==========================================
# Create Test Dataset
# ==========================================

# Create the test dataset without augmentation
test_dataset = create_dataset(
    X_test,
    y_test,
    training=False
)


# ==========================================
# Dataset Verification
# ==========================================

# Retrieve one batch from the training dataset
sample_images, sample_labels = next(iter(train_dataset))

# Display the shape of the training image batch
print("Training batch image shape:", sample_images.shape)

# Display the shape of the corresponding labels
print("Training batch label shape:", sample_labels.shape)

# Display the range of pixel values after preprocessing
print(
    "Training batch pixel range:",
    float(tf.reduce_min(sample_images)),
    "to",
    float(tf.reduce_max(sample_images))
)