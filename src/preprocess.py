# ==========================================
# Dataset Loading and Preprocessing
# ==========================================

from pathlib import Path

from sklearn.model_selection import train_test_split

import tensorflow as tf


# ==========================================
# Dataset Configuration
# ==========================================

DATA_DIR = Path("data/raw/PetImages")

CAT_DIR = DATA_DIR / "Cat"
DOG_DIR = DATA_DIR / "Dog"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32


# ==========================================
# Image Preprocessing
# ==========================================

def load_and_preprocess_image(image_path, label):
    """
    Load, resize, normalize, and return a single image.
    """

    image = tf.io.read_file(str(image_path))

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1)
])


# ==========================================
# TensorFlow Dataset Creation
# ==========================================

def create_dataset(image_paths, labels, training=False):
    """
    Create a TensorFlow data pipeline from image paths and labels.
    """

    image_paths = [str(path) for path in image_paths]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(image, training=True),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.shuffle(
            buffer_size=1000
        )

    dataset = dataset.ignore_errors()

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ==========================================
# Dataset Preparation
# ==========================================

def prepare_datasets():
    """
    Discover images, split the dataset, and create
    training, validation, and test datasets.
    """

    # Image Discovery
    cat_images = list(CAT_DIR.glob("*.jpg"))
    dog_images = list(DOG_DIR.glob("*.jpg"))

    print("Number of cat images:", len(cat_images))
    print("Number of dog images:", len(dog_images))

    # Label Preparation
    cat_labels = [0] * len(cat_images)
    dog_labels = [1] * len(dog_images)

    image_paths = cat_images + dog_images
    labels = cat_labels + dog_labels

    # Dataset Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    # Create datasets
    train_dataset = create_dataset(
        X_train,
        y_train,
        training=True
    )

    validation_dataset = create_dataset(
        X_val,
        y_val,
        training=False
    )

    test_dataset = create_dataset(
        X_test,
        y_test,
        training=False
    )

    # Dataset Verification
    sample_images, sample_labels = next(iter(train_dataset))

    print(
        "Training batch image shape:",
        sample_images.shape
    )

    print(
        "Training batch label shape:",
        sample_labels.shape
    )

    print(
        "Training batch pixel range:",
        float(tf.reduce_min(sample_images)),
        "to",
        float(tf.reduce_max(sample_images))
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )