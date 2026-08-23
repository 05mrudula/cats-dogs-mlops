# ==========================================
# CNN Model Configuration
# ==========================================

# Import TensorFlow for building and training the CNN
import tensorflow as tf


# Define the input image dimensions
IMAGE_SIZE = (224, 224)

# Define the number of color channels in an RGB image
CHANNELS = 3

# Define the input shape expected by the CNN
INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)


# ==========================================
# CNN Model Architecture
# ==========================================

# Create the CNN model using a sequential architecture
model = tf.keras.Sequential([

    # Define the shape of the input images
    tf.keras.layers.Input(shape=INPUT_SHAPE),

    # Extract basic spatial features such as edges and textures
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Reduce the spatial dimensions while retaining important features
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Extract more complex visual features
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Further reduce the spatial dimensions of the feature maps
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Extract higher-level visual features
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Reduce the spatial dimensions before classification
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Flatten the feature maps into a one-dimensional vector
    tf.keras.layers.Flatten(),

    # Learn high-level patterns from the extracted image features
    tf.keras.layers.Dense(
        units=128,
        activation="relu"
    ),

    # Randomly deactivate neurons during training to reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Produce a probability indicating whether the image is a dog
    tf.keras.layers.Dense(
        units=1,
        activation="sigmoid"
    )
])


# ==========================================
# Model Verification
# ==========================================

# Display the CNN architecture and number of parameters
model.summary()

# ==========================================
# Model Compilation
# ==========================================

# Configure the model for binary classification
model.compile(
    # Adam optimizer updates the model weights during training
    optimizer="adam",

    # Binary cross-entropy is suitable for two-class classification
    loss="binary_crossentropy",

    # Track classification accuracy during training and validation
    metrics=["accuracy"]
)