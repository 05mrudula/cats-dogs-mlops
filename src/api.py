# ==========================================
# Cats vs Dogs - FastAPI Inference Service
# ==========================================

from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

import time
import logging


# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = (224, 224)

MODEL_PATH = Path("models/baseline_cnn.keras")

CLASS_NAMES = {
    0: "Cat",
    1: "Dog"
}


# ==========================================
# Load Model
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="CNN-based image classification API",
    version="1.0.0"
)


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health_check():
    """
    Check whether the inference service is running.
    """

    return {
        "status": "healthy"
    }


# ==========================================
# Image Preprocessing
# ==========================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes into the format
    expected by the CNN model.
    """

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(IMAGE_SIZE)

        # Convert image to NumPy array
        image_array = np.array(image)

        # Normalize pixel values to 0-1
        image_array = image_array.astype(
            np.float32
        ) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}"
        )


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """
    Predict whether an uploaded image is a Cat or Dog.
    """

    # Validate content type
    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    # Read uploaded image
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Preprocess image
    image_array = preprocess_image(
        image_bytes
    )

    # Run model prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    # Sigmoid output:
    # < 0.5 -> Cat
    # >= 0.5 -> Dog
    predicted_class = (
        "Dog"
        if probability >= 0.5
        else "Cat"
    )

    # Probability of predicted class
    confidence = (
        probability
        if predicted_class == "Dog"
        else 1.0 - probability
    )

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "probability": round(
            confidence,
            4
        )
    }
# ==========================================
# Logging Configuration
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# Request Logging Middleware
# ==========================================

@app.middleware("http")
async def log_requests(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "%s %s - Status: %s - Latency: %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response