# Import necessary libraries for testing
import io

import numpy as np
from PIL import Image

from src.api import preprocess_image


def test_preprocess_image():
    # Create a temporary RGB image
    image = Image.new("RGB", (300, 300), color=(255, 0, 0))

    # Convert image to bytes, as the API receives
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    # Run the inference preprocessing function
    processed_image = preprocess_image(image_bytes.getvalue())

    # Verify the output shape expected by the CNN
    assert processed_image.shape == (1, 224, 224, 3)

    # Verify the image is normalized
    assert np.all(processed_image >= 0.0)
    assert np.all(processed_image <= 1.0)