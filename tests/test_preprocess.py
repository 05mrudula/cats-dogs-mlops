import numpy as np
from PIL import Image

from src.preprocess import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path):
    # Create a small temporary RGB image
    image_path = tmp_path / "test.jpg"

    image = Image.new("RGB", (300, 300), color=(255, 0, 0))
    image.save(image_path)

    # Run the preprocessing function
    processed_image, label = load_and_preprocess_image(
    str(image_path),
    1
    )

    # Verify the image was resized correctly
    assert processed_image.shape == (224, 224, 3)

    # Verify pixel values are normalized
    assert np.all(processed_image >= 0.0)
    assert np.all(processed_image <= 1.0)

    # Verify the label is preserved
    assert label == 1