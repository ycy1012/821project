import os
import numpy as np
from preprocessing.preprocess import preprocess_image
from features.deep import extract_deep_features

def test_deep_feature_shape():
    # Pick one image from input_images/
    img_path = "input_images/00001094_003.png"
    assert os.path.exists(img_path), f"Image not found: {img_path}"

    # Preprocess image
    img = preprocess_image(img_path)

    # Extract deep features
    features = extract_deep_features(img)

    # Assertions
    assert isinstance(features, np.ndarray), "Output should be a NumPy array"
    assert features.shape == (512,), f"Expected shape (512,), got {features.shape}"
    print("Deep feature vector extracted successfully.")
    print("First 5 values:", features[:5])

if __name__ == "__main__":
    test_deep_feature_shape()
