import cv2
import numpy as np


def preprocess_image(image_path: str, size=(224, 224)) -> np.ndarray:
    """
    Preprocess a chest X-ray image:
    1. Read image as grayscale
    2. Resize to target size
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalize pixel values to [0, 1]

    Parameters:
        image_path (str): Path to the image file
        size (tuple): Desired output size (width, height)

    Returns:
        np.ndarray: Preprocessed image as float32 array
    """
    # Load image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize
    img = cv2.resize(img, size)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Normalize to [0, 1]
    img_norm = img_clahe.astype(np.float32) / 255.0

    return img_norm


def preprocess_array(img: np.ndarray, size=(224, 224)) -> np.ndarray:
    """
    Preprocess a grayscale chest X-ray image from a NumPy array:
    1. Resize to target size
    2. Apply CLAHE (contrast enhancement)
    3. Normalize pixel values to [0, 1]

    Parameters:
        img (np.ndarray): Input grayscale image as a 2D NumPy array
        size (tuple): Target image size as (width, height). Default: (224, 224)

    Returns:
        np.ndarray: Preprocessed image of shape (size[1], size[0]), dtype float32
    """
    resized = cv2.resize(img, size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    norm = enhanced.astype(np.float32) / 255.0
    return norm
