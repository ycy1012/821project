import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2

def extract_handcrafted_features(img: np.ndarray) -> np.ndarray:
    """
    Extract handcrafted features from a preprocessed grayscale image.
    
    Features include:
    - GLCM texture descriptors (5)
    - Sobel edge magnitude mean and variance (2)
    
    Parameters:
        img (np.ndarray): 2D array (float32), preprocessed grayscale image [0, 1]

    Returns:
        np.ndarray: Feature vector of shape (7,)
    """
    # Convert to 8-bit grayscale (required for GLCM)
    img_uint8 = (img * 255).astype(np.uint8)

    # GLCM Features
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = [graycoprops(glcm, prop)[0, 0] for prop in props]

    # Sobel Edge Features
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_mean = np.mean(sobel_mag)
    edge_std = np.std(sobel_mag)

    edge_features = [edge_mean, edge_std]

    # Concatenate all features
    features = np.array(glcm_features + edge_features, dtype=np.float32)
    return features
