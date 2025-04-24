import numpy as np
from preprocessing.preprocess import preprocess_array


def test_preprocess_array_shape_and_type():
    dummy_img = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    preprocessed = preprocess_array(dummy_img)
    assert preprocessed.shape == (224, 224)
    assert preprocessed.dtype == np.float32
