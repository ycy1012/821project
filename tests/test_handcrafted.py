import numpy as np
from features.handcrafted import extract_handcrafted_features


def test_handcrafted_features_dim():
    dummy_img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    features = extract_handcrafted_features(dummy_img)
    assert isinstance(features, np.ndarray)
    assert features.shape == (7,)
