import numpy as np
from features.deep import extract_deep_features


def test_deep_features_dim():
    dummy_img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    features = extract_deep_features(dummy_img)
    assert isinstance(features, np.ndarray)
    assert features.shape == (512,)  # ResNet embedding
