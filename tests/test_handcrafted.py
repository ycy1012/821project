from preprocessing.preprocess import preprocess_image
from features.handcrafted import extract_handcrafted_features

img = preprocess_image("input_images/00001094_003.png")
features = extract_handcrafted_features(img)

print("Feature vector shape:", features.shape)
print("Features:", features)
