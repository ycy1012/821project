from preprocessing.preprocess import preprocess_image
import matplotlib.pyplot as plt

img = preprocess_image("input_images/00001094_003.png", size=(224, 224))

print("Shape:", img.shape)
print("Min/Max:", img.min(), img.max())

plt.imshow(img, cmap="gray")
plt.title("Preprocessed Image")
plt.axis("off")
plt.show()
