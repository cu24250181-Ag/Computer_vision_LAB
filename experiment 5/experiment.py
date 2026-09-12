import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage.feature import hog
from skimage import exposure

image_path = Path(__file__).parent / "image.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise SystemExit(f"Image not found: {image_path}")

print("Image loaded successfully.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

keypoints_image = cv2.drawKeypoints(
    image_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

gray_resized = cv2.resize(gray, (128, 128))

features, hog_image = hog(
    gray_resized,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

print("HOG feature vector length:", len(features))

hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray_resized, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(keypoints_image)
plt.title("SIFT Keypoints")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.show()