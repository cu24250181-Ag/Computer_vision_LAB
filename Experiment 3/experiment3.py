import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load image
image_path = Path(__file__).resolve().parent / "image.jpg"
img = cv2.imread(str(image_path))
if img is None:
	raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Low-pass filters
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
average = cv2.blur(img, (5, 5))

# High-pass filters
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
plt.title("Gaussian")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.title("Median")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(average, cv2.COLOR_BGR2RGB))
plt.title("Average")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(np.abs(laplacian), cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(np.abs(sobel_x), cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.tight_layout()
plt.show()

# Display Sobel Y separately
plt.imshow(np.abs(sobel_y), cmap="gray")
plt.title("Sobel Y")
plt.axis("off")
plt.show()