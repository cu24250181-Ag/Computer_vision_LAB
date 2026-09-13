import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# 1. Load grayscale image
# -------------------------------------------------

image_path = Path(__file__).with_name("image.jpg")

if not image_path.is_file():
    raise SystemExit(f"Error: Image not found at '{image_path}'. Add image.jpg to the Experiment 4 folder.")

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise SystemExit(f"Error: Could not read image at '{image_path}'. Check that it is a valid image file.")

print("Image loaded successfully.")
print("Image shape:", image.shape)


# -------------------------------------------------
# 2. Compute Discrete Fourier Transform (DFT)
# -------------------------------------------------

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)


# -------------------------------------------------
# 3. Shift zero-frequency component to center
# -------------------------------------------------

dft_shift = np.fft.fftshift(dft)


# -------------------------------------------------
# 4. Calculate magnitude spectrum
# -------------------------------------------------

magnitude_spectrum = cv2.magnitude(
    dft_shift[:, :, 0],
    dft_shift[:, :, 1]
)

# Log transformation for better visualization
magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)


# -------------------------------------------------
# 5. Create Low-Pass Filter
# -------------------------------------------------

rows, cols = image.shape

crow, ccol = rows // 2, cols // 2

# Create mask
low_pass_mask = np.zeros((rows, cols, 2), np.uint8)

# Radius of low-frequency region
radius = 30

cv2.circle(
    low_pass_mask,
    (ccol, crow),
    radius,
    (1, 1),
    -1
)


# -------------------------------------------------
# 6. Apply Low-Pass Filter
# -------------------------------------------------

low_pass_dft = dft_shift * low_pass_mask


# -------------------------------------------------
# 7. Inverse Fourier Transform for LPF
# -------------------------------------------------

low_pass_inverse = np.fft.ifftshift(low_pass_dft)

low_pass_image = cv2.idft(low_pass_inverse)

low_pass_image = cv2.magnitude(
    low_pass_image[:, :, 0],
    low_pass_image[:, :, 1]
)

# Normalize
low_pass_image = cv2.normalize(
    low_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

low_pass_image = np.uint8(low_pass_image)


# -------------------------------------------------
# 8. Create High-Pass Filter
# -------------------------------------------------

high_pass_mask = np.ones((rows, cols, 2), np.uint8)

cv2.circle(
    high_pass_mask,
    (ccol, crow),
    radius,
    (0, 0),
    -1
)


# -------------------------------------------------
# 9. Apply High-Pass Filter
# -------------------------------------------------

high_pass_dft = dft_shift * high_pass_mask


# -------------------------------------------------
# 10. Inverse Fourier Transform for HPF
# -------------------------------------------------

high_pass_inverse = np.fft.ifftshift(high_pass_dft)

high_pass_image = cv2.idft(high_pass_inverse)

high_pass_image = cv2.magnitude(
    high_pass_image[:, :, 0],
    high_pass_image[:, :, 1]
)

# Normalize
high_pass_image = cv2.normalize(
    high_pass_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

high_pass_image = np.uint8(high_pass_image)


# -------------------------------------------------
# 11. Display Results
# -------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(low_pass_image, cmap='gray')
plt.title("Low-Pass Filtered Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(high_pass_image, cmap='gray')
plt.title("High-Pass Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()