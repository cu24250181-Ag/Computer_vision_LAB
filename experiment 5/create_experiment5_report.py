from pathlib import Path
import textwrap

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from skimage import exposure
from skimage.feature import hog


BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "image.jpg"
OUTPUT_DIR = BASE_DIR / "experiment5_outputs"
PDF_PATH = BASE_DIR / "Experiment5_Arpit_Giri_Report.pdf"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_image(name, image, cmap=None):
    path = OUTPUT_DIR / f"{name}.png"
    plt.imsave(path, image, cmap=cmap)
    return path


def add_step(pdf, number, title, purpose, code, image_path=None, image_title=None, output_text=None):
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("white")
    figure.text(0.07, 0.955, f"Step {number}: {title}", fontsize=17, weight="bold", color="#17324d")
    figure.text(0.07, 0.918, purpose, fontsize=9.5, color="#333333")
    figure.text(0.07, 0.875, "Code", fontsize=11, weight="bold", color="#17324d")
    figure.text(
        0.07, 0.855, code, family="monospace", fontsize=6.8, va="top", linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f2f5f7", "edgecolor": "#c7d2da"}
    )
    if output_text:
        figure.text(0.07, 0.44, "Output", fontsize=11, weight="bold", color="#17324d")
        figure.text(0.07, 0.415, output_text, family="monospace", fontsize=8.5, va="top", linespacing=1.35,
                    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fafafa", "edgecolor": "#c7d2da"})
    if image_path:
        axis = figure.add_axes((0.09, 0.07, 0.82, 0.29 if output_text else 0.48))
        axis.imshow(plt.imread(image_path), cmap="gray")
        axis.set_title(image_title, fontsize=11)
        axis.axis("off")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoints_image = cv2.drawKeypoints(
        image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    hog_features, hog_image = hog(
        gray_resized, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    transformed = cv2.resize(image, None, fx=0.82, fy=0.82)
    transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
    transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    keypoints2, descriptors2 = sift.detectAndCompute(transformed_gray, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_image = cv2.drawMatches(
        image, keypoints, transformed, keypoints2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    outputs = {
        "original": save_image("01_original", image_rgb),
        "gray": save_image("02_grayscale", gray_resized, cmap="gray"),
        "sift": save_image("03_sift_keypoints", keypoints_image),
        "hog": save_image("04_hog_visualization", hog_image_rescaled, cmap="gray"),
        "match": save_image("05_sift_matching", cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)),
    }

    code = {
        "imports": """import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import hog
from skimage import exposure""",
        "load": """image_path = Path(__file__).parent / \"image.jpg\"
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(\"Image loaded successfully.\")""",
        "sift": """sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
keypoints_image = cv2.drawKeypoints(
    image_rgb, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
print(\"Number of keypoints:\", len(keypoints))
print(\"Descriptor shape:\", descriptors.shape)""",
        "hog": """gray_resized = cv2.resize(gray, (128, 128))
features, hog_image = hog(
    gray_resized, orientations=9,
    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
    visualize=True
)
hog_image_rescaled = exposure.rescale_intensity(
    hog_image, in_range=(0, 10)
)
print(\"HOG feature vector length:\", len(features))""",
        "matching": """transformed = cv2.rotate(
    cv2.resize(image, None, fx=0.82, fy=0.82),
    cv2.ROTATE_90_CLOCKWISE
)
keypoints2, descriptors2 = sift.detectAndCompute(
    cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY), None
)
matches = cv2.BFMatcher().knnMatch(descriptors, descriptors2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
print(\"Total matches:\", len(matches))
print(\"Good matches:\", len(good_matches))""",
    }

    with PdfPages(PDF_PATH) as pdf:
        cover = plt.figure(figsize=(8.27, 11.69))
        cover.patch.set_facecolor("#f6f8fa")
        cover.text(0.08, 0.82, "CU24250181", fontsize=13, color="#555555")
        cover.text(0.08, 0.77, "BTech CSE 3rd Year  |  Section B", fontsize=12, color="#555555")
        cover.text(0.08, 0.72, "Roll No. :- 11", fontsize=12, color="#555555")
        cover.text(0.08, 0.66, "Name :- Arpit Giri", fontsize=13, weight="bold", color="#17324d")
        cover.text(0.08, 0.54, "Experiment No. 5", fontsize=25, weight="bold", color="#17324d")
        cover.text(0.08, 0.47, "Feature Extraction and Image Analysis using SIFT and HOG Descriptors", fontsize=16, color="#2d6a8a")
        cover.text(0.08, 0.37, "Aim", fontsize=12, weight="bold", color="#17324d")
        cover.text(0.08, 0.33, "To implement SIFT and HOG for feature extraction and analyze their use in image representation,\nmatching, object recognition, and computer vision applications.", fontsize=10.5, linespacing=1.5)
        cover.text(0.08, 0.16, f"Input image used: {IMAGE_PATH.name}\nAll output images in this report were generated locally.", fontsize=10, color="#555555")
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        add_step(pdf, 1, "Import required libraries", "Load OpenCV, NumPy, Matplotlib, SIFT, HOG, and image enhancement tools.", code["imports"])
        add_step(pdf, 2, "Load and preprocess the image", "Read the local image, convert it to RGB for display, and create a grayscale image.", code["load"], outputs["original"], "Output: Original input image")
        add_step(pdf, 3, "Convert image to grayscale", "Resize the grayscale image to a consistent size for HOG analysis.", "gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\ngray_resized = cv2.resize(gray, (128, 128))", outputs["gray"], "Output: Grayscale image")
        add_step(pdf, 4, "Detect SIFT keypoints and descriptors", "SIFT identifies distinctive local points and creates a 128-dimensional descriptor for each point.", code["sift"], outputs["sift"], "Output: SIFT keypoints", f"Number of keypoints: {len(keypoints)}\nDescriptor shape: {descriptors.shape}")
        add_step(pdf, 5, "Extract HOG features", "HOG represents local edge directions and object shape using normalized gradient histograms.", code["hog"], outputs["hog"], "Output: HOG visualization", f"HOG feature vector length: {len(hog_features)}\nOrientations: 9\nPixels per cell: (8, 8)\nCells per block: (2, 2)")
        add_step(pdf, 6, "Match SIFT features", "Compare the original image with a rotated and resized copy using the ratio test.", code["matching"], outputs["match"], "Output: SIFT feature matching", f"Total matches: {len(matches)}\nGood matches: {len(good_matches)}")

        summary = plt.figure(figsize=(8.27, 11.69))
        summary.text(0.07, 0.95, "Experiment output and observations", fontsize=18, weight="bold", color="#17324d")
        text = (
            f"SIFT keypoints detected: {len(keypoints)}\n"
            f"SIFT descriptor shape: {descriptors.shape}\n"
            f"HOG feature vector length: {len(hog_features)}\n"
            f"Good SIFT matches with transformed image: {len(good_matches)}\n\n"
            "1. SIFT detects distinctive local keypoints and generates descriptors for feature matching.\n"
            "2. SIFT is robust to image scale and rotation changes.\n"
            "3. HOG captures edge direction and shape information.\n"
            "4. HOG is useful for object detection, especially when object shape is important.\n"
            "5. The number of detected features depends on image content and parameter settings.\n\n"
            "Conclusion: SIFT is preferable for local feature matching, while HOG is a compact shape descriptor\n"
            "for detection and classification pipelines."
        )
        summary.text(0.07, 0.88, text, fontsize=11, va="top", linespacing=1.55)
        pdf.savefig(summary, bbox_inches="tight")
        plt.close(summary)

        answers = plt.figure(figsize=(8.27, 11.69))
        answers.text(0.07, 0.95, "Answers to questions", fontsize=18, weight="bold", color="#17324d")
        qa = (
            "1. Feature extraction converts raw pixels into meaningful numerical representations such as edges, corners, shapes, and textures.\n\n"
            "2. SIFT constructs a scale space, detects stable keypoints, assigns orientations, and creates descriptors around the keypoints.\n\n"
            "3. A keypoint is a distinctive image location. A descriptor is the numerical representation of the local region around it.\n\n"
            "4. HOG divides an image into cells, calculates gradient orientation histograms, groups cells into blocks, and normalizes them.\n\n"
            "5. SIFT is robust to scale and rotation and is suited to image matching. HOG describes shape and edges and is suited to object detection.\n\n"
            "6. SIFT achieves scale invariance by detecting features across multiple scales and rotation invariance by assigning a dominant orientation.\n\n"
            "7. HOG is used in pedestrian detection, vehicle detection, and general shape-based object detection.\n\n"
            "8. Feature extraction reduces data complexity and highlights patterns that classifiers and detectors can use efficiently.\n\n"
            "9. Handcrafted descriptors are interpretable and can work with limited data, while deep learning learns richer features but usually needs more data and computation.\n\n"
            "10. SIFT descriptors support image matching and recognition; HOG descriptors support shape-based object detection."
        )
        answers.text(0.07, 0.89, qa, fontsize=9.6, va="top", linespacing=1.45)
        pdf.savefig(answers, bbox_inches="tight")
        plt.close(answers)

    print(f"Created PDF: {PDF_PATH}")
    print(f"Saved outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()