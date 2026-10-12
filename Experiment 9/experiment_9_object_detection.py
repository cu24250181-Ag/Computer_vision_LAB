# -*- coding: utf-8 -*-
"""Object detection experiment using YOLOv8.

This script is written to run as a normal Python script instead of a Google
Colab notebook cell. It can be executed locally and accepts an input image path.

Example:
    python "Experiment 9\\experiment_9_object_detection.py" --image "Experiment 4\\image.jpg"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_dependencies() -> None:
    """Install required packages when they are missing."""
    try:
        import cv2  # noqa: F401
        import matplotlib  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "ultralytics",
                "opencv-python",
                "matplotlib",
            ]
        )


ensure_dependencies()

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_objects(image_path: str, output_path: str, confidence: float, iou: float) -> None:
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    model = YOLO("yolov8n.pt")
    results = model(str(image_file), conf=confidence, iou=iou)

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_file), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title("YOLOv8 Object Detection Results")
    plt.tight_layout()
    plt.savefig(str(output_file.with_suffix(".png")), bbox_inches="tight")
    plt.close()

    print("Object detection completed!")
    total_objects = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_value = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"Object: {class_name}, Confidence: {confidence_value:.2f}")
            total_objects += 1

    print(f"Total Detected Objects: {total_objects}")
    print(f"Annotated image saved to: {output_file}")
    print(f"Figure saved to: {output_file.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image file.")
    parser.add_argument(
        "--output",
        default="detected_output.jpg",
        help="Path for the saved annotated image (default: detected_output.jpg)",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold used by the detector.")
    args = parser.parse_args()

    detect_objects(args.image, args.output, args.confidence, args.iou)


if __name__ == "__main__":
    main()


REPORT = """Q1. Differentiate between Image Classification, Object Detection, and Image Segmentation.

Answer:

These are three important computer vision tasks.

Feature    Image Classification    Object Detection    Image Segmentation
Meaning    Identifies the category of an image    Identifies and locates objects    Identifies the exact pixels of objects
Output    Class label    Bounding boxes + class labels    Pixel-level masks
Multiple objects    Generally gives image-level categories    Detects multiple objects    Separates individual objects or regions
Example    Image contains a cat    Cat detected with a bounding box    Exact pixels belonging to the cat are identified

Explanation:

Image Classification: Determines what an image contains.
Object Detection: Determines what objects are present and where they are located.
Image Segmentation: Determines which pixels belong to each object or region.

Conclusion: Object detection provides both object classification and localization.

Q2. Explain the working principle of the YOLO (You Only Look Once) object detection algorithm.

Answer:

YOLO stands for You Only Look Once. It is a deep learning-based object detection algorithm that predicts object classes and bounding boxes in an image.

Working Steps:
Input Image: The image is provided to the YOLO model.
Feature Extraction: A neural network extracts important features from the image.
Object Prediction: The model predicts possible object locations, classes, and confidence scores.
Bounding Boxes: Each detected object is represented using a bounding box.
Filtering: Low-confidence predictions and overlapping duplicate detections may be filtered.
Output: The model displays detected objects with their labels and confidence scores.
Example:

Input Image → YOLO Model → Detected Objects

Person (0.95)
Car    (0.89)
Dog    (0.91)

Advantages:

Fast detection.
Suitable for many real-time applications.
Can detect multiple objects in an image.

Conclusion: YOLO performs object detection efficiently by predicting object locations and classes through a neural network.

Q3. Compare YOLO, SSD, and Faster R-CNN based on speed, accuracy, and practical applications.

Answer:

YOLO, SSD, and Faster R-CNN are deep learning-based object detection models.

Feature    YOLO    SSD    Faster R-CNN
Full Form    You Only Look Once    Single Shot Detector    Faster Region-based CNN
Detection Approach    Single-shot    Single-shot    Two-stage
Speed    Generally fast    Generally fast    Generally slower
Accuracy    Depends on model and task    Depends on model and task    Often strong localization performance
Real-time Applications    Commonly used    Suitable for real-time tasks    Used when accuracy and region refinement are important
Example    Video object detection    Embedded detection systems    Research and detailed detection tasks
Explanation:

YOLO:

Designed for efficient object detection.
Commonly used in real-time applications.

SSD:

Predicts object classes and locations in a single detection process.
Uses multiple feature scales.

Faster R-CNN:

Uses a Region Proposal Network (RPN).
Refines proposed object regions in a second stage.

Conclusion: Speed and accuracy depend on the particular implementation, hardware, and dataset. There is no single model that is best for every application.

Q4. What is a bounding box? Why is it important in object detection?

Answer:

A bounding box is a rectangular box drawn around a detected object in an image.

It identifies the approximate location of the object.

Example:
+-------------------------+
|                         |
|    +------------+       |
|    |    CAR     |       |
|    +------------+       |
|                         |
+-------------------------+
Bounding Box Coordinates:

A bounding box can be represented as:

(x1, y1, x2, y2)

Where:

x1, y1: Top-left corner.
x2, y2: Bottom-right corner.
Importance:
Identifies the location of an object.
Helps distinguish multiple objects.
Supports object tracking.
Enables localization performance evaluation.
Helps downstream systems process detected objects.

Conclusion: Bounding boxes are essential because they show where an object is located in an image.

Q5. Explain the significance of Confidence Score and Intersection over Union (IoU) in object detection.

Answer:

Confidence Score and IoU are important concepts in object detection.

A. Confidence Score

A confidence score represents the model's estimated confidence in a detection.

Example:

Person: 0.95
Car:    0.87
Dog:    0.72

A score of 0.95 means the model has a high estimated confidence in that prediction.

Importance:

Helps filter detections.
Supports interpretation of model predictions.
Can be used to set a detection threshold.

Note: Confidence scores are not guaranteed to be perfectly calibrated probabilities.

B. Intersection over Union (IoU)

IoU measures the overlap between the predicted bounding box and the ground-truth bounding box.

Formula:

IoU = Area of Intersection / Area of Union

Example:

If the predicted box overlaps significantly with the actual box, the IoU value will generally be higher.

IoU    Meaning
0    No overlap
0.5    Partial overlap
1.0    Perfect overlap

Importance of IoU:

Evaluates bounding box localization.
Helps determine whether a detection matches the ground truth.
Is used in object detection evaluation metrics.

Conclusion:

Confidence Score → Model's estimated confidence.
IoU → Overlap between predicted and actual boxes.

Q6. What are the advantages of using pre-trained deep learning models over traditional image processing techniques?

Answer:

Pre-trained deep learning models are trained on existing datasets and can be adapted or used for new tasks.
"""
