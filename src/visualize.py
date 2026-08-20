"""
visualize.py

Visualize:
1. Original Image
2. Ground Truth Annotations
3. Model Predictions

"""

import os
from pathlib import Path
import cv2
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "filtered_opg_dataset"

PREDICTION_PATH = PROJECT_ROOT / "predictions"

NUM_SAMPLES = 5


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """
    Load class names from data.yaml.
    """

    yaml_path = os.path.join(
        DATASET_PATH,
        "data.yaml"
    )

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    return data["names"]


# ============================================================
# DRAW GROUND TRUTH POLYGONS
# ============================================================

def draw_ground_truth(
    image,
    label_path,
    class_names
):
    """
    Draw segmentation polygons.
    """

    gt_image = image.copy()

    h, w = gt_image.shape[:2]

    if not os.path.exists(label_path):
        return gt_image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) < 7:
            continue

        cls = int(values[0])

        polygon_points = list(
            map(float, values[1:])
        )

        polygon = []

        for i in range(
            0,
            len(polygon_points),
            2
        ):

            x = int(
                polygon_points[i] * w
            )

            y = int(
                polygon_points[i + 1] * h
            )

            polygon.append([x, y])

        polygon = np.array(
            polygon,
            dtype=np.int32
        )

        cv2.polylines(
            gt_image,
            [polygon],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        cv2.putText(
            gt_image,
            class_names[cls],
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return gt_image


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(
    num_samples=5
):
    """
    Visualize random samples.
    """

    class_names = load_class_names()

    test_image_dir = os.path.join(
        DATASET_PATH,
        "test",
        "images"
    )

    test_label_dir = os.path.join(
        DATASET_PATH,
        "test",
        "labels"
    )

    all_images = os.listdir(
        test_image_dir
    )

    sample_images = random.sample(
        all_images,
        min(num_samples, len(all_images))
    )

    for image_file in sample_images:

        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        original_path = os.path.join(
            test_image_dir,
            image_file
        )

        original = cv2.imread(
            original_path
        )

        if original is None:
            continue

        original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # GROUND TRUTH
        # ====================================================

        label_file = (
            image_file.rsplit(".", 1)[0]
            + ".txt"
        )

        label_path = os.path.join(
            test_label_dir,
            label_file
        )

        gt_image = draw_ground_truth(
            original,
            label_path,
            class_names
        )

        # ====================================================
        # PREDICTION IMAGE
        # ====================================================

        pred_path = os.path.join(
            PREDICTION_PATH,
            image_file
        )

        prediction = cv2.imread(
            pred_path
        )

        if prediction is None:
            continue

        prediction = cv2.cvtColor(
            prediction,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        plt.figure(
            figsize=(24, 8)
        )

        plt.subplot(1, 3, 1)

        plt.imshow(original)

        plt.title("Original")

        plt.axis("off")

        plt.subplot(1, 3, 2)

        plt.imshow(gt_image)

        plt.title("Ground Truth")

        plt.axis("off")

        plt.subplot(1, 3, 3)

        plt.imshow(prediction)

        plt.title("Prediction")

        plt.axis("off")

        plt.tight_layout()

        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    visualize_predictions(
        num_samples=NUM_SAMPLES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()