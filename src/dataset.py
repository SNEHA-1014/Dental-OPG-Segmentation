"""
dataset.py

Creates a filtered OPG segmentation dataset by:
1. Selecting target classes
2. Remapping class IDs
3. Applying CLAHE preprocessing
4. Creating a new YOLO segmentation dataset
5. Generating data.yaml


"""

import os
import cv2
import yaml
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset paths
ORIGINAL_DATASET = os.getenv(
    "ORIGINAL_DATASET_PATH",
    "data/original"
)

FILTERED_DATASET = os.getenv(
    "FILTERED_DATASET_PATH",
    "data/filtered_opg_dataset"
)


TARGET_CLASSES = {
    4: 0,     # Filling
    23: 1,    # impacted tooth
    14: 2,    # Root Canal Treatment
    2: 3,     # Crown
    1: 4,     # Caries
    10: 5,    # Periapical lesion
    9: 6,     # Missing teeth
    0: 7,     # Bone Loss
    15: 8,    # Root Piece
    6: 9      # Implant
}


CLASS_NAMES = [
    "Filling",
    "impacted tooth",
    "Root Canal Treatment",
    "Crown",
    "Caries",
    "Periapical lesion",
    "Missing teeth",
    "Bone Loss",
    "Root Piece",
    "Implant"
]


# ============================================================
# CREATE FOLDER STRUCTURE
# ============================================================

def create_dataset_structure(output_dir):
    """
    Create YOLO dataset folder structure.
    """

    splits = ["train", "valid", "test"]

    for split in splits:

        os.makedirs(
            os.path.join(output_dir, split, "images"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(output_dir, split, "labels"),
            exist_ok=True
        )


# ============================================================
# CLAHE PREPROCESSING
# ============================================================

def apply_clahe(image):
    """
    Apply CLAHE contrast enhancement.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )

    return enhanced


# ============================================================
# FILTER DATASET
# ============================================================

def filter_dataset(
    input_dir,
    output_dir,
    target_classes
):
    """
    Filter segmentation dataset.
    """

    splits = ["train", "valid", "test"]

    print("\nFiltering dataset...\n")

    for split in splits:

        print(f"\nProcessing {split}...\n")

        image_dir = os.path.join(
            input_dir,
            split,
            "images"
        )

        label_dir = os.path.join(
            input_dir,
            split,
            "labels"
        )

        output_image_dir = os.path.join(
            output_dir,
            split,
            "images"
        )

        output_label_dir = os.path.join(
            output_dir,
            split,
            "labels"
        )

        image_files = os.listdir(image_dir)

        for image_file in tqdm(image_files):

            image_path = os.path.join(
                image_dir,
                image_file
            )

            label_file = (
                image_file.rsplit(".", 1)[0]
                + ".txt"
            )

            label_path = os.path.join(
                label_dir,
                label_file
            )

            if not os.path.exists(label_path):
                continue

            filtered_lines = []

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:

                values = line.strip().split()

                if len(values) < 7:
                    continue

                old_class = int(values[0])

                if old_class in target_classes:

                    new_class = target_classes[
                        old_class
                    ]

                    values[0] = str(new_class)

                    filtered_lines.append(
                        " ".join(values)
                    )

            if len(filtered_lines) == 0:
                continue

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = apply_clahe(image)

            output_image_path = os.path.join(
                output_image_dir,
                image_file
            )

            cv2.imwrite(
                output_image_path,
                image
            )

            output_label_path = os.path.join(
                output_label_dir,
                label_file
            )

            with open(output_label_path, "w") as f:

                for line in filtered_lines:
                    f.write(line + "\n")

    print("\nDataset filtering completed!")


# ============================================================
# CREATE YAML
# ============================================================

def create_yaml(
    output_dir,
    class_names
):
    """
    Create YOLO data.yaml.
    """

    yaml_data = {

        "path": output_dir,

        "train": "train/images",

        "val": "valid/images",

        "test": "test/images",

        "nc": len(class_names),

        "names": class_names
    }

    yaml_path = os.path.join(
        output_dir,
        "data.yaml"
    )

    with open(yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            sort_keys=False
        )

    print("\nCreated:")
    print(yaml_path)


# ============================================================
# MAIN
# ============================================================

def main():

    create_dataset_structure(
        FILTERED_DATASET
    )

    filter_dataset(
        ORIGINAL_DATASET,
        FILTERED_DATASET,
        TARGET_CLASSES
    )

    create_yaml(
        FILTERED_DATASET,
        CLASS_NAMES
    )

    print("\nDataset preparation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()