"""
validate.py

Validate the trained YOLOv8 segmentation model.

"""

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolov8s_seg_baseline-2"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "filtered_opg_dataset"
    / "data.yaml"
)

# ============================================================
# VALIDATION
# ============================================================

def validate_model():
    """
    Validate trained YOLOv8 segmentation model.
    """

    print("\nLoading model...\n")

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully!")

    print("\nRunning validation...\n")

    metrics = model.val(
        data=DATA_YAML
    )

    return metrics


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_metrics(metrics):
    """
    Print important evaluation metrics.
    """

    print("\n========== RESULTS ==========\n")

    print(
        f"Box mAP50      : "
        f"{metrics.box.map50:.4f}"
    )

    print(
        f"Box mAP50-95   : "
        f"{metrics.box.map:.4f}"
    )

    print()

    print(
        f"Mask mAP50     : "
        f"{metrics.seg.map50:.4f}"
    )

    print(
        f"Mask mAP50-95  : "
        f"{metrics.seg.map:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    metrics = validate_model()

    print_metrics(metrics)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()