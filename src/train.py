"""
train.py

Train YOLOv8s-Seg on the filtered OPG dataset.

"""

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "data" / "filtered_opg_dataset" / "data.yaml"

PROJECT_DIR = PROJECT_ROOT / "runs"

MODEL_NAME = "yolov8s-seg.pt"


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model():
    """
    Train YOLOv8 segmentation model.
    """

    model = YOLO(MODEL_NAME)

    results = model.train(

        # ====================================================
        # DATASET
        # ====================================================

        data=DATA_YAML,

        # ====================================================
        # MODEL TRAINING
        # ====================================================

        epochs=100,

        imgsz=1024,

        batch=4,

        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer="AdamW",

        lr0=0.001,

        # ====================================================
        # SEGMENTATION
        # ====================================================

        overlap_mask=True,

        mask_ratio=4,

        # ====================================================
        # AUGMENTATIONS
        # ====================================================

        degrees=7,

        translate=0.05,

        scale=0.10,

        fliplr=0.0,

        flipud=0.0,

        mosaic=0.2,

        mixup=0.0,

        hsv_h=0.0,

        hsv_s=0.0,

        hsv_v=0.1,

        perspective=0.0001,

        # ====================================================
        # HARDWARE
        # ====================================================

        device=0,

        workers=2,

        amp=True,

        cache=True,

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        patience=20,

        # ====================================================
        # OUTPUTS
        # ====================================================

        project=PROJECT_DIR,

        name="yolov8s_seg_baseline",

        # ====================================================
        # SAVE SETTINGS
        # ====================================================

        save=True,

        save_period=10
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    train_model()

    print("\nTraining Completed!")

    print(
        "\nBest Model Location:\n"
    )

    print(
        "/content/drive/MyDrive/"
        "OPG_Abhinaya/training_runs/"
        "yolov8s_seg_baseline/"
        "weights/best.pt"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()