"""
predict.py

Run inference using the trained YOLOv8 segmentation model.

"""

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolov8s_seg_baseline-2"
    / "weights"
    / "best.pt"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "filtered_opg_dataset"
    / "test"
    / "images"
)

OUTPUT_DIR = PROJECT_ROOT / "predictions"

OUTPUT_NAME = "best_model_predictions"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained YOLOv8 segmentation model.
    """

    print("\nLoading model...\n")

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully!")

    return model


# ============================================================
# RUN PREDICTIONS
# ============================================================

def predict_folder(model):
    """
    Run predictions on all test images.
    """

    results = model.predict(

        source=TEST_IMAGES,

        imgsz=1024,

        conf=0.50,

        batch=1,

        stream=True,

        save=True,

        project=OUTPUT_DIR,

        name=OUTPUT_NAME
    )

    # Consume generator
    for _ in results:
        pass

    print("\nPrediction completed successfully!")

    print(
        f"\nSaved predictions to:\n"
        f"{OUTPUT_DIR}/{OUTPUT_NAME}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    predict_folder(model)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()