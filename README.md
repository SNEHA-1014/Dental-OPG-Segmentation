# Dental OPG Multi-Class Instance Segmentation using YOLOv8

## Overview

This project focuses on automatic detection and instance segmentation of dental conditions and anatomical structures from Orthopantomogram (OPG) panoramic dental X-rays using YOLOv8 Segmentation.

The original dataset contained 31 classes with severe class imbalance. To build a reliable baseline model, the dataset was filtered to retain only the 10 most frequently occurring classes.

The project includes:

* Dataset filtering and preprocessing
* Exploratory Data Analysis (EDA)
* YOLOv8 Segmentation Training
* Model Validation
* Inference on Test Images
* Ground Truth vs Prediction Visualization

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── OPG_data_roboflow.ipynb
│
└── src/
    ├── dataset.py
    ├── train.py
    ├── validate.py
    ├── predict.py
    └── visualize.py
```

---

## Dataset Information

### Original Dataset

* Total Images: 13,830
* Total Classes: 31
* Annotation Format: YOLO Segmentation
* Imaging Modality: Dental Panoramic X-ray (OPG)

The original dataset is not included in this repository due to its size.

For local execution, place the original dataset under:

```text
data/
└── original/
```

### Dataset Filtering

The dataset was filtered based on class occurrence frequency to address severe class imbalance. Only images containing at least one of the selected classes were retained.

The filtered dataset is generated under:

```text
data/
└── filtered_opg_dataset/
```

---

## Selected Classes

The following 10 classes were selected based on occurrence frequency.

| New ID | Class Name           | Instances |
| ------ | -------------------- | --------: |
| 0      | Filling              |    48,694 |
| 1      | Impacted Tooth       |    27,872 |
| 2      | Root Canal Treatment |    18,887 |
| 3      | Crown                |    11,155 |
| 4      | Caries               |    10,592 |
| 5      | Periapical Lesion    |     5,215 |
| 6      | Missing Teeth        |     3,456 |
| 7      | Bone Loss            |     3,120 |
| 8      | Root Piece           |     2,599 |
| 9      | Implant              |     1,768 |

---

## Dataset After Filtering

Only images containing at least one selected class were retained.

### Final Dataset

| Split      | Images |
| ---------- | -----: |
| Train      |  9,340 |
| Validation |  2,756 |
| Test       |  1,376 |
| Total      | 13,472 |

---

## Model Configuration

### Model

```text
YOLOv8s-Seg
```

### Input Resolution

```text
1024 × 1024
```

### Optimizer

```text
AdamW
```

### Training Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Epochs        | 100   |
| Batch Size    | 4     |
| Learning Rate | 0.001 |
| Optimizer     | AdamW |
| Patience      | 20    |
| Image Size    | 1024  |

---

## Data Augmentation

The following augmentations were used during training:

| Augmentation    | Value    |
| --------------- | -------- |
| Rotation        | ±7°      |
| Translation     | 0.05     |
| Scaling         | 0.10     |
| Mosaic          | 0.20     |
| Perspective     | 0.0001   |
| Horizontal Flip | Disabled |
| Vertical Flip   | Disabled |

---

## Results

### Validation Performance

| Metric           | Value |
| ---------------- | ----: |
| Precision (Box)  | 0.694 |
| Recall (Box)     | 0.683 |
| mAP@50 (Box)     | 0.673 |
| mAP@50-95 (Box)  | 0.404 |
| Precision (Mask) | 0.674 |
| Recall (Mask)    | 0.657 |
| mAP@50 (Mask)    | 0.642 |
| mAP@50-95 (Mask) | 0.334 |

### Validation Dataset

| Metric    |  Value |
| --------- | -----: |
| Images    |  2,756 |
| Instances | 27,374 |

---

## Visualization

The project provides visualization utilities for:

* Random Ground Truth Samples
* Prediction Visualization
* Original vs Ground Truth vs Prediction Comparison

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place the original YOLO-format OPG dataset inside:

```text
data/original/
```

Then run:

```bash
python src/dataset.py
```

This creates the filtered dataset under:

```text
data/filtered_opg_dataset/
```

### 3. Train the Model

```bash
python src/train.py
```

Training outputs are saved under:

```text
runs/
```

### 4. Validate the Model

```bash
python src/validate.py
```

### 5. Run Predictions

```bash
python src/predict.py
```

Prediction outputs are saved under:

```text
predictions/
```

### 6. Visualize Results

```bash
python src/visualize.py
```

---

## Future Improvements

* YOLOv11 Segmentation Experiments
* Class Balancing Techniques
* Advanced Dental Image Preprocessing
* Test-Time Augmentation (TTA)
* Ensemble Models
* Clinical Validation

---

## Author

**Jaajala Sneha Sri**

B.Tech – Artificial Intelligence & Machine Learning
