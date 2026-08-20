# Hard-Hat Detection with YOLOv8

Object detection for construction sites: classify workers as wearing a helmet, bare head, or person (context class).

![Validation labels — ground truth bounding boxes on site images](https://github.com/felipebasurto/hardhat-object-detection-yolov8/assets/62935664/761fcefa-3eda-4054-8758-7cacdbba27b9)

## What this repo contains

- **`hardhat-workers-detection-yolov8.ipynb`** — train and compare two Ultralytics detectors on the same dataset
- **`output/`** — YOLOv8 training curves, confusion matrix, and validation predictions from the run captured in the notebook

## Training setup

Both models were trained on the **Hard Hat Workers** dataset (75/25 train–test split, YOLO format) with three classes: `head`, `helmet`, and `person`.

| Model | Backbone | Epochs | Image size |
|-------|----------|--------|------------|
| YOLOv8 | `yolov8s.pt` | 30 | 640 |
| YOLOv5 | `yolov5su.pt` | 30 | 640 |

After 30 epochs on the validation set (1,766 images):

| Model | mAP@50 | Helmet mAP@50 | Head mAP@50 |
|-------|--------|---------------|-------------|
| YOLOv8 | 0.664 | 0.986 | 0.973 |
| YOLOv5 | 0.663 | 0.986 | 0.972 |

Overall metrics are nearly identical; the notebook compares confusion matrices and side-by-side validation predictions. Helmet vs. head classification is strong; the `person` class is underrepresented in the dataset and performs poorly on both models.

## How to run

1. **Clone the repo**
   ```bash
   git clone https://github.com/felipebasurto/hardhat-object-detection-yolov8.git
   cd hardhat-object-detection-yolov8
   ```

2. **Install dependencies**
   ```bash
   pip install ultralytics
   ```

3. **Get the dataset** — download the Hard Hat Workers dataset in YOLOv8 format (must include `data.yaml`, train/val image folders, and labels). The notebook expects it unzipped locally.

4. **Open the notebook** — `hardhat-workers-detection-yolov8.ipynb`
   - Update the `HOME` variable to point at your unzipped dataset directory.
   - The notebook was written for Google Colab; skip or replace the Google Drive mount cell if running locally.
   - A CUDA GPU is recommended for training (`nvidia-smi` cell checks availability).

5. **Train or inspect results** — run cells sequentially. Training writes weights and plots to `runs/detect/`. To run inference on new images without retraining, load the saved weights from that folder (see the final notebook section).

## Limitations

Performance depends on video/image quality, lighting, and occlusions. Small or partially hidden heads are harder to classify, and the model was validated on the Hard Hat Workers split—not deployed on live site footage in this repo.

## Stack

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) · Python · Jupyter
