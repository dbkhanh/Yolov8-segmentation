# YOLOv8 Bag Segmentation & Orientation Detection
YOLO-based bag segmentation and orientation detection pipeline. Trains a custom segmentation model and performs inference with post-processing (mask NMS), angle calculation, center detection, and structured JSON output.

This project extends standard object detection by computing:

- 📍 Object center coordinates
- 📐 Rotation angle estimation
- 🧮 Custom mask-based Non-Maximum Suppression (IoU)
- 🖼️ Annotated output images
- 📄 Structured JSON metadata

Designed for real-world automation and production integration.

## Technical Overview
### Model
- YOLOv8 segmentation (Ultralytics)
- Custom-trained segmentation weights
- Dataset defined via data.yaml

### Post-Processing Pipeline
After inference:
- Extract mask polygons
- Apply custom mask IoU-based NMS
- Sort polygon corners (TL, TR, BR, BL)

Compute:
- Center using diagonal midpoint
- Orientation from longest edge vector

Save:
- Custom annotated image
- Raw YOLO overlay image
- Structured JSON metadata

## Installation
Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

If dependencies are missing:

pip install -r requirements.txt

## Model Training

python3 src/train.py

### Output Structure

out/
└── {N}_bags/
    ├── image_timestamp.jpg
    ├── image_timestamp.json
    └── yolo_pred/
        └── image_timestamp_yolo.png


### JSON Output example

{
  "image": "photo/images/val/5bags_real.jpg",
  "timestamp": "2026-02-12T10:21:00",
  "num_bags": 5,
  "bags": [
    {
      "id": 1,
      "angle_deg": -12.45,
      "center": {
        "x": 345,
        "y": 210
      }
    }
  ]
}


### Demo
Custom Annotation Output

Raw YOLO Prediction


