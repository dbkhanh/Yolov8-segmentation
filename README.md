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

```bash
python3 -m venv .venv
source .venv/bin/activate
```
If dependencies are missing:

`pip install -r requirements.txt`

## Model Training

`python3 src/train.py`

### Output Structure

```
out/
└── {N}_bags/
    ├── image_timestamp.jpg
    ├── image_timestamp.json
    └── yolo_pred/
        └── image_timestamp_yolo.png
```


### JSON Output example

```
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
```

## Run Inference

### Option 1: Run with Docker
```
docker run --rm \
  -v $(pwd):/app \
  produce \
  photo/images/val/bag97.png
```
### Option 2: Run Locally

```
python3 src/produce/produce.py photo/images/val/5bags_real.jpg
```

## Demo
Custom Annotation Output
<img width="1077" height="841" alt="image" src="https://github.com/user-attachments/assets/068c3ff1-f721-4c6b-b4b7-ecb407238e52" />

Raw YOLO Prediction
<img width="1080" height="838" alt="image" src="https://github.com/user-attachments/assets/381e9858-f70b-465d-bb42-296ce80760c0" />

Final Result with 1 image
<img width="782" height="760" alt="image" src="https://github.com/user-attachments/assets/6b011548-def6-4b8e-9206-c3e3d6e86ff7" />


## ⚠️ Limitations
The model was trained on a relatively small dataset composed mainly of AI-generated images.
While the pipeline is functionally complete, real-world performance may vary and results are not yet production-grade.
