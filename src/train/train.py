import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load pretrained segmentation model
model = YOLO("model/yolo26n-seg.pt") 
# Train
model.train(
    data="../photo/data.yaml",
    epochs=100,
    imgsz=256,
    batch=4,
    device="cpu",
    mask_ratio=4,
    project="runs",
    name="bag-seg"
)

# Evaluate on validation set
model.val()
