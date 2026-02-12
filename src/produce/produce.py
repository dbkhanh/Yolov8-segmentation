#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect bags, draw center and angle"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--out-dir", default="out", help="Root output directory")
    return parser.parse_args()

# -----------------------------
# Geometry helpers
# -----------------------------
def sort_corners(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],  # TL
        pts[np.argmin(d)],  # TR
        pts[np.argmax(s)],  # BR
        pts[np.argmax(d)]   # BL
    ])

def bag_angle(corners):
    tl, tr, br, bl = corners
    edge_top = tr - tl
    edge_right = br - tr
    long_vec = edge_top if np.linalg.norm(edge_top) >= np.linalg.norm(edge_right) else edge_right
    return np.degrees(np.arctan2(long_vec[0], long_vec[1]))

def mask_iou(poly1, poly2, shape):
    m1 = np.zeros(shape[:2], np.uint8)
    m2 = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m1, [poly1], 1)
    cv2.fillPoly(m2, [poly2], 1)
    return np.sum(m1 & m2) / np.sum(m1 | m2)

def output_dir_for_count(base_dir, count):
    return Path(base_dir) / f"{count}_bags"

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    timestamp = datetime.now().isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    model = YOLO("./src/runs/segment/runs/bag-seg13/weights/best.pt")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # ---------- Run YOLO inference (no save yet) ----------
    r = model(args.image, save=False)[0]

    # ---------- Non-maximum suppression ----------
    kept = []
    IOU_THRESH = 0.5

    if r.masks is not None:
        detections = [
            {
                "poly": poly.astype(np.int32),
                "conf": float(r.boxes.conf[i]),
                "cls_id": int(r.boxes.cls[i]),
            }
            for i, poly in enumerate(r.masks.xy)
        ]

        detections.sort(key=lambda x: x["conf"], reverse=True)

        for det in detections:
            if all(
                det["cls_id"] != k["cls_id"]
                or mask_iou(det["poly"], k["poly"], img.shape) <= IOU_THRESH
                for k in kept
            ):
                kept.append(det)

    num_bags = len(kept)
    print(f"\nDetected bags (after NMS): {num_bags}\n")

    # ---------- Output directory ----------
    base_out = Path(args.out_dir)
    out_dir = output_dir_for_count(base_out, num_bags)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Run YOLO again to save pred image ----------
    r = model(args.image, save=False, save_dir=str(out_dir))[0]

    # Rename YOLO output to include timestamp
    yolo_raw = out_dir / "yolo_pred" / Path(args.image).name
    yolo_out = out_dir / "yolo_pred" / f"{Path(args.image).stem}_{ts}_yolo{Path(args.image).suffix}"
    if yolo_raw.exists():
        yolo_raw.rename(yolo_out)

    # ---------- Metadata ----------
    meta = {
        "image": args.image,
        "timestamp": timestamp,
        "num_bags": num_bags,
        "bags": []
    }

    # ---------- Draw custom annotations ----------
    for idx, det in enumerate(kept):
        poly = det["poly"]
        corners = sort_corners(poly)
        tl, tr, br, bl = corners

        center = ((tl + br) / 2).astype(int)
        angle = bag_angle(corners)

        meta["bags"].append({
            "id": idx + 1,
            "angle_deg": round(float(angle), 2),
            "center": {
                "x": int(center[0]),
                "y": int(center[1])
            }
        })

        print(f"Bag {idx + 1}: angle={angle:.2f}, center={tuple(center)}")

        cv2.polylines(img, [poly], True, (0, 255, 0), 2)
        for p in corners:
            cv2.circle(img, tuple(p), 5, (255, 0, 0), -1)

        cv2.line(img, tuple(tl), tuple(br), (255, 255, 0), 1)
        cv2.line(img, tuple(tr), tuple(bl), (255, 255, 0), 1)
        cv2.circle(img, tuple(center), 7, (0, 0, 255), -1)

        cv2.putText(
            img,
            f"angle={angle:.1f}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    # ---------- Save outputs ----------
    stem = Path(args.image).stem
    suffix = Path(args.image).suffix

    custom_img_path = out_dir / f"{stem}_{ts}{suffix}"
    json_path = out_dir / f"{stem}_{ts}.json"
    # -----------------------------
# Save YOLO prediction image
# -----------------------------
    yolo_pred_dir = out_dir / "yolo_pred"
    yolo_pred_dir.mkdir(exist_ok=True)

    yolo_img = r.plot()  # numpy image with YOLO overlays

    yolo_path = yolo_pred_dir / f"{stem}_{ts}_yolo.png"
    cv2.imwrite(str(yolo_path), yolo_img)

    print(f"Saved YOLO pred image → {yolo_path}")


    cv2.imwrite(str(custom_img_path), img)

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved custom image → {custom_img_path}")
    print(f"Saved YOLO pred image → {yolo_out}")
    print(f"Saved metadata → {json_path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
