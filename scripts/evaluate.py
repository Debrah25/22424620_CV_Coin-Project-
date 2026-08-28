# compares each method's segmentation mask against the hand-drawn ground truth
# masks and works out IoU / precision / recall / F1

import csv
from pathlib import Path

import cv2
import numpy as np

from utils import load_and_prep
import segment_threshold
import segment_region_growing
import segment_kmeans

RAW_DIR = Path("data/raw")
GT_DIR = Path("data/ground_truth")
OUT_CSV = Path("outputs/evaluation.csv")

METHODS = [
    ("threshold", lambda gray: segment_threshold.segment(gray)),
    ("region_growing", lambda gray: segment_region_growing.segment(gray)),
    ("kmeans_k2", lambda gray: segment_kmeans.segment(gray, k=2)),
]


def iou_precision_recall_f1(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union else 0

    precision = intersection / pred.sum() if pred.sum() else 0
    recall = intersection / gt.sum() if gt.sum() else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return iou, precision, recall, f1


def main():
    gt_files = sorted(GT_DIR.glob("*_mask.png"))
    if not gt_files:
        print("no ground truth masks found in data/ground_truth yet")
        print("run scripts/make_gt_masks.py first")
        return

    rows = []
    for gt_path in gt_files:
        image_name = gt_path.stem.replace("_mask", "") + ".jpg"
        img_path = RAW_DIR / image_name
        if not img_path.exists():
            print("skipping, raw image missing:", image_name)
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        _, gray = load_and_prep(img_path)

        if gt.shape != gray.shape:
            gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]))

        for method_name, fn in METHODS:
            mask = fn(gray)
            iou, prec, rec, f1 = iou_precision_recall_f1(mask, gt)
            rows.append({
                "image": image_name,
                "method": method_name,
                "iou": round(iou, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
            })
        print("evaluated", image_name)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
