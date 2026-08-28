# none of the A-F categories test noise directly, so this adds synthetic
# gaussian noise at a few levels to baseline images and checks how much
# each method's output changes. uses IoU against ground truth if a mask
# exists for that image, otherwise just tracks object count drift.

import csv
from pathlib import Path

import cv2
import numpy as np

from utils import load_and_prep
from evaluate import iou_precision_recall_f1
import segment_threshold
import segment_region_growing
import segment_kmeans

RAW_DIR = Path("data/raw")
GT_DIR = Path("data/ground_truth")
OUT_CSV = Path("outputs/noise_sensitivity.csv")

TEST_IMAGES = ["A1_baseline.jpg", "A4_baseline.jpg"]
NOISE_LEVELS = [0, 10, 20, 30, 40]

METHODS = [
    ("threshold", lambda gray: segment_threshold.segment(gray)),
    ("region_growing", lambda gray: segment_region_growing.segment(gray)),
    ("kmeans_k2", lambda gray: segment_kmeans.segment(gray, k=2)),
]


def add_gaussian_noise(gray, sigma):
    if sigma == 0:
        return gray
    noise = np.random.normal(0, sigma, gray.shape)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    rows = []
    for name in TEST_IMAGES:
        _, gray = load_and_prep(RAW_DIR / name)

        gt_path = GT_DIR / (Path(name).stem + "_mask.png")
        gt = None
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt.shape != gray.shape:
                gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]))

        for sigma in NOISE_LEVELS:
            noisy_gray = add_gaussian_noise(gray, sigma)
            for method_name, fn in METHODS:
                mask = fn(noisy_gray)
                n_objects = cv2.connectedComponentsWithStats(mask)[0] - 1

                row = {"image": name, "noise_sigma": sigma, "method": method_name,
                       "object_count": n_objects}
                if gt is not None:
                    iou, prec, rec, f1 = iou_precision_recall_f1(mask, gt)
                    row.update(iou=round(iou, 3), precision=round(prec, 3),
                               recall=round(rec, 3), f1=round(f1, 3))
                rows.append(row)
        print("done", name)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
