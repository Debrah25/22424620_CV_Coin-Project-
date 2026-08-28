# runs all three segmentation methods on every image in data/raw,
# saves an overlay figure per method and a results.csv summary

import csv
import time
from pathlib import Path

import cv2

from utils import load_and_prep, get_objects, draw_overlay
import segment_threshold
import segment_region_growing
import segment_kmeans

RAW_DIR = Path("data/raw")
FIG_DIR = Path("outputs/figures")
RESULTS_CSV = Path("outputs/results.csv")

METHODS = [
    ("threshold", lambda gray: segment_threshold.segment(gray)),
    ("region_growing", lambda gray: segment_region_growing.segment(gray)),
    ("kmeans_k2", lambda gray: segment_kmeans.segment(gray, k=2)),
]


def process_image(path):
    img, gray = load_and_prep(path)
    rows = []

    for method_name, fn in METHODS:
        t0 = time.perf_counter()
        mask = fn(gray)
        elapsed = time.perf_counter() - t0

        objects = get_objects(mask)
        overlay = draw_overlay(img, objects, method_name)

        fig_name = f"{path.stem}_{method_name}.jpg"
        cv2.imwrite(str(FIG_DIR / fig_name), overlay)

        total_area = sum(o["area"] for o in objects)
        avg_perim = round(sum(o["perimeter"] for o in objects) / len(objects), 1) if objects else 0

        rows.append({
            "image": path.name,
            "method": method_name,
            "count": len(objects),
            "total_area": total_area,
            "avg_perimeter": avg_perim,
            "time_sec": round(elapsed, 4),
        })

    return rows


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(RAW_DIR.glob("*.jpg"))

    all_rows = []
    for path in images:
        print("processing", path.name)
        all_rows.extend(process_image(path))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("wrote", RESULTS_CSV)


if __name__ == "__main__":
    main()
