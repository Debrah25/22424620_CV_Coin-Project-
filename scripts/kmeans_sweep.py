# separate experiment: how does K affect kmeans segmentation
# runs k=2,3,4 on every image, no overlays, just counts/areas/time for the writeup

import csv
import time
from pathlib import Path

from utils import load_and_prep, get_objects
import segment_kmeans

RAW_DIR = Path("data/raw")
OUT_CSV = Path("outputs/kmeans_k_sweep.csv")


def main():
    rows = []
    for path in sorted(RAW_DIR.glob("*.jpg")):
        _, gray = load_and_prep(path)
        for k in [2, 3, 4]:
            t0 = time.perf_counter()
            mask = segment_kmeans.segment(gray, k=k)
            elapsed = time.perf_counter() - t0

            objs = get_objects(mask)
            rows.append({
                "image": path.name,
                "k": k,
                "count": len(objs),
                "total_area": sum(o["area"] for o in objs),
                "time_sec": round(elapsed, 4),
            })
        print("done", path.name)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
