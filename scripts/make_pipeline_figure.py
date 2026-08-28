# builds one figure showing the pipeline stages side by side for the report:
# original -> grayscale -> blurred -> binary mask -> final boundary overlay

import cv2
import numpy as np

from utils import load_and_prep, get_objects, draw_overlay
import segment_threshold

IMAGE = "data/raw/A2_baseline.jpg"
OUT = "outputs/figures/pipeline_stages_A2.jpg"


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def to_bgr(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)


def main():
    color, gray = load_and_prep(IMAGE)
    blurred = gray  # load_and_prep already blurs, so grab the pre-blur gray separately below
    raw_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    mask = segment_threshold.segment(gray)
    objects = get_objects(mask)
    overlay = draw_overlay(color, objects, "threshold")

    stages = [
        label(color, "1. original"),
        label(to_bgr(raw_gray), "2. grayscale"),
        label(to_bgr(gray), "3. blurred"),
        label(to_bgr(mask), "4. binary mask"),
        label(overlay, "5. boundaries + measurements"),
    ]

    strip = np.hstack(stages)
    cv2.imwrite(OUT, strip)
    print("wrote", OUT, strip.shape)


if __name__ == "__main__":
    main()
