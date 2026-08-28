# method 2: region growing, seeded automatically from an eroded otsu mask
# uses cv2.floodFill to do the actual pixel-by-pixel growing

import cv2
import numpy as np
from utils import foreground_from_border

TOL = 40  # intensity tolerance for growing, picked by eye on the baseline images


def find_seeds(gray):
    _, rough = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough = foreground_from_border(rough)

    # shrink blobs down so each coin gives one seed point, not the whole blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded = cv2.erode(rough, kernel, iterations=2)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    seeds = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 30:
            continue
        cx, cy = centroids[i]
        seeds.append((int(cx), int(cy)))
    return seeds


def segment(gray):
    seeds = find_seeds(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    for sx, sy in seeds:
        ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), dtype=np.uint8)
        # FIXED_RANGE compares each pixel back to the seed value rather than its
        # neighbour, otherwise the fill drifts gradually across blurred coin edges
        # and leaks into the background (found this the hard way during testing)
        cv2.floodFill(gray.copy(), ff_mask, (sx, sy), 255,
                      loDiff=TOL, upDiff=TOL,
                      flags=4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
        region = ff_mask[1:-1, 1:-1]  # already 0/255 since newMaskVal was set to 255
        mask = cv2.bitwise_or(mask, region)

    return mask
