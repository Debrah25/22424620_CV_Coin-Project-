# shared helper functions used by all three segmentation scripts

import cv2
import numpy as np


def load_and_prep(path, max_dim=900):
    # loads image, resizes so the long side is max_dim (region growing gets
    # way too slow on full 3000px phone photos), converts to gray + blurs it
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray


def foreground_from_border(mask):
    # assume background touches the image edges more than the coins do.
    # works out which side of a binary mask is "foreground" without
    # hardcoding whether coins are lighter or darker than the background
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    on_frac = mask[border].mean() / 255
    if on_frac > 0.5:
        return cv2.bitwise_not(mask)
    return mask


def foreground_from_border_multi(labels, k):
    # same idea but for k-means labels (0..k-1) instead of a binary mask
    h, w = labels.shape
    border_vals = np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ])
    bg_label = np.bincount(border_vals, minlength=k).argmax()
    mask = np.where(labels == bg_label, 0, 255).astype(np.uint8)
    return mask


def get_objects(mask, min_area=80):
    # connected components -> list of dicts with area/perimeter/centroid/bbox
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    objects = []
    for i in range(1, n):  # skip label 0, that's the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        blob = np.uint8(labels == i) * 255
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = cv2.arcLength(contours[0], True) if contours else 0

        objects.append({
            "area": int(area),
            "perimeter": round(perim, 1),
            "centroid": (round(centroids[i][0], 1), round(centroids[i][1], 1)),
            "bbox": tuple(stats[i, :4]),
            "contour": contours[0] if contours else None,
        })

    return objects


def draw_overlay(img, objects, method_name):
    out = img.copy()
    for i, obj in enumerate(objects):
        if obj["contour"] is not None:
            cv2.drawContours(out, [obj["contour"]], -1, (0, 255, 0), 2)
        x, y, w, h = obj["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cx, cy = obj["centroid"]
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, str(i + 1), (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    label = f"{method_name}: {len(objects)} objects"
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out
