# interactive tool to draw ground-truth masks by hand.
# click and drag from a coin's centre out to its edge to draw a circle over it,
# do that for every coin in the image, then press s to save and move to the next.
# r = undo last circle, q = quit early
#
# run this yourself, it needs a display: python3 scripts/make_gt_masks.py

import cv2
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
GT_DIR = Path("data/ground_truth")

# representative subset picked with the user, 1-2 per category
GT_IMAGES = [
    "A1_baseline.jpg",
    "A4_baseline.jpg",
    "B2_harsh.jpg",
    "B5_flash.jpg",
    "C2_cloth.jpg",
    "D3_touch.jpg",
    "E2_normal.jpg",
    "F4_verydark.jpg",
]

circles = []
drawing = False
center = None


def on_mouse(event, x, y, flags, param):
    global drawing, center, circles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        center = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        param["preview"] = (center, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        r = int(((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5)
        if r > 3:
            circles.append((center[0], center[1], r))
        param["preview"] = None


def annotate(path):
    global circles
    circles = []
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    win = f"GT: {path.name}"
    cv2.namedWindow(win)
    state = {"preview": None}
    cv2.setMouseCallback(win, on_mouse, state)

    while True:
        disp = img.copy()
        for (cx, cy, r) in circles:
            cv2.circle(disp, (cx, cy), r, (0, 255, 0), 2)
        if state["preview"]:
            (cx, cy), mx, my = state["preview"]
            r = int(((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5)
            cv2.circle(disp, (cx, cy), r, (0, 200, 255), 1)
        cv2.putText(disp, f"{len(circles)} coins marked  [s]=save [r]=undo [q]=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s"):
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for (cx, cy, r) in circles:
                cv2.circle(mask, (cx, cy), r, 255, -1)
            out_path = GT_DIR / (path.stem + "_mask.png")
            cv2.imwrite(str(out_path), mask)
            print("saved", out_path)
            break
        elif key == ord("r") and circles:
            circles.pop()
        elif key == ord("q"):
            cv2.destroyWindow(win)
            return False

    cv2.destroyWindow(win)
    return True


def main():
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for name in GT_IMAGES:
        path = RAW_DIR / name
        out_path = GT_DIR / (path.stem + "_mask.png")
        if out_path.exists():
            print("skipping, already done:", name)
            continue
        if not annotate(path):
            print("stopped early")
            break


if __name__ == "__main__":
    main()
