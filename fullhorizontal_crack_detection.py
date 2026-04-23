import cv2
import numpy as np

# -----------------------------
# Input images
# -----------------------------
IMAGE_PATHS = [
    "data/raw/Screenshot 2026-01-27 014409.png",
    "data/raw/Screenshot 2026-01-27 014426.png"
]

# -----------------------------
# Fixed inspection ROIs
# (x, y, width, height)
# -----------------------------
ROIS = [
    (568, 232, 128, 244),
    (388, 336, 145, 163)
]

for img_no, img_path in enumerate(IMAGE_PATHS):

    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load:", img_path)
        continue

    print(f"\nShowing demo for image {img_no + 1}")

    # ------------------------------------------------
    # 1. Show original image
    # ------------------------------------------------
    cv2.imshow(f"Image {img_no + 1} - Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------------------------
    # 2. FULL IMAGE EDGE VISUALISATION
    # ------------------------------------------------
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_full = cv2.GaussianBlur(gray_full, (5, 5), 0)

    edges_full = cv2.Canny(blur_full, 50, 150)

    cv2.imshow(f"Image {img_no + 1} - Full Image Edges", edges_full)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------------------------
    # 3. Directional edges (horizontal emphasis)
    # ------------------------------------------------
    grad_x_full = cv2.Sobel(blur_full, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_full = cv2.Sobel(blur_full, cv2.CV_64F, 0, 1, ksize=3)

    abs_x_full = np.abs(grad_x_full)
    abs_y_full = np.abs(grad_y_full)

    horizontal_edges_full = np.zeros_like(abs_y_full)
    mask = abs_y_full > abs_x_full * 1.5
    horizontal_edges_full[mask] = abs_y_full[mask]

    horizontal_edges_full = np.uint8(
        255 * horizontal_edges_full / (horizontal_edges_full.max() + 1e-5)
    )

    cv2.imshow(
        f"Image {img_no + 1} - Full Image Horizontal Edges",
        horizontal_edges_full
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------------------------
    # 4. Final detection using ROIs only
    # ------------------------------------------------
    output = image.copy()

    for roi_id, (rx, ry, rw, rh) in enumerate(ROIS):

        roi = image[ry:ry + rh, rx:rx + rw]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

        abs_x = np.abs(grad_x)
        abs_y = np.abs(grad_y)

        horiz = np.zeros_like(abs_y)
        mask = abs_y > abs_x * 1.5
        horiz[mask] = abs_y[mask]

        horiz = np.uint8(255 * horiz / (horiz.max() + 1e-5))

        _, binary = cv2.threshold(horiz, 40, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw ROI boundary
        cv2.rectangle(
            output,
            (rx, ry),
            (rx + rw, ry + rh),
            (255, 0, 0),
            2
        )

        # Draw detected horizontal crack candidates
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h + 1e-5)

            if aspect_ratio > 2.0:
                cv2.rectangle(
                    output,
                    (rx + x, ry + y),
                    (rx + x + w, ry + y + h),
                    (0, 0, 255),
                    2
                )

    cv2.imshow(f"Image {img_no + 1} - Final ROI Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("finished.")
