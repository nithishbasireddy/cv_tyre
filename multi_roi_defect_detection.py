import cv2
import numpy as np
import os

# -----------------------
# Images to test (2 images)
# -----------------------
IMAGE_PATHS = [
    "data/raw/Screenshot 2026-01-27 014409.png",
    "data/raw/Screenshot 2026-01-27 014426.png"
]

# -----------------------
# Fixed ROIs (x, y, w, h)
# -----------------------
ROIS = [
    (568, 232, 128, 244),
    (388, 336, 145, 163)
]

# -----------------------
# Loop over images
# -----------------------
for img_index, image_path in enumerate(IMAGE_PATHS):

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load:", image_path)
        continue

    output = image.copy()

    print(f"\nProcessing image {img_index + 1}: {image_path}")

    # -----------------------
    # Loop over ROIs
    # -----------------------
    for roi_index, (roi_x, roi_y, roi_w, roi_h) in enumerate(ROIS):

        roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # ---- Show ROI crop ----
        cv2.imshow(
            f"Image {img_index + 1} - ROI {roi_index + 1}",
            roi
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ---- Preprocess ROI ----
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # ---- Find contours (defects) ----
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ---- Draw ROI boundary (blue) ----
        cv2.rectangle(
            output,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            (255, 0, 0),
            2
        )
        # inside ROI
        valid_y_start = int(roi_h * 0.2)
        valid_y_end   = int(roi_h * 0.8)

        valid_mask = np.zeros_like(gray)
        valid_mask[valid_y_start:valid_y_end, :] = 255

        # ---- Draw defect boxes (red) ----
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w + 1e-5)

            if aspect_ratio > 1.5 or aspect_ratio < 0.4:
                cv2.rectangle(
                    output,
                    (roi_x + x, roi_y + y),
                    (roi_x + x + w, roi_y + y + h),
                    (0, 0, 255),
                    2
                )

    # -----------------------
    # Show final result for image
    # -----------------------
    cv2.imshow(f"Final Output - Image {img_index + 1}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Done")
