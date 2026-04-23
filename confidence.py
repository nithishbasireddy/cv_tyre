import cv2
import numpy as np

IMAGE_PATHS = [
    "data/raw/Screenshot 2026-01-27 014409.png",
    "data/raw/Screenshot 2026-01-27 014426.png"
]

ROIS = [
    (568, 232, 128, 244),
    (388, 336, 145, 163)
]

for idx, path in enumerate(IMAGE_PATHS):

    image = cv2.imread(path)
    if image is None:
        continue

    cv2.imshow(f"Image {idx+1} - Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = image.copy()

    for rx, ry, rw, rh in ROIS:

        roi = image[ry:ry+rh, rx:rx+rw]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ---- Top-hat filtering (extract surface cracks) ----
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Normalize
        tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

        # Threshold
        _, binary = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

        # Clean
        clean_kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, clean_kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw ROI
        cv2.rectangle(output, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = w / float(h + 1e-5)
            if aspect_ratio < 2.5:
                continue

            cv2.rectangle(
                output,
                (rx + x, ry + y),
                (rx + x + w, ry + y + h),
                (0, 0, 255),
                2
            )

        # Show intermediate (IMPORTANT FOR DEMO)
        cv2.imshow(f"Image {idx+1} - ROI TopHat Response", tophat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imshow(f"Image {idx+1} - Final Crack Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("completed.")
