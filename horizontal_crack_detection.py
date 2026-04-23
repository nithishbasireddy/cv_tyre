import cv2
import numpy as np

# -----------------------
# Input images
# -----------------------
IMAGE_PATHS = [
    "data/raw/Screenshot 2026-01-27 014409.png",
    "data/raw/Screenshot 2026-01-27 014426.png"
]

# -----------------------
# Fixed inspection ROIs
# -----------------------
ROIS = [
    (568, 232, 128, 244),
    (388, 336, 145, 163)
]

for idx, image_path in enumerate(IMAGE_PATHS):

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load:", image_path)
        continue

    print(f"\nShowing demo for image {idx + 1}")

    # -----------------------
    # 1️⃣ Show original image
    # -----------------------
    cv2.imshow(f"Image {idx + 1} - Original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = image.copy()

    # -----------------------
    # Process each ROI
    # -----------------------
    for roi_id, (rx, ry, rw, rh) in enumerate(ROIS):

        roi = image[ry:ry + rh, rx:rx + rw]

        # -----------------------
        # Preprocessing
        # -----------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # -----------------------
        # Sobel gradients
        # -----------------------
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        abs_x = np.abs(grad_x)
        abs_y = np.abs(grad_y)

        # -----------------------
        # Keep horizontal edges only
        # -----------------------
        horiz = np.zeros_like(abs_y)
        mask = abs_y > abs_x * 1.5
        horiz[mask] = abs_y[mask]

        horiz = np.uint8(255 * horiz / (horiz.max() + 1e-5))

        # -----------------------
        # Show edge response (DEMO)
        # -----------------------
        cv2.imshow(
            f"Image {idx + 1} - ROI {roi_id + 1} - Horizontal Edges",
            horiz
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------
        # Threshold + clean
        # -----------------------
        _, binary = cv2.threshold(horiz, 40, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # -----------------------
        # Find contours
        # -----------------------
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

        # -----------------------
        # Draw crack candidates
        # -----------------------
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h + 1e-5)

            # Horizontal crack rule
            if aspect_ratio > 2.0:
                cv2.rectangle(
                    output,
                    (rx + x, ry + y),
                    (rx + x + w, ry + y + h),
                    (0, 0, 255),
                    2
                )

    # -----------------------
    # 3️⃣ Show final detection
    # -----------------------
    cv2.imshow(f"Image {idx + 1} - Final Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("completed.")
