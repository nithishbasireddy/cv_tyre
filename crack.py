import cv2
import numpy as np

IMAGE_PATH = "data/raw/Screenshot 2026-01-27 014426.png"
#IMAGE_PATH = "data/raw/Screenshot 2026-01-27 014409.png"

# -----------------------
# Load image
# -----------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not loaded")
    exit()

cv2.imshow("Original Image", image)
cv2.waitKey(1)

# -----------------------
# Select ROI (returns coordinates)
# -----------------------
roi_coords = cv2.selectROI(
    "Select ROI (press ENTER)",
    image,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyAllWindows()

roi_x, roi_y, roi_w, roi_h = roi_coords

print("ROI coordinates:", roi_coords)

x, y, w, h = roi_coords

# -----------------------
# Extract ROI IMAGE (this is critical)
# -----------------------
roi_image = image[y:y+h, x:x+w]

cv2.imshow("ROI Image", roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# -----------------------
# NOW processing starts
# -----------------------
gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("ROI Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Contrast enhancement
clahe = cv2.createCLAHE(2.0, (8, 8))
enhanced = clahe.apply(gray)

cv2.imshow("ROI - Contrast Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Noise reduction
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

cv2.imshow("ROI - Blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Edge detection
edges = cv2.Canny(blurred, 40, 120)

cv2.imshow("ROI - Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphological cleaning
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imshow("ROI - Cleaned Edges", cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours, _ = cv2.findContours(
    cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # shift box back to original image coordinates
    cv2.rectangle(
        output,
        (roi_x + x, roi_y + y),
        (roi_x + x + w, roi_y + y + h),
        (0, 0, 255),
        2
    )

cv2.imshow("Final Detection (ROI-based)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
