import cv2
import numpy as np

# -----------------------
# Load image
# -----------------------
IMAGE_PATH = "data/raw/Screenshot 2026-01-27 014426.png"
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not loaded")
    exit()

# -----------------------
# FIXED REGION OF INTEREST (HARD CODED)
# Adjust these ONCE after visual inspection
# -----------------------
roi_x = 300   # left
roi_y = 200   # top
roi_w = 450   # width
roi_h = 250   # height

roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# Show fixed ROI
cv2.imshow("1 - Fixed ROI (Inspection Area)", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Preprocessing INSIDE ROI
# -----------------------
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(2.0, (8, 8))
enhanced = clahe.apply(gray)

blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

edges = cv2.Canny(blurred, 40, 120)

cv2.imshow("2 - ROI Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Clean edges
# -----------------------
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# -----------------------
# Find contours (possible defects)
# -----------------------
contours, _ = cv2.findContours(
    cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------
# Draw defect boxes on ORIGINAL image
# -----------------------
output = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 40:
        continue  # ignore noise

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = h / float(w + 1e-5)

    # cracks are long & thin
    if aspect_ratio > 1.5 or aspect_ratio < 0.4:
        cv2.rectangle(
            output,
            (roi_x + x, roi_y + y),
            (roi_x + x + w, roi_y + y + h),
            (0, 0, 255),   # RED box
            2
        )

# Draw ROI boundary (for explanation)
cv2.rectangle(
    output,
    (roi_x, roi_y),
    (roi_x + roi_w, roi_y + roi_h),
    (255, 0, 0),  # BLUE ROI
    2
)

cv2.imshow("3 - Final Defect Detection (Fixed ROI)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection completed using fixed ROI.")
