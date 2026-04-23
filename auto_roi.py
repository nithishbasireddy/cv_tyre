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
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Convert to grayscale
# -----------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Blur to reduce noise
# -----------------------
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# -----------------------
# Edge detection
# -----------------------
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Find contours
# -----------------------
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------
# Find largest contour (assumed tyre)
# -----------------------
largest_contour = None
max_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        largest_contour = cnt

# Safety check
if largest_contour is None:
    print("No contour found")
    exit()

# -----------------------
# Get bounding box of tyre
# -----------------------
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw auto ROI
output = image.copy()
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow("Automatic ROI (Tyre Region)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Extract ROI image
# -----------------------
roi_image = image[y:y+h, x:x+w]

cv2.imshow("Extracted Tyre ROI", roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Auto ROI coordinates:", (x, y, w, h))
