import cv2
import numpy as np

IMAGE_PATH = "data/raw/Screenshot 2026-01-27 014426.png"

# -----------------------
# Load image
# -----------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("ERROR: Image not loaded")
    exit()

cv2.imshow("1 - Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Convert to grayscale
# -----------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2 - Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Smooth image
# -----------------------
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# -----------------------
# Detect edges
# -----------------------
edges = cv2.Canny(blurred, 40, 120)
cv2.imshow("3 - Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Hough Circle to find tyre
# -----------------------
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=200,
    param1=100,
    param2=40,
    minRadius=120,
    maxRadius=260
)

if circles is None:
    print("ERROR: Tyre circle not detected")
    exit()

circles = np.uint16(np.around(circles))
cx, cy, r = circles[0][0]

# Visualize detected circle
circle_vis = image.copy()
cv2.circle(circle_vis, (cx, cy), r, (0, 255, 0), 3)
cv2.circle(circle_vis, (cx, cy), 5, (0, 0, 255), -1)

cv2.imshow("4 - Detected Tyre Circle", circle_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Polar unwrapping
# -----------------------
unwrapped = cv2.warpPolar(
    gray,
    (360, r),
    (cx, cy),
    r,
    cv2.WARP_POLAR_LINEAR
)

cv2.imshow("5 - Unwrapped Tyre Surface", unwrapped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Enhance surface texture
# -----------------------
clahe = cv2.createCLAHE(2.0, (8, 8))
enhanced = clahe.apply(unwrapped)

cv2.imshow("6 - Enhanced Unwrapped Surface", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Crack detection on unwrapped image
# -----------------------
blur_u = cv2.GaussianBlur(enhanced, (5, 5), 0)
edges_u = cv2.Canny(blur_u, 40, 120)

kernel = np.ones((3, 3), np.uint8)
cleaned_u = cv2.morphologyEx(edges_u, cv2.MORPH_CLOSE, kernel)

cv2.imshow("7 - Cracks on Unwrapped Surface", cleaned_u)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Polar unwrapping and crack analysis completed.")
