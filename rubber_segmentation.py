import cv2
import numpy as np

IMAGE_PATH = "data/raw/Screenshot 2026-01-27 014409.png"

# -----------------------
# Load image
# -----------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Image not loaded")
    exit()

cv2.imshow("1 - Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Convert to HSV
# -----------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

cv2.imshow("2 - Saturation Channel", s)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("3 - Value Channel", v)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Rubber characteristics:
# low saturation + medium/low brightness
# -----------------------
_, sat_mask = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY_INV)
_, val_mask = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY_INV)

rubber_mask = cv2.bitwise_and(sat_mask, val_mask)

cv2.imshow("4 - Initial Rubber Mask", rubber_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Clean mask
# -----------------------
kernel = np.ones((7, 7), np.uint8)
rubber_mask = cv2.morphologyEx(rubber_mask, cv2.MORPH_CLOSE, kernel)
rubber_mask = cv2.morphologyEx(rubber_mask, cv2.MORPH_OPEN, kernel)

cv2.imshow("5 - Cleaned Rubber Mask", rubber_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Apply mask to original image
# -----------------------
rubber_only = cv2.bitwise_and(image, image, mask=rubber_mask)

cv2.imshow("6 - Rubber Surface Only", rubber_only)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Rubber segmentation completed.")
