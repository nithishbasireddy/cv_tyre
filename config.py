# Configuration parameters for Tyre Crack Inspection System

# Region of Interest (ROI) definitions
# Format: (x, y, width, height)
ROIS = [
    (568, 232, 128, 244),
    (388, 336, 145, 163)
]

# Image processing parameters
BLUR_KERNEL_SIZE = (5, 5)

# Gradient thresholding
# How much stronger the vertical gradient (horizontal cracks) needs to be compared to horizontal gradient
GRADIENT_RATIO_THRESHOLD = 1.5

# Binary thresholding value for highlighting the cracks
BINARY_THRESHOLD = 55

# Contour filtering parameters
MIN_CONTOUR_AREA = 80
MAX_ASPECT_RATIO = 3.0  # Width / Height ratio limit (cracks shouldn't be too wide compared to height)
