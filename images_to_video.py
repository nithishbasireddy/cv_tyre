import cv2
import os

IMAGE_PATHS = [
    "data/raw/Screenshot 2026-01-27 014409.png",
    "data/raw/Screenshot 2026-01-27 014426.png"
]

OUTPUT_DIR = "data/output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "tyre_demo.avi")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read first image to define video size
first = cv2.imread(IMAGE_PATHS[0])
if first is None:
    raise RuntimeError("First image not found")

height, width = first.shape[:2]

fps = 5

# Use AVI + XVID (most stable on Windows)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not writer.isOpened():
    raise RuntimeError("VideoWriter failed to open")

frames_per_image = 100

for img_path in IMAGE_PATHS:
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Resize EVERY image to match first image
    img = cv2.resize(img, (width, height))

    for _ in range(frames_per_image):
        writer.write(img)

writer.release()

print("Video created successfully:", OUTPUT_VIDEO)
