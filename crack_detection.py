import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading from:", INPUT_DIR)
print("Saving to:", OUTPUT_DIR)

for filename in os.listdir(INPUT_DIR):
    print("Found file:", filename)

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {filename}, unable to read image")
        continue

    print("Image loaded:", image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"{filename}: {len(contours)} contours found")

    output = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w + 1e-5)

        if aspect_ratio > 1.5 or aspect_ratio < 0.3:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, output)

    print("Saved:", output_path)

print("Processing completed.")
