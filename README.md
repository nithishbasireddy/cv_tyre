# Tyre Surface Crack Inspection — Computer Vision Project

A Python-based computer vision system for detecting cracks and surface defects on tyres using classical image processing techniques. The project includes a collection of standalone analysis scripts and a desktop GUI application.

---

## Overview

Tyres degrade over time, and visual inspection is one of the earliest ways to catch dangerous cracks before they cause failures. This project explores several algorithmic approaches to automate that inspection process — no machine learning involved, just carefully tuned image processing pipelines built on top of OpenCV.

The codebase evolved incrementally: starting from basic edge detection experiments and working up through ROI-based filtering, directional gradient analysis, polar unwrapping, and finally a real-time video inspection GUI.

---

## Tech Stack

- **Python 3**
- **OpenCV** (`cv2`) — image I/O, filtering, edge detection, contour analysis, morphological operations, polar warping, Hough transforms
- **NumPy** — array operations and gradient math
- **PyQt5** — desktop GUI (used in `ui_app.py` only)

---

## Project Structure

```
cv_tyre/
│
├── data/
│   ├── raw/           # Source tyre images
│   ├── processed/     # Output from batch processing
│   ├── output/        # Generated demo video (tyre_demo.avi)
│   └── outputs/       # Additional saved outputs
│
├── annotations/       # Bounding box annotation files
│
├── auto_roi.py                      # Contour-based automatic ROI detection
├── auto_roi_tyre.py                 # Hough circle + polar unwrapping approach
├── rubber_segmentation.py           # HSV-based rubber surface segmentation
├── crack.py                         # Interactive ROI selection + edge detection
├── roi_crack_detection.py           # Fixed ROI crack detection (single image)
├── crack_detection.py               # Batch image processor (raw → processed)
├── confidence.py                    # Top-hat morphology based crack detection
├── horizontal_crack_detection.py    # Sobel gradient direction filtering
├── fullhorizontal_crack_detection.py # Full image + ROI horizontal edge demo
├── multi_roi_defect_detection.py    # Multi-ROI defect detection with zone masking
├── images_to_video.py               # Converts source images into a demo AVI
├── video_horizontal_crack_detection.py # Frame-by-frame crack detection on video
├── ui_app.py                        # PyQt5 desktop GUI application
└── requirements.txt
```

---

## Detection Pipeline

Most scripts follow this sequence, applied inside one or more Regions of Interest (ROIs):

1. Convert frame/image to **grayscale**
2. Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance surface texture
3. **Gaussian blur** to suppress noise
4. **Edge detection** — either Canny or directional Sobel (horizontal emphasis)
5. **Morphological closing** to reconnect fragmented edges
6. **Contour extraction** — find all edge-connected regions
7. **Aspect ratio filter** — long, narrow shapes are classified as cracks
8. Draw **red bounding boxes** around detected cracks; **blue outlines** mark the ROI zones

---

## Scripts at a Glance

### Exploration

| Script | What it does |
|---|---|
| `auto_roi.py` | Finds the tyre automatically by locating the largest contour in the image |
| `auto_roi_tyre.py` | Detects the tyre as a circle using Hough transform, then polar-unwraps the curved surface into a flat image for easier analysis |
| `rubber_segmentation.py` | Isolates the rubber portion of the tyre by masking pixels with low saturation and medium-low brightness in HSV space |

### Crack Detection (Static Images)

| Script | What it does |
|---|---|
| `crack.py` | Lets you draw a custom ROI manually using `cv2.selectROI`, then runs edge detection within it |
| `roi_crack_detection.py` | Uses a single hard-coded rectangular ROI; step-by-step intermediate windows for each processing stage |
| `crack_detection.py` | Batch mode — reads every image in `data/raw/`, runs detection, saves annotated results to `data/processed/` |
| `confidence.py` | Uses **top-hat morphological filtering** with a wide horizontal kernel to pull out fine surface cracks |
| `horizontal_crack_detection.py` | Filters Sobel gradients by direction — only keeps pixels where the vertical gradient dominates (i.e., horizontal edges = horizontal cracks) |
| `fullhorizontal_crack_detection.py` | Extends the above with a full-image edge visualization before zooming into ROIs |
| `multi_roi_defect_detection.py` | Runs over two fixed ROIs per image with a valid zone mask that ignores the top and bottom 20% of each ROI to avoid border artifacts |

### Video

| Script | What it does |
|---|---|
| `images_to_video.py` | Takes the two source images and writes each one as 100 repeated frames into `data/output/tyre_demo.avi` at 5 FPS |
| `video_horizontal_crack_detection.py` | Opens the generated video and runs the horizontal crack detection pipeline on every frame in real time |

### GUI

| Script | What it does |
|---|---|
| `ui_app.py` | A full desktop application — load any AVI/MP4 video, hit Start to begin inspection, and watch annotated frames update live in the window. Uses `QTimer` at 10 FPS to drive frame processing |

---

## Running the Project

### Install dependencies

```bash
pip install opencv-python numpy PyQt5
```

### Run the GUI app

```bash
python ui_app.py
```

Use the **Load Video** button to select a tyre inspection video, then click **Start Inspection**.

### Run a detection script directly

```bash
python horizontal_crack_detection.py
```

Each script opens OpenCV windows showing intermediate stages — press any key to advance through them.

### Batch process images

```bash
python crack_detection.py
```

Reads from `data/raw/`, writes annotated results to `data/processed/`.

---

## Notes

- The fixed ROIs `(568, 232, 128, 244)` and `(388, 336, 145, 163)` are tuned for the specific tyre images included in `data/raw/`. You will need to adjust these coordinates for different input images.
- All detection logic uses classical CV only — no deep learning or pretrained models.
- Scripts are self-contained with no shared modules between them; the logic is intentionally duplicated for clarity during development.
