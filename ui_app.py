import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QStatusBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class TyreInspectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tyre Crack Inspection System")
        self.setGeometry(100, 100, 1200, 700)

        self.video_path = None
        self.cap = None

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def init_ui(self):
        self.video_label = QLabel("No video loaded")
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Video")
        self.start_button = QPushButton("Start Inspection")
        self.stop_button = QPushButton("Stop")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        self.load_button.clicked.connect(self.load_video)
        self.start_button.clicked.connect(self.start_inspection)
        self.stop_button.clicked.connect(self.stop_inspection)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Inspection Video", "", "Video Files (*.avi *.mp4)"
        )

        if not path:
            return

        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.status.showMessage("Failed to open video")
            return

        self.start_button.setEnabled(True)
        self.status.showMessage(f"Loaded: {path}")

    def start_inspection(self):
        if not self.cap:
            return

        self.timer.start(100)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.showMessage("Inspection running")

    def stop_inspection(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.showMessage("Inspection stopped")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_inspection()
            self.cap.release()
            return

        processed = self.detect_crack(frame)
        self.display_frame(processed)

    def detect_crack(self, frame):
        # --- SAME LOGIC AS YOUR PIPELINE (SIMPLIFIED) ---
        ROIS = [
            (568, 232, 128, 244),
            (388, 336, 145, 163)
        ]

        output = frame.copy()

        for rx, ry, rw, rh in ROIS:
            roi = frame[ry:ry + rh, rx:rx + rw]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
            grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

            abs_x = np.abs(grad_x)
            abs_y = np.abs(grad_y)

            mask = abs_y > abs_x * 1.5
            horiz = np.zeros_like(abs_y)
            horiz[mask] = abs_y[mask]

            horiz = np.uint8(255 * horiz / (horiz.max() + 1e-5))
            _, binary = cv2.threshold(horiz, 55, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.rectangle(output, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 80:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w / (h + 1e-5) < 3.0:
                    continue

                cv2.rectangle(
                    output,
                    (rx + x, ry + y),
                    (rx + x + w, ry + y + h),
                    (0, 0, 255),
                    2
                )

        return output

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TyreInspectionApp()
    window.show()
    sys.exit(app.exec_())
