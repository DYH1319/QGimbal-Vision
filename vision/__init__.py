"""Vision modules for this project.

This package contains small, reusable computer vision utilities used by `main.py`.
"""

from .rect_detect import DetectedRect, detect_rectangles, draw_detected_rect
from .detectors import (
    BaseDetector,
    CircleDetector,
    DetectionMode,
    DetectedTarget,
    ObjectDetector,
    RectDetector,
    create_detector,
)

__all__ = [
    "DetectedRect",
    "detect_rectangles",
    "draw_detected_rect",
    "BaseDetector",
    "CircleDetector",
    "DetectionMode",
    "DetectedTarget",
    "ObjectDetector",
    "RectDetector",
    "create_detector",
]
