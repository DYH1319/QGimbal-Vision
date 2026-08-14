"""Multi-mode detection pipeline for QGimbal-Vision.

Supports rectangle (legacy), circle, and specific-object detection using ORB
feature matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .rect_detect import DetectedRect, detect_rectangles


@dataclass(frozen=True)
class DetectedTarget:
    """Generic detection result used by the control loop."""

    kind: str  # 'rect', 'circle', 'object'
    center: Tuple[float, float]
    area: float
    points: Optional[np.ndarray] = None  # (N, 2) float32; used for drawing
    radius: float = 0.0  # used when kind == 'circle'
    label: str = ""
    confidence: float = 0.0

    @property
    def center_int(self) -> Tuple[int, int]:
        return int(round(self.center[0])), int(round(self.center[1]))


class DetectionMode(str, Enum):
    """Detection modes available at runtime."""

    RECT = "rect"
    CIRCLE = "circle"
    OBJECT = "object"


class BaseDetector:
    """Base class for detection modes."""

    def detect(self, frame: np.ndarray) -> List[DetectedTarget]:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook for stateful detectors."""

    def draw(self, frame: np.ndarray, target: DetectedTarget | None) -> None:
        """Draw a single best target on the frame in-place."""
        if target is None:
            return

        color = (0, 255, 0)
        cx, cy = target.center

        if target.kind == "circle" and target.radius > 0:
            cv2.circle(frame, (int(cx), int(cy)), int(target.radius), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        elif target.points is not None and len(target.points) >= 3:
            pts = target.points.astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        label = f"{target.label}:{int(target.area)}"
        cv2.putText(
            frame,
            label,
            (int(cx) - 60, int(cy) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        if target.confidence > 0:
            cv2.putText(
                frame,
                f"conf:{target.confidence:.2f}",
                (int(cx) - 60, int(cy) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
            )


class RectDetector(BaseDetector):
    """Wraps the legacy rectangle detector."""

    def __init__(
        self,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.5,
        angle_tol: float = 25.0,
    ) -> None:
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._angle_tol = angle_tol

    def detect(self, frame: np.ndarray) -> List[DetectedTarget]:
        rects = detect_rectangles(
            frame,
            min_area_ratio=self._min_area_ratio,
            max_area_ratio=self._max_area_ratio,
            angle_tol=self._angle_tol,
        )
        return [
            DetectedTarget(
                kind="rect",
                center=r.center,
                area=r.area,
                points=r.box,
                label="rect",
                confidence=1.0,
            )
            for r in rects
        ]


class CircleDetector(BaseDetector):
    """Detect round-ish objects using contour circularity."""

    def __init__(
        self,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.5,
        min_circularity: float = 0.7,
        min_radius: int = 10,
        max_radius: int = 300,
    ) -> None:
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._min_circularity = min_circularity
        self._min_radius = min_radius
        self._max_radius = max_radius

    @staticmethod
    def _preprocess_contours(frame: np.ndarray):
        """Shared pipeline: returns contours from a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        opened = cv2.erode(opened, kernel, iterations=1)
        edges = cv2.Canny(opened, 25, 75)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect(self, frame: np.ndarray) -> List[DetectedTarget]:
        h, w = frame.shape[:2]
        img_area = h * w
        min_area = img_area * float(self._min_area_ratio)
        max_area = img_area * float(self._max_area_ratio)

        contours = self._preprocess_contours(frame)
        circles: List[DetectedTarget] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = (4.0 * math.pi * area) / (peri * peri)
            if circularity < self._min_circularity:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self._min_radius or radius > self._max_radius:
                continue

            circles.append(
                DetectedTarget(
                    kind="circle",
                    center=(float(cx), float(cy)),
                    area=area,
                    radius=float(radius),
                    label="circle",
                    confidence=circularity,
                )
            )

        circles.sort(key=lambda c: c.area, reverse=True)
        return circles


class ObjectDetector(BaseDetector):
    """Detect a specific object from a reference image using ORB feature matching.

    The reference image should be a clear, front-facing view of the target object
    (e.g. a cup, a card, a marker) on a relatively plain background.
    """

    DEFAULT_MIN_MATCHES = 10
    DEFAULT_RATIO_TEST = 0.75
    DEFAULT_RANSAC_REPROJ = 5.0

    def __init__(
        self,
        reference_path: str,
        min_matches: int = DEFAULT_MIN_MATCHES,
        ratio_test: float = DEFAULT_RATIO_TEST,
        ransac_reproj: float = DEFAULT_RANSAC_REPROJ,
    ) -> None:
        self._min_matches = min_matches
        self._ratio_test = ratio_test
        self._ransac_reproj = ransac_reproj

        ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise ValueError(f"无法加载参考图像: {reference_path}")

        self._ref_h, self._ref_w = ref_img.shape[:2]
        if self._ref_h < 10 or self._ref_w < 10:
            raise ValueError("参考图像尺寸过小")

        self._orb = cv2.ORB_create(nfeatures=1000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(ref_img, None)
        if self._ref_desc is None or len(self._ref_kp) < self._min_matches:
            raise ValueError("参考图像特征点过少，请换一张更清晰的图片")

        self._ref_corners = np.float32(
            [[0, 0], [self._ref_w, 0], [self._ref_w, self._ref_h], [0, self._ref_h]]
        ).reshape(-1, 1, 2)

    def detect(self, frame: np.ndarray) -> List[DetectedTarget]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < self._min_matches:
            return []

        knn = self._bf.knnMatch(self._ref_desc, desc, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._ratio_test * n.distance:
                good.append(m)

        if len(good) < self._min_matches:
            return []

        src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self._ransac_reproj)
        if H is None or mask is None:
            return []

        inliers = int(mask.sum())
        if inliers < self._min_matches:
            return []

        pts = cv2.perspectiveTransform(self._ref_corners, H)
        pts = pts.reshape(-1, 2).astype(np.float32)

        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        area = float(cv2.contourArea(pts))

        return [
            DetectedTarget(
                kind="object",
                center=(cx, cy),
                area=area,
                points=pts,
                label="object",
                confidence=inliers / len(good),
            )
        ]


def create_detector(
    mode: DetectionMode,
    *,
    rect_min_area_ratio: float = 0.005,
    rect_max_area_ratio: float = 0.5,
    rect_angle_tol: float = 25.0,
    circle_min_area_ratio: float = 0.005,
    circle_max_area_ratio: float = 0.5,
    circle_min_circularity: float = 0.7,
    circle_min_radius: int = 10,
    circle_max_radius: int = 300,
    object_ref_path: Optional[str] = None,
    object_min_matches: int = 10,
) -> BaseDetector:
    """Factory to create the requested detector."""
    if mode == DetectionMode.RECT:
        return RectDetector(
            min_area_ratio=rect_min_area_ratio,
            max_area_ratio=rect_max_area_ratio,
            angle_tol=rect_angle_tol,
        )
    if mode == DetectionMode.CIRCLE:
        return CircleDetector(
            min_area_ratio=circle_min_area_ratio,
            max_area_ratio=circle_max_area_ratio,
            min_circularity=circle_min_circularity,
            min_radius=circle_min_radius,
            max_radius=circle_max_radius,
        )
    if mode == DetectionMode.OBJECT:
        if not object_ref_path:
            raise ValueError("--object-ref 必须指定一个参考图像路径")
        return ObjectDetector(
            reference_path=object_ref_path,
            min_matches=object_min_matches,
        )
    raise ValueError(f"Unknown detection mode: {mode}")
