from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class QualityConfig:
    min_width: int = 80
    min_height: int = 80
    blur_threshold: float = 45.0
    dark_threshold: float = 35.0
    overexposed_threshold: float = 225.0
    max_dark_fraction: float = 0.70
    max_bright_fraction: float = 0.35
    yaw_threshold: float = 45.0


@dataclass
class QualityDecision:
    keep: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def laplacian_variance(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_metrics(image_rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "mean_luminance": float(gray.mean()),
        "dark_fraction": float((gray < 30).mean()),
        "bright_fraction": float((gray > 245).mean()),
    }


def estimate_yaw_degrees(landmarks: np.ndarray) -> float:
    """Estimate yaw from five-point MTCNN landmarks.

    This is a lightweight geometry heuristic, not a full 3D pose estimator.
    It is stable enough for filtering extreme side profiles before training.
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks.astype(np.float32)
    eye_center = (left_eye + right_eye) * 0.5
    mouth_center = (left_mouth + right_mouth) * 0.5
    face_center = (eye_center + mouth_center) * 0.5
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance <= 1e-6:
        return 90.0

    nose_offset = float((nose[0] - face_center[0]) / eye_distance)
    left_eye_nose = float(abs(nose[0] - left_eye[0]))
    right_eye_nose = float(abs(right_eye[0] - nose[0]))
    asymmetry = (left_eye_nose - right_eye_nose) / max(left_eye_nose + right_eye_nose, 1e-6)

    yaw_from_offset = np.degrees(np.arctan(nose_offset * 2.2))
    yaw_from_asymmetry = np.degrees(np.arctan(asymmetry * 1.8))
    return float(abs(0.55 * yaw_from_offset + 0.45 * yaw_from_asymmetry))


def check_basic_quality(image_rgb: np.ndarray, config: QualityConfig) -> QualityDecision:
    reasons: list[str] = []
    h, w = image_rgb.shape[:2]
    metrics = {"width": float(w), "height": float(h)}

    if w < config.min_width or h < config.min_height:
        reasons.append("low_resolution")

    blur = laplacian_variance(image_rgb)
    metrics["laplacian_variance"] = blur
    if blur < config.blur_threshold:
        reasons.append("blurry")

    metrics.update(exposure_metrics(image_rgb))
    if metrics["mean_luminance"] < config.dark_threshold or metrics["dark_fraction"] > config.max_dark_fraction:
        reasons.append("too_dark")
    if metrics["mean_luminance"] > config.overexposed_threshold or metrics["bright_fraction"] > config.max_bright_fraction:
        reasons.append("overexposed")

    return QualityDecision(keep=not reasons, reasons=reasons, metrics=metrics)


def check_pose_quality(landmarks: np.ndarray, config: QualityConfig) -> QualityDecision:
    yaw = estimate_yaw_degrees(landmarks)
    reasons = ["extreme_yaw"] if yaw > config.yaw_threshold else []
    return QualityDecision(keep=not reasons, reasons=reasons, metrics={"estimated_yaw": yaw})

