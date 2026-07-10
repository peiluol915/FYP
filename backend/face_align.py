from __future__ import annotations

import cv2
import numpy as np


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([points.astype(np.float32), ones])
    return homogeneous @ matrix.T


def align_face(
    image_rgb: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 128,
    margin: float = 0.28,
) -> np.ndarray:
    """Rotate eyes to horizontal, crop the aligned face, and resize.

    The crop is based on the detected face box and five-point landmarks after
    rotation. A small margin keeps useful identity context such as chin, hairline,
    and cheeks while removing most background.
    """
    h, w = image_rgb.shape[:2]
    points = landmarks.astype(np.float32)
    left_eye, right_eye = points[0], points[1]
    eye_center = (left_eye + right_eye) * 0.5
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = np.degrees(np.arctan2(dy, dx))

    rotation = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
    rotated = cv2.warpAffine(
        image_rgb,
        rotation,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    aligned_landmarks = _transform_points(points, rotation)
    x1, y1, x2, y2 = box.astype(np.float32)
    box_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    aligned_box = _transform_points(box_points, rotation)

    all_points = np.vstack([aligned_landmarks, aligned_box])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    crop_w = max_x - min_x
    crop_h = max_y - min_y
    side = max(crop_w, crop_h) * (1.0 + margin)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    crop_x1 = int(round(center_x - side * 0.5))
    crop_y1 = int(round(center_y - side * 0.5))
    crop_x2 = int(round(center_x + side * 0.5))
    crop_y2 = int(round(center_y + side * 0.5))

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        rotated = cv2.copyMakeBorder(
            rotated,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        crop_x1 += pad_left
        crop_x2 += pad_left
        crop_y1 += pad_top
        crop_y2 += pad_top

    crop = rotated[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise ValueError("Aligned crop is empty.")
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def normalized_augmentation_regions(output_size: int = 128) -> dict[str, list[int]]:
    s = output_size
    return {
        "mask_region": [int(0.20 * s), int(0.48 * s), int(0.80 * s), int(0.88 * s)],
        "sunglasses_region": [int(0.16 * s), int(0.24 * s), int(0.84 * s), int(0.46 * s)],
        "hand_object_region": [int(0.18 * s), int(0.34 * s), int(0.82 * s), int(0.82 * s)],
        "random_block_region": [int(0.12 * s), int(0.18 * s), int(0.88 * s), int(0.88 * s)],
    }

