import argparse
import os

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _stable_rng(image_name: str, mode: str) -> np.random.Generator:
    seed = 0
    for idx, ch in enumerate(f"{mode}:{image_name}"):
        seed = (seed * 131 + (idx + 17) * ord(ch)) % (2**32)
    return np.random.default_rng(seed)


def _build_mask_polygon(box: np.ndarray, landmarks: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    face_width = max(1.0, x2 - x1)
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float((left_mouth[1] + right_mouth[1]) / 2.0)
    top_y = int(max(y1, nose[1] - 0.12 * face_width))
    bottom_y = int(min(y2, mouth_y + 0.30 * face_width))

    left_top_x = int(max(0, x1 + 0.08 * face_width))
    right_top_x = int(min(w - 1, x2 - 0.08 * face_width))
    left_bottom_x = int(max(0, left_mouth[0] - 0.18 * face_width))
    right_bottom_x = int(min(w - 1, right_mouth[0] + 0.18 * face_width))

    return np.array(
        [
            [left_top_x, top_y],
            [right_top_x, top_y],
            [right_bottom_x, bottom_y],
            [left_bottom_x, bottom_y],
        ],
        dtype=np.int32,
    )


def _apply_mask_overlay(img: np.ndarray, polygon: np.ndarray, opacity: float = 0.82) -> np.ndarray:
    overlay = img.copy()
    # Soft blue mask with a bit of opacity so the result looks like an occlusion instead of a flat block.
    mask_color = (210, 220, 230)
    cv2.fillConvexPoly(overlay, polygon, mask_color)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    blended = cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0)

    top_left = tuple(polygon[0])
    top_right = tuple(polygon[1])
    bottom_right = tuple(polygon[2])
    bottom_left = tuple(polygon[3])
    cv2.line(blended, top_left, top_right, (180, 190, 200), 2, cv2.LINE_AA)
    cv2.line(blended, bottom_left, bottom_right, (165, 175, 185), 2, cv2.LINE_AA)

    # Draw straps to reinforce the synthetic mask appearance.
    strap_color = (185, 185, 185)
    strap_left_start = (int(top_left[0]), int((top_left[1] + bottom_left[1]) / 2))
    strap_left_end = (max(0, strap_left_start[0] - 12), max(0, strap_left_start[1] - 6))
    strap_right_start = (int(top_right[0]), int((top_right[1] + bottom_right[1]) / 2))
    strap_right_end = (min(img.shape[1] - 1, strap_right_start[0] + 12), max(0, strap_right_start[1] - 6))
    cv2.line(blended, strap_left_start, strap_left_end, strap_color, 2, cv2.LINE_AA)
    cv2.line(blended, strap_right_start, strap_right_end, strap_color, 2, cv2.LINE_AA)

    return blended


def _build_sunglasses_polygon(box: np.ndarray, landmarks: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    left_eye, right_eye, nose, _, _ = landmarks
    face_width = max(1.0, x2 - x1)
    top_y = int(max(y1, min(left_eye[1], right_eye[1]) - 0.16 * face_width))
    bottom_y = int(min(y2, nose[1] + 0.03 * face_width))
    left_x = int(max(0, x1 - 0.02 * face_width))
    right_x = int(min(w - 1, x2 + 0.02 * face_width))

    return np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y],
        ],
        dtype=np.int32,
    )


def _build_forehead_band_polygon(box: np.ndarray, landmarks: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    left_eye, right_eye, _, _, _ = landmarks
    face_width = max(1.0, x2 - x1)
    top_y = int(max(y1, min(left_eye[1], right_eye[1]) - 0.34 * face_width))
    bottom_y = int(max(top_y + 2, min(h - 1, min(left_eye[1], right_eye[1]) - 0.10 * face_width)))
    left_x = int(max(0, x1 - 0.05 * face_width))
    right_x = int(min(w - 1, x2 + 0.05 * face_width))

    return np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y],
        ],
        dtype=np.int32,
    )


def _apply_heavy_occlusion(img: np.ndarray, box: np.ndarray, landmarks: np.ndarray, opacity: float = 0.82) -> np.ndarray:
    occluded = img.copy()

    lower_face = _build_mask_polygon(box, landmarks, img.shape)
    occluded = _apply_mask_overlay(occluded, lower_face, opacity=opacity)

    sunglasses = _build_sunglasses_polygon(box, landmarks, img.shape)
    glasses_overlay = occluded.copy()
    cv2.fillConvexPoly(glasses_overlay, sunglasses, (20, 20, 20))
    occluded = cv2.addWeighted(glasses_overlay, 0.9, occluded, 0.1, 0)
    cv2.polylines(occluded, [sunglasses], isClosed=True, color=(55, 55, 55), thickness=2, lineType=cv2.LINE_AA)

    left_edge = tuple(sunglasses[0])
    right_edge = tuple(sunglasses[1])
    bridge_y = int((sunglasses[0][1] + sunglasses[2][1]) / 2)
    cv2.line(occluded, (left_edge[0], bridge_y), (right_edge[0], bridge_y), (70, 70, 70), 2, cv2.LINE_AA)

    forehead_band = _build_forehead_band_polygon(box, landmarks, img.shape)
    band_overlay = occluded.copy()
    cv2.fillConvexPoly(band_overlay, forehead_band, (35, 55, 90))
    occluded = cv2.addWeighted(band_overlay, 0.88, occluded, 0.12, 0)

    return occluded


def _estimate_skin_tone(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    patch_x1 = int(x1 + 0.20 * max(1, x2 - x1))
    patch_x2 = int(x1 + 0.80 * max(1, x2 - x1))
    patch_y1 = int(y1 + 0.22 * max(1, y2 - y1))
    patch_y2 = int(y1 + 0.70 * max(1, y2 - y1))
    patch = img[max(0, patch_y1):min(h, patch_y2), max(0, patch_x1):min(w, patch_x2)]
    if patch.size == 0:
        return np.array([155.0, 175.0, 205.0], dtype=np.float32)

    tone = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
    tone += np.array([8.0, 12.0, 18.0], dtype=np.float32)
    return np.clip(tone, 60.0, 240.0)


def _apply_polygon_overlay(
    img: np.ndarray,
    polygon: np.ndarray,
    color: np.ndarray,
    opacity: float,
    *,
    outline: tuple[int, int, int] | None = None,
    blur_kernel: int = 9,
) -> np.ndarray:
    overlay = img.copy()
    fill_color = tuple(int(channel) for channel in color)
    cv2.fillConvexPoly(overlay, polygon, fill_color)
    blended = cv2.addWeighted(overlay, float(np.clip(opacity, 0.0, 1.0)), img, 1.0 - float(np.clip(opacity, 0.0, 1.0)), 0)

    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (blur_kernel, blur_kernel), 0)[..., None]
        blended = img.astype(np.float32) * (1.0 - soft_mask) + blended.astype(np.float32) * soft_mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)

    if outline is not None:
        cv2.polylines(blended, [polygon], isClosed=True, color=outline, thickness=2, lineType=cv2.LINE_AA)
    return blended


def _build_opaque_panel_polygon(
    box: np.ndarray, landmarks: np.ndarray, image_shape: tuple[int, int, int], rng: np.random.Generator
) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    face_width = max(1.0, x2 - x1)
    left_x = int(max(0, left_eye[0] - rng.uniform(0.12, 0.22) * face_width))
    right_x = int(min(w - 1, right_eye[0] + rng.uniform(0.12, 0.22) * face_width))
    top_y = int(max(y1, min(left_eye[1], right_eye[1]) - rng.uniform(0.03, 0.10) * face_width))
    bottom_y = int(min(y2, max(left_mouth[1], right_mouth[1]) + rng.uniform(0.14, 0.28) * face_width))

    return np.array(
        [
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y],
        ],
        dtype=np.int32,
    )


def _apply_opaque_panel_occlusion(
    img: np.ndarray, box: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator, opacity: float = 0.84
) -> np.ndarray:
    polygon = _build_opaque_panel_polygon(box, landmarks, img.shape, rng)
    panel_color = np.full(3, rng.uniform(178, 240), dtype=np.float32)
    return _apply_polygon_overlay(img, polygon, panel_color, opacity, outline=(176, 176, 176), blur_kernel=9)


def _apply_hand_occlusion(
    img: np.ndarray, box: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator, opacity: float = 0.82
) -> np.ndarray:
    occluded = img.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    _, _, nose, left_mouth, right_mouth = landmarks
    face_width = max(1.0, x2 - x1)
    center = (
        int(np.clip(nose[0] + rng.uniform(-0.10, 0.10) * face_width, 0, w - 1)),
        int(np.clip((nose[1] + left_mouth[1] + right_mouth[1]) / 3.0 + rng.uniform(0.02, 0.12) * face_width, 0, h - 1)),
    )
    axes = (
        int(max(8, face_width * rng.uniform(0.22, 0.34))),
        int(max(8, face_width * rng.uniform(0.16, 0.26))),
    )
    angle = float(rng.uniform(-30, 30))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
    for offset in (-0.20, -0.06, 0.07, 0.20):
        finger_center = (
            int(np.clip(center[0] + offset * face_width, 0, w - 1)),
            int(np.clip(center[1] - axes[1] * rng.uniform(0.48, 0.80), 0, h - 1)),
        )
        finger_axes = (
            int(max(4, axes[0] * rng.uniform(0.16, 0.24))),
            int(max(5, axes[1] * rng.uniform(0.34, 0.52))),
        )
        cv2.ellipse(mask, finger_center, finger_axes, angle + rng.uniform(-10, 10), 0, 360, 255, thickness=-1)

    overlay = occluded.copy()
    hand_color = tuple(int(channel) for channel in _estimate_skin_tone(img, box))
    overlay[mask > 0] = hand_color
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (11, 11), 0)[..., None]
    alpha = np.clip(soft_mask * float(np.clip(opacity, 0.0, 1.0)), 0.0, 1.0)
    blended = img.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_scarf_occlusion(
    img: np.ndarray, box: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator, opacity: float = 0.78
) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    _, _, nose, left_mouth, right_mouth = landmarks
    face_width = max(1.0, x2 - x1)
    top_y = int(max(y1, nose[1] + rng.uniform(0.04, 0.14) * face_width))
    bottom_y = int(min(y2, max(left_mouth[1], right_mouth[1]) + rng.uniform(0.28, 0.42) * face_width))

    xs = np.linspace(x1, x2, 7).astype(np.int32)
    top_curve = []
    bottom_curve = []
    for idx, x in enumerate(xs):
        wave = np.sin(idx / max(1, len(xs) - 1) * np.pi)
        top_curve.append([int(x), int(top_y + wave * rng.uniform(-0.06, 0.08) * face_width)])
        bottom_curve.append([int(x), int(bottom_y + wave * rng.uniform(-0.04, 0.06) * face_width)])
    polygon = np.array(top_curve + bottom_curve[::-1], dtype=np.int32)
    scarf_color = np.full(3, rng.uniform(44, 206), dtype=np.float32)
    return _apply_polygon_overlay(img, polygon, scarf_color, opacity, blur_kernel=11)


def _apply_partial_occlusion(
    img: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray,
    image_name: str = "",
    opacity: float = 0.78,
) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    face_width = max(1.0, x2 - x1)
    cover_left = (sum(ord(ch) for ch in image_name) % 2) == 0

    if cover_left:
        side_x1 = int(max(0, x1 + 0.02 * face_width))
        side_x2 = int(min(w - 1, nose[0] + 0.06 * face_width))
        free_eye = right_eye
    else:
        side_x1 = int(max(0, nose[0] - 0.06 * face_width))
        side_x2 = int(min(w - 1, x2 - 0.02 * face_width))
        free_eye = left_eye

    # This covers roughly half the lower/side face while keeping at least one eye and central identity cues visible.
    top_y = int(max(y1, free_eye[1] + 0.07 * face_width))
    bottom_y = int(min(y2, max(left_mouth[1], right_mouth[1]) + 0.28 * face_width))
    polygon = np.array(
        [
            [side_x1, top_y],
            [side_x2, top_y],
            [side_x2, bottom_y],
            [side_x1, bottom_y],
        ],
        dtype=np.int32,
    )

    overlay = img.copy()
    cv2.fillConvexPoly(overlay, polygon, (210, 220, 230))
    opacity = float(np.clip(opacity, 0.0, 1.0))
    occluded = cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0)
    cv2.polylines(occluded, [polygon], isClosed=True, color=(165, 175, 185), thickness=2, lineType=cv2.LINE_AA)

    # Add one soft diagonal edge to simulate cloth without erasing the full facial pattern.
    cv2.line(occluded, tuple(polygon[0]), tuple(polygon[2]), (185, 195, 205), 1, cv2.LINE_AA)
    return occluded


def _apply_mixed_occlusion(
    img: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray,
    image_name: str,
    opacity: float = 0.82,
) -> np.ndarray:
    rng = _stable_rng(image_name, "mixed")
    primitives = [
        "standard",
        "partial",
        "heavy",
        "opaque",
        "hand",
        "scarf",
        "sunglasses",
    ]
    count = 1 if rng.random() < 0.68 else 2
    selected = list(rng.choice(primitives, size=count, replace=False))

    occluded = img.copy()
    for primitive in selected:
        if primitive == "standard":
            polygon = _build_mask_polygon(box, landmarks, img.shape)
            occluded = _apply_mask_overlay(occluded, polygon, opacity=opacity)
        elif primitive == "partial":
            occluded = _apply_partial_occlusion(occluded, box, landmarks, image_name, opacity=max(0.68, opacity - 0.04))
        elif primitive == "heavy":
            occluded = _apply_heavy_occlusion(occluded, box, landmarks, opacity=min(0.92, opacity + 0.04))
        elif primitive == "opaque":
            occluded = _apply_opaque_panel_occlusion(occluded, box, landmarks, rng, opacity=min(0.94, opacity + 0.03))
        elif primitive == "hand":
            occluded = _apply_hand_occlusion(occluded, box, landmarks, rng, opacity=opacity)
        elif primitive == "scarf":
            occluded = _apply_scarf_occlusion(occluded, box, landmarks, rng, opacity=max(0.64, opacity - 0.06))
        elif primitive == "sunglasses":
            sunglasses = _build_sunglasses_polygon(box, landmarks, img.shape)
            glasses_overlay = occluded.copy()
            cv2.fillConvexPoly(glasses_overlay, sunglasses, (20, 20, 20))
            occluded = cv2.addWeighted(glasses_overlay, 0.9, occluded, 0.1, 0)
            cv2.polylines(occluded, [sunglasses], isClosed=True, color=(55, 55, 55), thickness=2, lineType=cv2.LINE_AA)
    return occluded


def apply_face_mask(input_dir: str, output_dir: str, mode: str = "standard", opacity: float = 0.82):
    """
    Applies a synthetic surgical mask to a face database.
    Expects dataset structure: input_dir/person_name/image.jpg
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)

    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    masked_images = 0
    copied_without_mask = 0

    for person_name in sorted(os.listdir(input_dir)):
        person_indir = os.path.join(input_dir, person_name)

        if not os.path.isdir(person_indir):
            continue

        person_outdir = os.path.join(output_dir, person_name)
        os.makedirs(person_outdir, exist_ok=True)

        for img_name in sorted(os.listdir(person_indir)):
            img_path = os.path.join(person_indir, img_name)
            _, ext = os.path.splitext(img_name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            img = cv2.imread(img_path)

            if img is None:
                copied_without_mask += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

            if boxes is not None and landmarks is not None and len(boxes) > 0 and len(landmarks) > 0:
                if mode == "heavy":
                    img = _apply_heavy_occlusion(img, boxes[0], landmarks[0], opacity=opacity)
                elif mode == "partial":
                    img = _apply_partial_occlusion(img, boxes[0], landmarks[0], img_name, opacity=opacity)
                elif mode == "opaque":
                    img = _apply_opaque_panel_occlusion(img, boxes[0], landmarks[0], _stable_rng(img_name, mode), opacity=opacity)
                elif mode == "hand":
                    img = _apply_hand_occlusion(img, boxes[0], landmarks[0], _stable_rng(img_name, mode), opacity=opacity)
                elif mode == "scarf":
                    img = _apply_scarf_occlusion(img, boxes[0], landmarks[0], _stable_rng(img_name, mode), opacity=opacity)
                elif mode == "mixed":
                    img = _apply_mixed_occlusion(img, boxes[0], landmarks[0], img_name, opacity=opacity)
                else:
                    polygon = _build_mask_polygon(boxes[0], landmarks[0], img.shape)
                    img = _apply_mask_overlay(img, polygon, opacity=opacity)
                masked_images += 1
            else:
                copied_without_mask += 1

            out_path = os.path.join(person_outdir, img_name)
            cv2.imwrite(out_path, img)

    print(
        f"Finished processing {total_images} images. "
        f"Masked: {masked_images}. Copied without mask: {copied_without_mask}."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch apply masks to dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset folder (Ex: /data/original)")
    parser.add_argument("--output", type=str, required=True, help="Output dataset folder (Ex: /data/masked)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "heavy", "partial", "opaque", "hand", "scarf", "mixed"],
        default="standard",
        help="Occlusion style",
    )
    parser.add_argument("--opacity", type=float, default=0.82, help="Occlusion opacity from 0.0 to 1.0")
    args = parser.parse_args()
    
    apply_face_mask(args.input, args.output, args.mode, args.opacity)
