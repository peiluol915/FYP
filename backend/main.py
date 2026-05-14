import base64
import time
import warnings
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

try:
    from models import DEGAN, OAN
except ModuleNotFoundError:
    from backend.models import DEGAN, OAN


warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GALLERY_CACHE_VERSION = 6
BASE_DIR = Path(__file__).resolve().parent.parent
weights_dir = BASE_DIR / "weights"
gallery_dir = BASE_DIR / "database" / "lfw-deepfunneled"
gallery_cache_path = weights_dir / "gallery_index.pt"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=112, margin=0, post_process=False, device=device)


def _load_model_weights(model: torch.nn.Module, weight_name: str) -> bool:
    weight_path = weights_dir / weight_name
    if not weight_path.exists():
        return False

    state_dict = torch.load(weight_path, map_location=device)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible_state:
        return False
    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=False)
    return True


oan_model = OAN().to(device)
degan_model = DEGAN().to(device)
recognition_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
oan_loaded = _load_model_weights(oan_model, "oan_model_final.pth")
degan_loaded = _load_model_weights(degan_model, "degan_model_final.pth")
oan_model.eval()
degan_model.eval()

gallery_embeddings: torch.Tensor | None = None
gallery_names: list[str] = []
gallery_image_paths: list[str] = []


def _decode_upload(img_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image.")
    return img_bgr


def _rgb_image_to_face_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _normalize_mtcnn_face_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    tensor = face_tensor.float()
    # MTCNN with post_process=False always returns [0, 255] tensors.
    # Normalize to [-1, 1] unconditionally to avoid silent corruption
    # when a face happens to have pixel values all below 1.0.
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.min() >= 0.0:
        tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _prepare_face_tensor(img_bgr: np.ndarray) -> tuple[torch.Tensor, bool]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    detected_face = face_tensor is not None

    if face_tensor is None:
        resized = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_AREA)
        face_tensor = _rgb_image_to_face_tensor(resized).squeeze(0)
    else:
        face_tensor = _normalize_mtcnn_face_tensor(face_tensor).squeeze(0)

    return face_tensor.unsqueeze(0).to(device), detected_face


def _landmarks_to_face_coords(box: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    mapped = landmarks.astype(np.float32).copy()
    mapped[:, 0] = ((mapped[:, 0] - x1) / width) * (size - 1)
    mapped[:, 1] = ((mapped[:, 1] - y1) / height) * (size - 1)
    return np.clip(mapped, 0, size - 1)


def _scale_landmarks_to_resized_image(
    landmarks: np.ndarray, original_shape: tuple[int, int], size: int = 112
) -> np.ndarray:
    original_h, original_w = original_shape
    scaled = landmarks.astype(np.float32).copy()
    scale_x = (size - 1) / max(1.0, float(original_w - 1))
    scale_y = (size - 1) / max(1.0, float(original_h - 1))
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return np.clip(scaled, 0, size - 1)


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    return mask


def _occlusion_mask_to_tensor(
    occlusion_mask: np.ndarray, *, device_target: torch.device, dtype: torch.dtype, batch_size: int
) -> torch.Tensor:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(mask).to(device=device_target, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return tensor.expand(batch_size, -1, -1, -1)


def _estimate_bright_panel_mask(face_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    grayscale = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    texture_energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(grayscale, cv2.CV_32F)), (5, 5), 0)

    h, w = face_rgb.shape[:2]
    focus_mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, int(w * 0.22))
    x2 = min(w, int(w * 0.78))
    y1 = max(0, int(h * 0.16))
    y2 = min(h, int(h * 0.86))
    focus_mask[y1:y2, x1:x2] = 255

    candidate = (
        (hsv[:, :, 1] < 70)
        & (hsv[:, :, 2] > 145)
        & (texture_energy < 10.0)
        & (focus_mask > 0)
    )
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    panel_mask = np.zeros_like(candidate_u8)
    contours, _ = cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return panel_mask.astype(np.float32)

    image_center_x = w / 2.0
    image_center_y = h / 2.0
    best_contour = None
    best_score = float("-inf")
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 30:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / max(1.0, float(bh))
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        center_offset_x = abs(center_x - image_center_x) / max(1.0, image_center_x)
        center_offset_y = abs(center_y - image_center_y) / max(1.0, image_center_y)
        aspect_penalty = abs(np.log(max(aspect_ratio, 1e-6)))
        size_penalty = 0.0
        if bw > w * 0.42 or bh > h * 0.42:
            size_penalty += 1.2
        if aspect_ratio < 0.45 or aspect_ratio > 1.6:
            size_penalty += 1.6
        score = (
            area
            - 220.0 * center_offset_x
            - 90.0 * center_offset_y
            - 120.0 * aspect_penalty
            - 180.0 * size_penalty
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return panel_mask.astype(np.float32)

    x, y, bw, bh = cv2.boundingRect(best_contour)
    pad_x = max(2, int(0.02 * w))
    pad_y = max(2, int(0.02 * h))
    panel_mask[max(0, y - pad_y):min(h, y + bh + pad_y), max(0, x - pad_x):min(w, x + bw + pad_x)] = 255
    return cv2.GaussianBlur(panel_mask.astype(np.float32) / 255.0, (9, 9), 0)


def _estimate_skin_occluder_mask(
    face_rgb: np.ndarray,
    focus_region: np.ndarray,
    center_region: np.ndarray,
    eye_region: np.ndarray,
    lower_region: np.ndarray,
    texture_energy: np.ndarray,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    focus_pixels = texture_energy[focus_region > 0]
    adaptive_texture_threshold = 7.0
    if focus_pixels.size > 0:
        adaptive_texture_threshold = float(np.clip(np.percentile(focus_pixels, 32), 4.8, 8.6))

    candidate = (
        (ycrcb[:, :, 1] >= 132)
        & (ycrcb[:, :, 1] <= 182)
        & (ycrcb[:, :, 2] >= 82)
        & (ycrcb[:, :, 2] <= 138)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 1] <= 165)
        & (texture_energy <= adaptive_texture_threshold)
        & (focus_region > 0)
    )
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    h, w = face_rgb.shape[:2]
    best_contour = None
    best_score = float("-inf")
    for contour in cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area < 45:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > w * 0.92 and bh > h * 0.72:
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        center_overlap = float((component_mask[center_region > 0] > 0).mean()) if np.any(center_region > 0) else 0.0
        eye_overlap = float((component_mask[eye_region > 0] > 0).mean()) if np.any(eye_region > 0) else 0.0
        lower_overlap = float((component_mask[lower_region > 0] > 0).mean()) if np.any(lower_region > 0) else 0.0
        if center_overlap < 0.10 and max(eye_overlap, lower_overlap) < 0.12:
            continue

        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        score = (
            area
            + 180.0 * center_overlap
            + 120.0 * max(eye_overlap, lower_overlap)
            - 100.0 * abs(center_x - (w / 2.0)) / max(1.0, w / 2.0)
            - 80.0 * abs(center_y - (h / 2.0)) / max(1.0, h / 2.0)
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return np.zeros((h, w), dtype=np.float32)

    output = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(output, [best_contour], -1, 255, thickness=-1)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(output.astype(np.float32) / 255.0, (9, 9), 0)


def _estimate_occlusion(face_rgb: np.ndarray, face_landmarks: np.ndarray | None) -> tuple[np.ndarray, bool, str, float]:
    bright_panel_mask = _estimate_bright_panel_mask(face_rgb)
    if face_landmarks is None:
        coverage = float(bright_panel_mask.mean())
        region = "mixed" if coverage > 0.015 else "none"
        return bright_panel_mask, coverage > 0.015, region, coverage

    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    left_eye, right_eye, nose, left_mouth, right_mouth = face_landmarks
    face_width = max(1.0, right_mouth[0] - left_eye[0])

    lower_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.10 * face_width)), max(0, int(nose[1] - 0.02 * face_width))],
            [min(111, int(right_eye[0] + 0.10 * face_width)), max(0, int(nose[1] - 0.02 * face_width))],
            [min(111, int(right_mouth[0] + 0.18 * face_width)), min(111, int(right_mouth[1] + 0.32 * face_width))],
            [max(0, int(left_mouth[0] - 0.18 * face_width)), min(111, int(left_mouth[1] + 0.32 * face_width))],
        ],
        dtype=np.int32,
    )
    eye_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.20 * face_width))],
            [min(111, int(right_eye[0] + 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.20 * face_width))],
            [min(111, int(right_eye[0] + 0.18 * face_width)), min(111, int(nose[1] + 0.02 * face_width))],
            [max(0, int(left_eye[0] - 0.18 * face_width)), min(111, int(nose[1] + 0.02 * face_width))],
        ],
        dtype=np.int32,
    )
    center_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.16 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.05 * face_width))],
            [min(111, int(right_eye[0] + 0.16 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.05 * face_width))],
            [min(111, int(right_mouth[0] + 0.12 * face_width)), min(111, int(right_mouth[1] + 0.14 * face_width))],
            [max(0, int(left_mouth[0] - 0.12 * face_width)), min(111, int(left_mouth[1] + 0.14 * face_width))],
        ],
        dtype=np.int32,
    )
    focus_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.02 * face_width))],
            [min(111, int(right_eye[0] + 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.02 * face_width))],
            [min(111, int(right_mouth[0] + 0.16 * face_width)), min(111, int(right_mouth[1] + 0.18 * face_width))],
            [max(0, int(left_mouth[0] - 0.16 * face_width)), min(111, int(left_mouth[1] + 0.18 * face_width))],
        ],
        dtype=np.int32,
    )

    lower_region = _polygon_mask(face_rgb.shape[:2], lower_polygon)
    eye_region = _polygon_mask(face_rgb.shape[:2], eye_polygon)
    center_region = _polygon_mask(face_rgb.shape[:2], center_polygon)
    focus_region = _polygon_mask(face_rgb.shape[:2], focus_polygon)
    upper_reference_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.14 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.28 * face_width))],
            [min(111, int(right_eye[0] + 0.14 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.28 * face_width))],
            [min(111, int(right_eye[0] + 0.08 * face_width)), min(111, int(nose[1] - 0.04 * face_width))],
            [max(0, int(left_eye[0] - 0.08 * face_width)), min(111, int(nose[1] - 0.04 * face_width))],
        ],
        dtype=np.int32,
    )
    upper_reference_region = _polygon_mask(face_rgb.shape[:2], upper_reference_polygon)

    cloth_mask = (
        (hsv[:, :, 1] < 70)
        & (hsv[:, :, 2] > 130)
        & (lower_region > 0)
    )
    upper_values = hsv[:, :, 2][upper_reference_region > 0]
    adaptive_lower_dark_threshold = 75.0
    if upper_values.size > 0:
        adaptive_lower_dark_threshold = float(np.clip(np.percentile(upper_values, 60) - 35.0, 45.0, 120.0))
    lower_dark_mask = (
        (hsv[:, :, 2] < adaptive_lower_dark_threshold)
        & (hsv[:, :, 1] < 135)
        & (lower_region > 0)
    )
    grayscale = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    texture_energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(grayscale, cv2.CV_32F)), (5, 5), 0)
    skin_occluder_mask = _estimate_skin_occluder_mask(
        face_rgb,
        focus_region,
        center_region,
        eye_region,
        lower_region,
        texture_energy,
    )
    bright_neutral_mask = (
        (hsv[:, :, 1] < 60)
        & (hsv[:, :, 2] > 150)
        & (texture_energy < 8.0)
        & (focus_region > 0)
    )
    bright_neutral_mask_u8 = np.where(bright_neutral_mask, 255, 0).astype(np.uint8)
    bright_neutral_mask_u8 = cv2.morphologyEx(bright_neutral_mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    opaque_panel_mask = np.zeros_like(bright_neutral_mask_u8)
    contours, _ = cv2.findContours(bright_neutral_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 35:
            x, y, w, h = cv2.boundingRect(largest)
            pad_x = max(2, int(0.06 * face_width))
            pad_y = max(2, int(0.04 * face_width))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(face_rgb.shape[1], x + w + pad_x)
            y2 = min(face_rgb.shape[0], y + h + pad_y)
            opaque_panel_mask[y1:y2, x1:x2] = 255
    dark_mask = (
        (hsv[:, :, 2] < 75)
        & (eye_region > 0)
    )

    combined = np.where(
        cloth_mask
        | lower_dark_mask
        | (opaque_panel_mask > 0)
        | (bright_panel_mask > 0.15)
        | (skin_occluder_mask > 0.18)
        | dark_mask,
        255,
        0,
    ).astype(np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coverage = float(combined.mean() / 255.0)
    is_occluded = coverage > 0.012

    region = "none"
    lower_coverage = float((combined[lower_region > 0] > 0).mean()) if np.any(lower_region > 0) else 0.0
    eye_coverage = float((combined[eye_region > 0] > 0).mean()) if np.any(eye_region > 0) else 0.0
    center_coverage = float((combined[center_region > 0] > 0).mean()) if np.any(center_region > 0) else 0.0
    if is_occluded:
        if lower_coverage > 0.08 and eye_coverage > 0.08:
            region = "mixed"
        elif center_coverage > 0.14 and eye_coverage > 0.04:
            region = "mixed"
        elif lower_coverage >= eye_coverage:
            region = "lower-face"
        else:
            region = "eye-region"

    soft_mask = cv2.GaussianBlur(combined.astype(np.float32) / 255.0, (9, 9), 0)
    return soft_mask, is_occluded, region, coverage


def _prepare_face_analysis(img_bgr: np.ndarray) -> dict:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reconstruction_rgb = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    reconstruction_tensor = _rgb_image_to_face_tensor(reconstruction_rgb)
    boxes, _, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    face_tensor = mtcnn(img_rgb)
    detected_face = face_tensor is not None and boxes is not None and landmarks is not None and len(boxes) > 0

    if detected_face:
        face_tensor = _normalize_mtcnn_face_tensor(face_tensor)
        face_rgb = _tensor_to_rgb_image(face_tensor)
        face_landmarks = _landmarks_to_face_coords(boxes[0], landmarks[0])
        reconstruction_landmarks = _scale_landmarks_to_resized_image(landmarks[0], img_rgb.shape[:2])
    else:
        face_rgb = reconstruction_rgb
        face_tensor = reconstruction_tensor
        face_landmarks = None
        reconstruction_landmarks = None

    recognition_mask, recognition_is_occluded, _, _ = _estimate_occlusion(face_rgb, face_landmarks)
    occlusion_mask, is_occluded, occlusion_region, occlusion_ratio = _estimate_occlusion(
        reconstruction_rgb, reconstruction_landmarks
    )
    return {
        "face_tensor": face_tensor,
        "detected_face": detected_face,
        "face_rgb": face_rgb,
        "face_landmarks": face_landmarks,
        "occlusion_mask": occlusion_mask,
        "recognition_mask": recognition_mask,
        "recognition_is_occluded": recognition_is_occluded,
        "is_occluded": is_occluded,
        "occlusion_region": occlusion_region,
        "occlusion_ratio": occlusion_ratio,
        "reconstruction_tensor": reconstruction_tensor,
        "reconstruction_rgb": reconstruction_rgb,
    }


def _reconstruct_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray | None = None) -> tuple[torch.Tensor, np.ndarray]:
    with torch.no_grad():
        mask_tensor = None
        if occlusion_mask is not None:
            mask_tensor = _occlusion_mask_to_tensor(
                occlusion_mask, device_target=face_tensor.device, dtype=face_tensor.dtype, batch_size=face_tensor.shape[0]
            )
        reconstructed = degan_model(face_tensor, mask_tensor)

    reconstructed_rgb = _tensor_to_rgb_image(reconstructed)
    return reconstructed, reconstructed_rgb


def _blend_reconstructed_face(
    face_tensor: torch.Tensor,
    occlusion_mask: np.ndarray,
    *,
    enhance_patch: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    reconstructed, reconstructed_rgb = _reconstruct_face(face_tensor, occlusion_mask)

    original_rgb = _tensor_to_rgb_image(face_tensor).astype(np.float32)
    if enhance_patch:
        reconstructed_rgb = _enhance_reconstructed_patch(
            original_rgb.astype(np.uint8),
            reconstructed_rgb,
            occlusion_mask,
        )
    reconstructed_rgb = reconstructed_rgb.astype(np.float32)
    # Dilate the mask to ensure it fully covers the occluder boundary so
    # original white-box pixels cannot leak into the blended result.
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = mask[..., None]
    blended_rgb = original_rgb * (1.0 - mask) + reconstructed_rgb * mask
    blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)
    blended_tensor = _rgb_image_to_face_tensor(blended_rgb)
    return blended_tensor, blended_rgb


def _extract_region_only_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray, fill_value: float = 0.0) -> tuple[torch.Tensor, np.ndarray]:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask).to(device=device, dtype=face_tensor.dtype).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.expand(face_tensor.shape[0], face_tensor.shape[1], -1, -1)
    region_tensor = face_tensor * mask_tensor + fill_value * (1.0 - mask_tensor)
    region_rgb = _tensor_to_rgb_image(region_tensor)
    return region_tensor, region_rgb


def _reconstructed_region_only_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    reconstructed, _ = _reconstruct_face(face_tensor, occlusion_mask)
    return _extract_region_only_face(reconstructed, occlusion_mask)


def _load_image_tensor_from_path(image_path: Path) -> torch.Tensor | None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    tensor, _ = _prepare_face_tensor(img_bgr)
    return tensor


def _load_rgb_image_from_path(image_path: Path) -> np.ndarray | None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _enhance_reconstructed_patch(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(occlusion_mask.astype(np.float32), 0.0, 1.0)
    mask_u8 = (mask * 255).astype(np.uint8)
    if mask_u8.max() == 0:
        return reconstructed_rgb

    nonzero = np.argwhere(mask_u8 > 16)
    if nonzero.size == 0:
        return reconstructed_rgb

    y1, x1 = nonzero.min(axis=0)
    y2, x2 = nonzero.max(axis=0) + 1
    if (y2 - y1) < 8 or (x2 - x1) < 8:
        return reconstructed_rgb

    source_patch = reconstructed_rgb[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2]
    h, w = original_rgb.shape[:2]
    pad_y = max(8, int(0.14 * (y2 - y1)))
    pad_x = max(8, int(0.14 * (x2 - x1)))
    sample_y1 = max(0, y1 - pad_y)
    sample_y2 = min(h, y2 + pad_y)
    sample_x1 = max(0, x1 - pad_x)
    sample_x2 = min(w, x2 + pad_x)

    source_gray = cv2.cvtColor(source_patch, cv2.COLOR_RGB2GRAY)
    source_texture = float(cv2.Laplacian(source_gray, cv2.CV_32F).var())
    target_texture = np.clip(70.0 / max(source_texture, 12.0), 0.9, 1.65)

    denoised_bgr = cv2.bilateralFilter(cv2.cvtColor(source_patch, cv2.COLOR_RGB2BGR), 3, 14, 14)
    denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    upscaled = cv2.resize(denoised, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up_blur = cv2.GaussianBlur(upscaled, (0, 0), 0.7)
    up_sharp = cv2.addWeighted(upscaled, 1.22 * target_texture, up_blur, -0.22 * target_texture, 1.0)
    sharpened = cv2.resize(np.clip(up_sharp, 0, 255).astype(np.uint8), (source_patch.shape[1], source_patch.shape[0]), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(sharpened, cv2.CV_32F, ksize=3)
    sharpened = np.clip(sharpened.astype(np.float32) + 0.14 * target_texture * laplacian, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    l_channel = cv2.convertScaleAbs(l_channel, alpha=1.04, beta=2)
    sharpened = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)

    # Match the reconstructed patch color to nearby visible skin so the patch
    # returns closer to the person's original complexion.
    sample_region = original_rgb[sample_y1:sample_y2, sample_x1:sample_x2].copy()
    sample_mask = mask[sample_y1:sample_y2, sample_x1:sample_x2]
    inpaint_mask = (sample_mask > 0.12).astype(np.uint8) * 255
    donor_patch = None
    if inpaint_mask.max() > 0:
        donor_bgr = cv2.inpaint(
            cv2.cvtColor(sample_region, cv2.COLOR_RGB2BGR),
            inpaint_mask,
            3,
            cv2.INPAINT_TELEA,
        )
        donor_rgb = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2RGB)
        donor_patch = donor_rgb[y1 - sample_y1:y2 - sample_y1, x1 - sample_x1:x2 - sample_x1].copy()
    hsv = cv2.cvtColor(sample_region, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(sample_region, cv2.COLOR_RGB2YCrCb)
    skin_mask = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 1] <= 185)
        & (ycrcb[:, :, 1] >= 132)
        & (ycrcb[:, :, 1] <= 180)
        & (ycrcb[:, :, 2] >= 80)
        & (ycrcb[:, :, 2] <= 135)
        & (sample_mask < 0.08)
    )
    if skin_mask.sum() >= 40:
        source_lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(sample_region, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_pixels = target_lab[skin_mask]
        source_pixels = source_lab.reshape(-1, 3)

        target_mean = target_pixels.mean(axis=0)
        source_mean = source_pixels.mean(axis=0)
        target_std = target_pixels.std(axis=0) + 1e-6
        source_std = source_pixels.std(axis=0) + 1e-6

        remapped = source_lab.copy()
        target_l_p90 = float(np.percentile(target_pixels[:, 0], 90))
        target_l_p97 = float(np.percentile(target_pixels[:, 0], 97))
        source_l_mean = float(source_mean[0])
        extra_l_strength = 0.92 if source_l_mean > target_mean[0] + 6.0 else 0.72
        for channel_idx, strength in ((0, extra_l_strength), (1, 0.86), (2, 0.86)):
            normalized = (source_lab[:, :, channel_idx] - source_mean[channel_idx]) / source_std[channel_idx]
            matched = normalized * target_std[channel_idx] + target_mean[channel_idx]
            remapped[:, :, channel_idx] = (
                source_lab[:, :, channel_idx] * (1.0 - strength)
                + matched * strength
            )
        # Compress high luminance outliers so the patch stays within the local
        # complexion range instead of keeping a washed-out rectangular highlight.
        remapped[:, :, 0] = np.where(
            remapped[:, :, 0] > target_l_p90,
            target_l_p90 + (remapped[:, :, 0] - target_l_p90) * 0.28,
            remapped[:, :, 0],
        )
        remapped[:, :, 0] = np.clip(remapped[:, :, 0], 0.0, target_l_p97 + 4.0)
        sharpened = cv2.cvtColor(np.clip(remapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Run a second lightweight Y-channel match against the nearby visible
        # face region to reduce exposure jumps inside the reconstructed patch.
        sharpened_ycrcb = cv2.cvtColor(sharpened, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        target_ycrcb = cv2.cvtColor(sample_region, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        target_y = target_ycrcb[:, :, 0][skin_mask]
        patch_y = sharpened_ycrcb[:, :, 0]
        if target_y.size >= 40:
            target_y_mean = float(target_y.mean())
            target_y_std = float(target_y.std() + 1e-6)
            patch_y_mean = float(patch_y.mean())
            patch_y_std = float(patch_y.std() + 1e-6)
            matched_y = (patch_y - patch_y_mean) * (target_y_std / patch_y_std) + target_y_mean
            blend_strength = 0.72 if patch_y_mean > target_y_mean + 5.0 else 0.5
            sharpened_ycrcb[:, :, 0] = patch_y * (1.0 - blend_strength) + matched_y * blend_strength
            high_clip = float(np.percentile(target_y, 97) + 5.0)
            sharpened_ycrcb[:, :, 0] = np.clip(sharpened_ycrcb[:, :, 0], 0.0, high_clip)
            sharpened = cv2.cvtColor(np.clip(sharpened_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    visible_context = original_rgb[sample_y1:sample_y2, sample_x1:sample_x2].copy()
    context_mask = mask[sample_y1:sample_y2, sample_x1:sample_x2] < 0.08
    if context_mask.sum() >= 50:
        context_ycrcb = cv2.cvtColor(visible_context, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        context_y = context_ycrcb[:, :, 0][context_mask]
        context_hsv = cv2.cvtColor(visible_context, cv2.COLOR_RGB2HSV).astype(np.float32)
        context_s = context_hsv[:, :, 1][context_mask]
        patch_ycrcb = cv2.cvtColor(sharpened, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        patch_y = patch_ycrcb[:, :, 0]
        patch_hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV).astype(np.float32)
        patch_s = patch_hsv[:, :, 1]
        context_mean = float(context_y.mean())
        context_p95 = float(np.percentile(context_y, 95))
        patch_mean = float(patch_y.mean())
        context_sat_mean = float(context_s.mean()) if context_s.size > 0 else 0.0
        patch_sat_mean = float(patch_s.mean())
        highlight_fraction = float((patch_y > context_p95 + 6.0).mean())
        overbright = patch_mean > context_mean + 7.0 or highlight_fraction > 0.08
        washed_out = patch_sat_mean < max(18.0, context_sat_mean * 0.62)
        if overbright:
            matched_y = patch_y - (patch_mean - context_mean) * 0.88
            matched_y = np.where(
                matched_y > context_p95,
                context_p95 + (matched_y - context_p95) * 0.18,
                matched_y,
            )
            patch_ycrcb[:, :, 0] = np.clip(matched_y, 0.0, context_p95 + 4.0)
            sharpened = cv2.cvtColor(np.clip(patch_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        if washed_out:
            patch_hsv[:, :, 1] = np.clip(
                patch_s * 0.35 + context_sat_mean * 0.65,
                0.0,
                max(255.0, context_sat_mean + 20.0),
            )
            sharpened = cv2.cvtColor(np.clip(patch_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        if donor_patch is not None and (overbright or washed_out):
            donor_rgb = donor_patch.astype(np.float32)
            current_rgb = sharpened.astype(np.float32)
            donor_y = cv2.cvtColor(donor_patch, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
            donor_mean = float(donor_y.mean())
            severity = 0.35
            if patch_mean > context_mean + 11.0 or highlight_fraction > 0.16:
                severity = 0.55
            if washed_out and context_sat_mean > patch_sat_mean + 10.0:
                severity += 0.10
            severity = float(np.clip(severity, 0.25, 0.65))
            blend = (local_mask[..., None] ** 1.25) * severity
            sharpened = np.clip(current_rgb * (1.0 - blend) + donor_rgb * blend, 0, 255).astype(np.uint8)

    if context_mask.sum() >= 50:
        context_gray = cv2.cvtColor(visible_context, cv2.COLOR_RGB2GRAY)
        context_texture = float(cv2.Laplacian(context_gray, cv2.CV_32F).var())
        patch_gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        patch_texture = float(cv2.Laplacian(patch_gray, cv2.CV_32F).var())
        if patch_texture > 1e-6 and context_texture > patch_texture:
            extra_gain = np.clip(context_texture / patch_texture, 1.0, 1.35)
            blur_small = cv2.GaussianBlur(sharpened, (0, 0), 0.6)
            sharpened = cv2.addWeighted(sharpened, 1.08 * extra_gain, blur_small, -0.08 * extra_gain, 0.5)

    enhanced = original_rgb.copy().astype(np.float32)
    alpha = (local_mask[..., None] ** 1.35) * 0.88
    enhanced[y1:y2, x1:x2] = (
        enhanced[y1:y2, x1:x2] * (1.0 - alpha)
        + sharpened.astype(np.float32) * alpha
    )
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _encode_rgb_to_base64(img_rgb: np.ndarray) -> str:
    image = Image.fromarray(img_rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _embed_faces(face_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mask_tensor = torch.zeros((face_tensor.shape[0], 1, face_tensor.shape[2], face_tensor.shape[3]), device=device, dtype=face_tensor.dtype)
        reconstructed = degan_model(face_tensor, mask_tensor)
        embedding = oan_model(reconstructed)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def _recognizer_input_from_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    rgb_image = _tensor_to_rgb_image(face_tensor)
    resized = cv2.resize(rgb_image, (160, 160), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _recognition_embedding(face_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        recognizer_input = _recognizer_input_from_tensor(face_tensor)
        embedding = recognition_model(recognizer_input)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def _iter_gallery_reference_images() -> list[tuple[str, Path]]:
    references: list[tuple[str, Path]] = []
    if not gallery_dir.exists():
        return references

    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        image_paths = sorted(
            path for path in person_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            references.append((person_dir.name, image_path))
    return references


def _build_gallery_index() -> tuple[torch.Tensor | None, list[str], list[str]]:
    names: list[str] = []
    embeddings: list[torch.Tensor] = []
    image_paths: list[str] = []

    for person_name, image_path in _iter_gallery_reference_images():
        face_tensor = _load_image_tensor_from_path(image_path)
        if face_tensor is None:
            continue
        embedding = _recognition_embedding(face_tensor)
        names.append(person_name)
        embeddings.append(embedding.squeeze(0).cpu())
        image_paths.append(str(image_path))

    if not embeddings:
        return None, [], []

    return torch.stack(embeddings), names, image_paths


def refresh_gallery_index() -> int:
    global gallery_embeddings, gallery_names, gallery_image_paths
    if gallery_cache_path.exists():
        try:
            cache = torch.load(gallery_cache_path, map_location="cpu")
            cached_names = cache.get("names", [])
            cached_embeddings = cache.get("embeddings")
            cached_image_paths = cache.get("image_paths", [])
            cached_gallery_dir = cache.get("gallery_dir")
            if (
                cache.get("version") == GALLERY_CACHE_VERSION
                and cached_gallery_dir == str(gallery_dir)
                and isinstance(cached_names, list)
                and isinstance(cached_embeddings, torch.Tensor)
                and isinstance(cached_image_paths, list)
                and len(cached_names) == cached_embeddings.shape[0]
                and len(cached_image_paths) == len(cached_names)
            ):
                gallery_embeddings = cached_embeddings
                gallery_names = cached_names
                gallery_image_paths = cached_image_paths
                return len(gallery_names)
        except Exception:
            gallery_cache_path.unlink(missing_ok=True)

    gallery_embeddings, gallery_names, gallery_image_paths = _build_gallery_index()
    if gallery_embeddings is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": GALLERY_CACHE_VERSION,
                "gallery_dir": str(gallery_dir),
                "names": gallery_names,
                "embeddings": gallery_embeddings,
                "image_paths": gallery_image_paths,
            },
            gallery_cache_path,
        )
    return 0 if gallery_embeddings is None else len(gallery_names)


def _predict_identity(face_tensor: torch.Tensor) -> tuple[str | None, float, Path | None]:
    if gallery_embeddings is None or not gallery_names:
        refresh_gallery_index()
    if gallery_embeddings is None or not gallery_names:
        return None, 0.0, None

    probe_embedding = _recognition_embedding(face_tensor).squeeze(0).cpu()
    similarities = torch.matmul(gallery_embeddings, probe_embedding)
    best_index = int(torch.argmax(similarities).item())
    matched_image_path = None
    if best_index < len(gallery_image_paths):
        matched_image_path = Path(gallery_image_paths[best_index])
    return gallery_names[best_index], float(similarities[best_index].item()), matched_image_path


def _name_from_dataset_filename(filename: str | None) -> str | None:
    if not filename:
        return None

    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None

    candidate = parts[0]
    candidate_dir = gallery_dir / candidate
    if candidate_dir.is_dir():
        return candidate
    return None


def _original_image_from_match(matched_image_path: Path | None) -> np.ndarray | None:
    if matched_image_path is None or not matched_image_path.is_file():
        return None
    return _load_rgb_image_from_path(matched_image_path)


@app.get("/")
def read_root():
    return {
        "message": "Hybrid OAN API is running smoothly!",
        "device": str(device),
        "oan_weights_loaded": oan_loaded,
        "degan_weights_loaded": degan_loaded,
        "gallery_size": len(gallery_names),
    }


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    analysis = _prepare_face_analysis(img_bgr)
    if analysis["is_occluded"]:
        _, reconstructed_rgb = _blend_reconstructed_face(
            analysis["reconstruction_tensor"],
            analysis["occlusion_mask"],
            enhance_patch=True,
        )
    else:
        reconstructed_rgb = analysis["reconstruction_rgb"]
    original_rgb = analysis["reconstruction_rgb"]
    mse = float(((reconstructed_rgb.astype(np.float32) - original_rgb.astype(np.float32)) ** 2).mean())
    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": analysis["is_occluded"],
        "occlusion_region": analysis["occlusion_region"],
        "occlusion_ratio": round(float(analysis["occlusion_ratio"]), 4),
        "reconstruction_model_loaded": degan_loaded,
        "reconstruction_mse": round(mse, 6),
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "input_image_base64": _encode_rgb_to_base64(original_rgb),
        "reconstructed_image_base64": _encode_rgb_to_base64(reconstructed_rgb),
    }
    return response


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    analysis = _prepare_face_analysis(img_bgr)
    face_for_recognition = analysis["face_tensor"]
    recognition_region_rgb = analysis["face_rgb"]
    if analysis["recognition_is_occluded"]:
        face_for_recognition, recognition_region_rgb = _reconstructed_region_only_face(
            face_for_recognition, analysis["recognition_mask"]
        )

    predicted_name, similarity, matched_image_path = _predict_identity(face_for_recognition)
    gallery_original_rgb = _original_image_from_match(matched_image_path)
    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": analysis["is_occluded"],
        "occlusion_region": analysis["occlusion_region"],
        "occlusion_ratio": round(float(analysis["occlusion_ratio"]), 4),
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "match": predicted_name is not None,
        "confidence": round((similarity + 1.0) / 2.0, 4),
        "similarity": round(similarity, 4),
        "predicted_name": predicted_name,
        "branch": "Hybrid",
        "reconstruction_model_loaded": degan_loaded,
        "recognition_model_loaded": oan_loaded,
        "gallery_size": len(gallery_names),
        "recognition_mode": "reconstructed-region-only" if analysis["recognition_is_occluded"] else "full-face",
        "recognition_probe_image_base64": _encode_rgb_to_base64(recognition_region_rgb),
    }
    if gallery_original_rgb is not None:
        response["original_image_base64"] = _encode_rgb_to_base64(gallery_original_rgb)
    return response


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    analysis = _prepare_face_analysis(img_bgr)
    face_for_recognition = analysis["face_tensor"]
    recognition_region_rgb = analysis["face_rgb"]
    if analysis["is_occluded"]:
        _, reconstructed_rgb = _blend_reconstructed_face(
            analysis["reconstruction_tensor"],
            analysis["occlusion_mask"],
            enhance_patch=True,
        )
        if analysis["recognition_is_occluded"]:
            face_for_recognition, recognition_region_rgb = _reconstructed_region_only_face(
                face_for_recognition, analysis["recognition_mask"]
            )
    else:
        reconstructed_rgb = analysis["reconstruction_rgb"]

    predicted_name, similarity, matched_image_path = _predict_identity(face_for_recognition)
    gallery_original_rgb = _original_image_from_match(matched_image_path)
    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": analysis["is_occluded"],
        "occlusion_region": analysis["occlusion_region"],
        "occlusion_ratio": round(float(analysis["occlusion_ratio"]), 4),
        "predicted_name": predicted_name,
        "similarity": round(similarity, 4),
        "confidence": round((similarity + 1.0) / 2.0, 4),
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "recognition_mode": "reconstructed-region-only" if analysis["recognition_is_occluded"] else "full-face",
        "input_image_base64": _encode_rgb_to_base64(analysis["reconstruction_rgb"]),
        "reconstructed_image_base64": _encode_rgb_to_base64(reconstructed_rgb),
        "recognition_probe_image_base64": _encode_rgb_to_base64(recognition_region_rgb),
    }
    if gallery_original_rgb is not None:
        response["original_image_base64"] = _encode_rgb_to_base64(gallery_original_rgb)
    return response


@app.post("/refresh-gallery")
def refresh_gallery():
    gallery_size = refresh_gallery_index()
    return {"status": "success", "gallery_size": gallery_size}
