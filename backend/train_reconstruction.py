import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models, transforms

try:
    from models import DEGAN, MTRUNet, PatchDiscriminator
except ModuleNotFoundError:
    from backend.models import DEGAN, MTRUNet, PatchDiscriminator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MOUTH_CHIN_MASK_RESTRICT_ENABLED_DEFAULT = True


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


class SyntheticFaceOcclusionAugmentor:
    def __init__(self, opacity_range: tuple[float, float] = (0.58, 0.94)):
        self.opacity_range = opacity_range
        self.modes = (
            "lower-mask",
            "sunglasses",
            "opaque-panel",
            "side-occlusion",
            "hand",
            "scarf",
        )

    def _sample_occluder_color(self, rng: np.random.Generator, style: str) -> np.ndarray:
        if style == "opaque-panel":
            palette = (
                np.array([226.0, 218.0, 206.0], dtype=np.float32),
                np.array([216.0, 203.0, 190.0], dtype=np.float32),
                np.array([196.0, 174.0, 154.0], dtype=np.float32),
                np.array([168.0, 140.0, 112.0], dtype=np.float32),
                np.array([132.0, 144.0, 156.0], dtype=np.float32),
                np.array([156.0, 164.0, 172.0], dtype=np.float32),
            )
            base = palette[int(rng.integers(0, len(palette)))]
            jitter = rng.uniform(-16.0, 16.0, size=3).astype(np.float32)
            return np.clip(base + jitter, 54.0, 238.0)
        if style == "side-occlusion":
            base = np.full(3, rng.uniform(70, 170), dtype=np.float32)
            return np.clip(base + rng.uniform(-18.0, 18.0, size=3).astype(np.float32), 48.0, 188.0)
        if style == "scarf":
            return np.full(3, rng.uniform(42, 178), dtype=np.float32)
        return np.full(3, rng.uniform(72, 196), dtype=np.float32)

    def _blend_mask(
        self,
        image_rgb: np.ndarray,
        mask_u8: np.ndarray,
        color: np.ndarray,
        opacity: float,
        *,
        blur_kernel: int = 9,
    ) -> tuple[np.ndarray, np.ndarray]:
        soft_mask = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (blur_kernel, blur_kernel), 0)
        alpha = np.clip(soft_mask[..., None] * opacity, 0.0, 1.0)
        color_layer = np.broadcast_to(color.reshape(1, 1, 3), image_rgb.shape)
        blended = image_rgb.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8), soft_mask

    def _skin_tone(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        y1 = max(0, int(h * 0.26))
        y2 = min(h, int(h * 0.72))
        x1 = max(0, int(w * 0.22))
        x2 = min(w, int(w * 0.78))
        patch = image_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            base = np.array([196.0, 164.0, 142.0], dtype=np.float32)
        else:
            base = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
        jitter = np.array([18.0, 8.0, 0.0], dtype=np.float32)
        return np.clip(base + jitter, 72.0, 240.0)

    def _draw_lower_mask(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        polygon = np.array(
            [
                [int(w * rng.uniform(0.12, 0.20)), int(h * rng.uniform(0.43, 0.50))],
                [int(w * rng.uniform(0.80, 0.88)), int(h * rng.uniform(0.43, 0.50))],
                [int(w * rng.uniform(0.72, 0.86)), int(h * rng.uniform(0.74, 0.88))],
                [int(w * rng.uniform(0.14, 0.28)), int(h * rng.uniform(0.74, 0.88))],
            ],
            dtype=np.int32,
        )
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, polygon, 255)
        neutral = np.full(3, rng.uniform(165, 228), dtype=np.float32)
        image_rgb, soft_mask = self._blend_mask(
            image_rgb,
            mask_u8,
            neutral,
            opacity=float(rng.uniform(*self.opacity_range)),
            blur_kernel=9,
        )
        cv2.polylines(image_rgb, [polygon], isClosed=True, color=(148, 148, 148), thickness=1, lineType=cv2.LINE_AA)
        return image_rgb, soft_mask

    def _draw_sunglasses(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        x1 = int(w * rng.uniform(0.12, 0.18))
        x2 = int(w * rng.uniform(0.82, 0.88))
        y1 = int(h * rng.uniform(0.20, 0.28))
        y2 = int(h * rng.uniform(0.39, 0.47))
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_u8, (x1, y1), (x2, y2), 255, thickness=-1)
        dark_color = np.full(3, rng.uniform(12, 42), dtype=np.float32)
        image_rgb, soft_mask = self._blend_mask(
            image_rgb,
            mask_u8,
            dark_color,
            opacity=float(rng.uniform(0.82, 0.97)),
            blur_kernel=7,
        )
        bridge_y = int((y1 + y2) / 2)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (58, 58, 58), thickness=2)
        cv2.line(image_rgb, (x1, bridge_y), (x2, bridge_y), (72, 72, 72), 1, cv2.LINE_AA)
        return image_rgb, soft_mask

    def _draw_opaque_panel(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        x1 = int(w * rng.uniform(0.16, 0.34))
        x2 = int(w * rng.uniform(0.64, 0.86))
        y1 = int(h * rng.uniform(0.31, 0.49))
        y2 = int(h * rng.uniform(0.61, 0.84))
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_u8, (x1, y1), (x2, y2), 255, thickness=-1)
        panel = self._sample_occluder_color(rng, "opaque-panel")
        image_rgb, soft_mask = self._blend_mask(
            image_rgb,
            mask_u8,
            panel,
            opacity=float(rng.uniform(0.86, 1.0)),
            blur_kernel=5,
        )
        outline = tuple(int(channel) for channel in np.clip(panel * 0.82, 40.0, 170.0))
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), outline, thickness=1)
        if rng.random() < 0.65:
            line_color = tuple(int(channel) for channel in np.clip(panel * rng.uniform(0.45, 0.78), 28.0, 190.0))
            if rng.random() < 0.5:
                cv2.line(image_rgb, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), line_color, 1, cv2.LINE_AA)
            else:
                cv2.line(image_rgb, (x1 + 2, y2 - 2), (x2 - 2, y1 + 2), line_color, 1, cv2.LINE_AA)
        if rng.random() < 0.45:
            strap_color = tuple(int(channel) for channel in np.clip(panel * 0.75 + 28.0, 60.0, 220.0))
            y_mid = int((y1 + y2) * 0.5)
            cv2.line(image_rgb, (max(0, x1 - int(w * 0.12)), y_mid + int(rng.uniform(-5, 5))), (x1, y_mid), strap_color, 1, cv2.LINE_AA)
            cv2.line(image_rgb, (x2, y_mid), (min(w - 1, x2 + int(w * 0.12)), y_mid + int(rng.uniform(-5, 5))), strap_color, 1, cv2.LINE_AA)
        return image_rgb, soft_mask

    def _draw_side_occlusion(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        cover_left = bool(rng.integers(0, 2))
        if cover_left:
            polygon = np.array(
                [
                    [int(w * rng.uniform(0.00, 0.10)), int(h * rng.uniform(0.16, 0.26))],
                    [int(w * rng.uniform(0.32, 0.46)), int(h * rng.uniform(0.20, 0.32))],
                    [int(w * rng.uniform(0.40, 0.54)), int(h * rng.uniform(0.74, 0.90))],
                    [int(w * rng.uniform(0.00, 0.12)), int(h * rng.uniform(0.78, 0.96))],
                ],
                dtype=np.int32,
            )
        else:
            polygon = np.array(
                [
                    [int(w * rng.uniform(0.54, 0.68)), int(h * rng.uniform(0.20, 0.32))],
                    [int(w * rng.uniform(0.90, 1.00)), int(h * rng.uniform(0.16, 0.26))],
                    [int(w * rng.uniform(0.88, 1.00)), int(h * rng.uniform(0.78, 0.96))],
                    [int(w * rng.uniform(0.46, 0.60)), int(h * rng.uniform(0.74, 0.90))],
                ],
                dtype=np.int32,
            )
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, polygon, 255)
        panel = self._sample_occluder_color(rng, "side-occlusion")
        image_rgb, soft_mask = self._blend_mask(
            image_rgb,
            mask_u8,
            panel,
            opacity=float(rng.uniform(0.62, 0.90)),
            blur_kernel=11,
        )
        return image_rgb, soft_mask

    def _draw_hand(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        center = (
            int(w * rng.uniform(0.35, 0.65)),
            int(h * rng.uniform(0.48, 0.68)),
        )
        axes = (
            int(w * rng.uniform(0.22, 0.34)),
            int(h * rng.uniform(0.14, 0.26)),
        )
        angle = float(rng.uniform(-28, 28))
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_u8, center, axes, angle, 0, 360, 255, thickness=-1)
        finger_offsets = (-0.18, -0.06, 0.06, 0.18)
        for offset in finger_offsets:
            finger_center = (
                int(center[0] + offset * w * 0.40),
                int(center[1] - axes[1] * rng.uniform(0.45, 0.78)),
            )
            finger_axes = (
                int(axes[0] * rng.uniform(0.16, 0.24)),
                int(axes[1] * rng.uniform(0.36, 0.52)),
            )
            cv2.ellipse(mask_u8, finger_center, finger_axes, angle + rng.uniform(-12, 12), 0, 360, 255, thickness=-1)
        color = self._skin_tone(image_rgb)
        return self._blend_mask(
            image_rgb,
            mask_u8,
            color,
            opacity=float(rng.uniform(0.65, 0.92)),
            blur_kernel=11,
        )

    def _draw_scarf(self, image_rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        top_y = int(h * rng.uniform(0.52, 0.62))
        bottom_y = int(h * rng.uniform(0.84, 0.96))
        steps = 7
        xs = np.linspace(0, w - 1, steps).astype(np.int32)
        top_curve = []
        bottom_curve = []
        for idx, x in enumerate(xs):
            wave = np.sin(idx / max(1, steps - 1) * np.pi)
            top_curve.append([int(x), int(top_y + wave * rng.uniform(-8, 10))])
            bottom_curve.append([int(x), int(bottom_y + wave * rng.uniform(-4, 8))])
        polygon = np.array(top_curve + bottom_curve[::-1], dtype=np.int32)
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_u8, [polygon], 255)
        scarf_color = self._sample_occluder_color(rng, "scarf")
        image_rgb, soft_mask = self._blend_mask(
            image_rgb,
            mask_u8,
            scarf_color,
            opacity=float(rng.uniform(0.60, 0.88)),
            blur_kernel=11,
        )
        return image_rgb, soft_mask

    def _apply_single(self, image_rgb: np.ndarray, style: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if style == "lower-mask":
            return self._draw_lower_mask(image_rgb, rng)
        if style == "sunglasses":
            return self._draw_sunglasses(image_rgb, rng)
        if style == "opaque-panel":
            return self._draw_opaque_panel(image_rgb, rng)
        if style == "side-occlusion":
            return self._draw_side_occlusion(image_rgb, rng)
        if style == "hand":
            return self._draw_hand(image_rgb, rng)
        if style == "scarf":
            return self._draw_scarf(image_rgb, rng)
        raise ValueError(f"Unsupported occlusion style: {style}")

    def apply(self, image_rgb: np.ndarray, rng: np.random.Generator, mode: str = "mixed") -> tuple[np.ndarray, np.ndarray, str]:
        output = image_rgb.copy()
        aggregate_mask = np.zeros(image_rgb.shape[:2], dtype=np.float32)

        if mode == "mixed":
            available = list(self.modes)
            count = 1 if rng.random() < 0.72 else 2
            selected = list(rng.choice(available, size=count, replace=False))
            applied_mode = "+".join(selected)
        else:
            selected = [mode]
            applied_mode = mode

        for style in selected:
            output, soft_mask = self._apply_single(output, style, rng)
            aggregate_mask = np.maximum(aggregate_mask, soft_mask)

        return output, np.clip(aggregate_mask, 0.0, 1.0), applied_mode


class MaskedFacePairDataset(Dataset):
    def __init__(
        self,
        masked_root: str,
        original_root: str,
        mask_threshold: float = 0.08,
        synthetic_probability: float = 0.65,
        synthetic_mode: str = "mixed",
        extra_masked_roots: list[str] | None = None,
        extra_original_roots: list[str] | None = None,
        max_extra_originals: int | None = None,
        filter_extra_originals: bool = True,
        match_extra_illumination: bool = True,
        indices: list[int] | None = None,
        image_size: int = 112,
        restrict_mouth_chin_mask: bool = MOUTH_CHIN_MASK_RESTRICT_ENABLED_DEFAULT,
    ):
        self.masked_root = Path(masked_root)
        self.original_root = Path(original_root)
        self.mask_threshold = mask_threshold
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 because DEGAN uses three down/up-sampling stages.")
        self.image_size = int(image_size)
        self.restrict_mouth_chin_mask = restrict_mouth_chin_mask
        self.synthetic_probability = float(np.clip(synthetic_probability, 0.0, 1.0))
        self.synthetic_mode = synthetic_mode
        self.extra_masked_roots = [Path(root) for root in (extra_masked_roots or [])]
        self.extra_original_roots = [Path(root) for root in (extra_original_roots or [])]
        self.max_extra_originals = max_extra_originals
        self.filter_extra_originals = filter_extra_originals
        self.match_extra_illumination = match_extra_illumination
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.synthetic_augmentor = SyntheticFaceOcclusionAugmentor()
        all_samples = self._collect_pairs()
        if indices is None:
            self.samples = all_samples
        else:
            self.samples = [all_samples[idx] for idx in indices]
        self.reference_luminance_mean, self.reference_luminance_std = self._estimate_reference_luminance()
        self.extra_originals = self._collect_extra_originals()

    def _collect_pairs(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        masked_roots = [self.masked_root] + self.extra_masked_roots
        for masked_root in masked_roots:
            if not masked_root.exists():
                print(f"Skipping extra masked dataset because it was not found: {masked_root}")
                continue
            for masked_path in masked_root.rglob("*"):
                if masked_path.suffix.lower() not in IMAGE_EXTENSIONS or not masked_path.is_file():
                    continue

                rel_path = masked_path.relative_to(masked_root)
                original_path = self.original_root / rel_path
                if original_path.exists():
                    pairs.append((masked_path, original_path))

        if not pairs:
            raise RuntimeError("No masked/original training pairs were found.")
        return pairs

    def _collect_extra_originals(self) -> list[Path]:
        collected: list[Path] = []
        seen: set[Path] = set()
        cap = self.max_extra_originals if self.max_extra_originals is not None and self.max_extra_originals >= 0 else None
        for root in self.extra_original_roots:
            if not root.exists():
                print(f"Skipping extra original dataset because it was not found: {root}")
                continue
            for image_path in root.rglob("*"):
                if cap is not None and len(collected) >= cap:
                    break
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    if image_path in seen:
                        continue
                    if self.filter_extra_originals and not self._is_usable_extra_original(image_path):
                        continue
                    collected.append(image_path)
                    seen.add(image_path)
            if cap is not None and len(collected) >= cap:
                break
        return sorted(collected)

    def _estimate_reference_luminance(self, sample_limit: int = 512) -> tuple[float, float]:
        luminance_means: list[float] = []
        for _, original_path in self.samples[:sample_limit]:
            try:
                original = Image.open(original_path).convert("RGB").resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR,
                )
            except Exception:
                continue
            y_channel = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2YCrCb)[:, :, 0]
            luminance_means.append(float(y_channel.mean()))
        if not luminance_means:
            return 110.0, 38.0
        return float(np.mean(luminance_means)), float(np.std(luminance_means) + 1e-6)

    def _brightness_metrics(self, image_path: Path) -> tuple[float, float]:
        with Image.open(image_path).convert("RGB") as img:
            arr = np.array(
                img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        mean_luma = float(cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)[:, :, 0].mean())
        highlight_fraction = float((arr.mean(axis=2) > 235).mean())
        return mean_luma, highlight_fraction

    def _is_usable_extra_original(self, image_path: Path) -> bool:
        try:
            mean_luma, highlight_fraction = self._brightness_metrics(image_path)
        except Exception:
            return False
        if highlight_fraction > 0.18:
            return False
        if mean_luma > self.reference_luminance_mean + 1.75 * max(self.reference_luminance_std, 1.0):
            return False
        return True

    def _match_reference_illumination(self, original_rgb: np.ndarray) -> np.ndarray:
        if not self.match_extra_illumination:
            return original_rgb

        ycrcb = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y_channel = ycrcb[:, :, 0]
        current_mean = float(y_channel.mean())
        current_std = float(y_channel.std() + 1e-6)
        target_mean = self.reference_luminance_mean
        target_std = max(self.reference_luminance_std, 1.0)

        remapped_y = (y_channel - current_mean) * (target_std / current_std) + target_mean
        remapped_y = np.clip(remapped_y, 0.0, 255.0)

        highlight_mask = remapped_y > 228.0
        if np.any(highlight_mask):
            remapped_y[highlight_mask] = 228.0 + (remapped_y[highlight_mask] - 228.0) * 0.35

        ycrcb[:, :, 0] = remapped_y
        return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    def __len__(self) -> int:
        return len(self.samples) + len(self.extra_originals)

    def _paired_sample(self, masked: Image.Image, original: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_tensor = self.transform(masked)
        original_tensor = self.transform(original)

        masked_01 = (masked_tensor + 1.0) / 2.0
        original_01 = (original_tensor + 1.0) / 2.0
        diff_map = (masked_01 - original_01).abs().mean(dim=0, keepdim=True)
        soft_diff = torch.clamp((diff_map - self.mask_threshold) / max(self.mask_threshold, 1e-6), min=0.0, max=1.0)
        occlusion_mask = soft_diff.unsqueeze(0)
        occlusion_mask = F.max_pool2d(occlusion_mask, kernel_size=7, stride=1, padding=3).squeeze(0)
        if self.restrict_mouth_chin_mask:
            occlusion_mask = _restrict_mask_to_mouth_chin(occlusion_mask.unsqueeze(0)).squeeze(0)
        return masked_tensor, original_tensor, occlusion_mask

    def _synthetic_sample(self, original: Image.Image, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_resized = original.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        original_rgb = np.array(original_resized.convert("RGB"))
        original_rgb = self._match_reference_illumination(original_rgb)
        seed = (torch.initial_seed() + index * 7919) % (2**32)
        rng = np.random.default_rng(seed)
        masked_rgb, occlusion_mask, _ = self.synthetic_augmentor.apply(original_rgb, rng, mode=self.synthetic_mode)

        masked_tensor = self.transform(Image.fromarray(masked_rgb))
        original_tensor = self.transform(Image.fromarray(original_rgb))
        occlusion_tensor = torch.from_numpy(occlusion_mask).float().unsqueeze(0)
        if self.restrict_mouth_chin_mask:
            occlusion_tensor = _restrict_mask_to_mouth_chin(occlusion_tensor.unsqueeze(0)).squeeze(0)
        return masked_tensor, original_tensor, occlusion_tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < len(self.samples):
            masked_path, original_path = self.samples[index]
            masked = Image.open(masked_path).convert("RGB")
            original = Image.open(original_path).convert("RGB")
            use_synthetic = self.synthetic_probability > 0.0 and torch.rand(1).item() < self.synthetic_probability
            if use_synthetic:
                return self._synthetic_sample(original, index)
            return self._paired_sample(masked, original)

        extra_index = index - len(self.samples)
        original = Image.open(self.extra_originals[extra_index]).convert("RGB")
        if self.synthetic_probability <= 0.0:
            # Validation and metrics rely on paired supervision only, so extra
            # original-only images should not normally be used when synthetic
            # samples are disabled.
            return self._synthetic_sample(original, extra_index)
        else:
            return self._synthetic_sample(original, index)


def _masked_average(error: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mask.shape[1] == 1 and error.shape[1] != 1:
        mask = mask.expand(-1, error.shape[1], -1, -1)
    elif mask.shape[1] != error.shape[1]:
        mask = mask.mean(dim=1, keepdim=True).expand(-1, error.shape[1], -1, -1)
    weighted_error = error * mask
    denom = mask.sum(dim=(1, 2, 3), keepdim=False)
    denom = denom.clamp_min(eps)
    return weighted_error.sum(dim=(1, 2, 3)) / denom


def _extract_region_only_face(face_tensor: torch.Tensor, occlusion_mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    mask = occlusion_mask.clamp(0.0, 1.0)
    if mask.shape[1] == 1 and face_tensor.shape[1] != 1:
        mask = mask.expand(-1, face_tensor.shape[1], -1, -1)
    return face_tensor * mask + fill_value * (1.0 - mask)


def _expand_mask(mask: torch.Tensor, channels: int) -> torch.Tensor:
    if mask.shape[1] == channels:
        return mask
    return mask.expand(-1, channels, -1, -1)


def _mouth_chin_roi_like(mask: torch.Tensor) -> torch.Tensor:
    h, w = mask.shape[-2:]
    roi = torch.zeros_like(mask)
    y1 = max(0, int(round(0.30 * h)))
    y2 = min(h, int(round(0.84 * h)))
    x1 = max(0, int(round(0.10 * w)))
    x2 = min(w, int(round(0.90 * w)))
    roi[:, :, y1:y2, x1:x2] = 1.0
    return roi


def _restrict_mask_to_mouth_chin(mask: torch.Tensor) -> torch.Tensor:
    restricted = mask.clamp(0.0, 1.0) * _mouth_chin_roi_like(mask)
    original_area = mask.sum(dim=(1, 2, 3), keepdim=True)
    restricted_area = restricted.sum(dim=(1, 2, 3), keepdim=True)
    # Keep non-lower-face training cases usable if the caller still trains mixed occlusions.
    keep_original = (original_area > 20.0) & (restricted_area < original_area * 0.18)
    return torch.where(keep_original, mask.clamp(0.0, 1.0), restricted)


def _harden_mask(mask: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    hard_mask = (mask > threshold).to(mask.dtype)
    hard_mask = F.max_pool2d(hard_mask, kernel_size=7, stride=1, padding=3)
    hard_mask = F.max_pool2d(hard_mask, kernel_size=5, stride=1, padding=2)
    return hard_mask.clamp(0.0, 1.0)


def _sanitize_masked_input(masked_inputs: torch.Tensor, occlusion_masks: torch.Tensor) -> torch.Tensor:
    hard_mask = _expand_mask(_harden_mask(occlusion_masks), masked_inputs.shape[1])
    local_mean = F.avg_pool2d(masked_inputs * (1.0 - hard_mask), kernel_size=17, stride=1, padding=8)
    local_count = F.avg_pool2d(1.0 - hard_mask, kernel_size=17, stride=1, padding=8).clamp_min(1e-3)
    fill = (local_mean / local_count).clamp(-1.0, 1.0)
    return masked_inputs * (1.0 - hard_mask) + fill * hard_mask


def _gradient_map(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def _laplacian_map(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    return F.conv2d(image, kernel, padding=1, groups=channels)


def _ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    window_size: int = 7,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction_01 = (prediction.clamp(-1.0, 1.0) + 1.0) / 2.0
    target_01 = (target.clamp(-1.0, 1.0) + 1.0) / 2.0

    mu_x = F.avg_pool2d(prediction_01, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target_01, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(prediction_01 * prediction_01, window_size, stride=1, padding=window_size // 2) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target_01 * target_01, window_size, stride=1, padding=window_size // 2) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction_01 * target_01, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + eps
    )
    loss_map = (1.0 - ssim_map.clamp(-1.0, 1.0)) * 0.5

    if mask is None:
        return loss_map.mean()

    mask_rgb = _expand_mask(mask.clamp(0.0, 1.0), prediction.shape[1])
    denom = mask_rgb.sum(dim=(1, 2, 3)).clamp_min(eps)
    return ((loss_map * mask_rgb).sum(dim=(1, 2, 3)) / denom).mean()


def _recognizer_input_from_normalized_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    rgb_255 = ((face_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).clamp(0.0, 255.0)
    resized = F.interpolate(rgb_255, size=(160, 160), mode="bilinear", align_corners=False)
    return (resized - 127.5) / 128.0


class PerceptualLoss(nn.Module):
    """VGG-16 feature matching loss that compares high-level texture and
    structure features rather than raw pixels, preventing the blurry
    averaged-face outputs caused by pure L1/MSE supervision."""

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        # VGG expects ImageNet normalization on [0,1] input
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from [-1,1] to [0,1], resize to 128 (saves VRAM vs 224)
        x_01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        x_resized = F.interpolate(x_01, size=128, mode="bilinear", align_corners=False)
        return (x_resized - self.mean) / self.std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.vgg(self._prepare(prediction)), self.vgg(self._prepare(target.detach())))


def _masked_perceptual_focus(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    perceptual_loss_fn: PerceptualLoss,
) -> torch.Tensor:
    """Apply VGG loss mainly to the hidden region without losing face context."""
    mask = mask.clamp(0.0, 1.0)
    mask_rgb = _expand_mask(mask, prediction.shape[1])
    # Keep a little neutral context around the hole; pure black outside the
    # mask makes VGG chase artificial boundaries instead of facial structure.
    neutral = torch.zeros_like(prediction)
    prediction_focus = prediction * mask_rgb + neutral * (1.0 - mask_rgb)
    target_focus = target.detach() * mask_rgb + neutral * (1.0 - mask_rgb)
    return perceptual_loss_fn(prediction_focus, target_focus)


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1,1] tensor to a [H,W,3] uint8 RGB image."""
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) * 127.5).squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(img, 0, 255).astype(np.uint8)


def _forward_reconstruction(
    model: DEGAN,
    masked_inputs: torch.Tensor,
    occlusion_masks: torch.Tensor,
    refiner: MTRUNet | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard_masks = _harden_mask(occlusion_masks)
    mask_rgb = _expand_mask(hard_masks, masked_inputs.shape[1])
    sanitized_inputs = _sanitize_masked_input(masked_inputs, occlusion_masks)
    if refiner is None:
        predicted = model(sanitized_inputs, hard_masks)
        blended = masked_inputs * (1.0 - mask_rgb) + predicted * mask_rgb
        return blended, predicted

    with torch.no_grad():
        predicted = model(sanitized_inputs, hard_masks)
    refined = refiner(masked_inputs, predicted.detach(), hard_masks)
    return refined, predicted


def _save_epoch_samples(
    model: DEGAN,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    max_samples: int = 6,
    refiner: MTRUNet | None = None,
) -> None:
    """Save per-epoch raw generator outputs and Original/Masked/Raw/Blended grids."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = output_dir / "raw_generator_outputs" / f"epoch_{epoch + 1:03d}"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    model_was_training = model.training
    refiner_was_training = refiner.training if refiner is not None else False
    model.eval()
    if refiner is not None:
        refiner.eval()
    rows = []
    count = 0

    with torch.no_grad():
        for masked_inputs, original_targets, occlusion_masks in dataloader:
            masked_inputs = masked_inputs.to(device)
            original_targets = original_targets.to(device)
            occlusion_masks = occlusion_masks.to(device)

            blended, predicted = _forward_reconstruction(model, masked_inputs, occlusion_masks, refiner)
            hard_masks = _harden_mask(occlusion_masks)

            for j in range(masked_inputs.shape[0]):
                if count >= max_samples:
                    break
                original = _tensor_to_rgb_uint8(original_targets[j])
                masked = _tensor_to_rgb_uint8(masked_inputs[j])
                raw = _tensor_to_rgb_uint8(predicted[j])
                hard_mask_rgb = _expand_mask(hard_masks[j:j + 1], masked_inputs.shape[1])
                region_only = _tensor_to_rgb_uint8(
                    masked_inputs[j:j + 1] * (1.0 - hard_mask_rgb)
                    + predicted[j:j + 1] * hard_mask_rgb
                )
                rec = _tensor_to_rgb_uint8(blended[j])
                diff = np.clip(np.abs(rec.astype(np.float32) - original.astype(np.float32)) * 3.0, 0, 255).astype(np.uint8)
                cv2.imwrite(
                    str(raw_output_dir / f"raw_{count + 1:02d}.png"),
                    cv2.cvtColor(raw, cv2.COLOR_RGB2BGR),
                )
                row = np.hstack([original, masked, raw, region_only, rec, diff])
                rows.append(row)
                count += 1
            if count >= max_samples:
                break

    if refiner is not None:
        if refiner_was_training:
            refiner.train()
        model.eval()
    elif model_was_training:
        model.train()
    if not rows:
        return

    # Add column headers on first row
    h, w = rows[0].shape[:2]
    header = np.zeros((20, w, 3), dtype=np.uint8)
    col_w = w // 6
    for idx, label in enumerate(["Original", "Masked", "Raw GAN", "Region Only", "Final Blend", "Difference"]):
        cv2.putText(header, label, (idx * col_w + 4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    grid = np.vstack([header] + rows)
    out_path = output_dir / f"epoch_{epoch + 1:03d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved training samples -> {out_path}")


def _load_partial_checkpoint(model: nn.Module, checkpoint_path: str | None, device: torch.device) -> bool:
    if not checkpoint_path:
        return False

    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Checkpoint not found, starting from scratch: {path}")
        return False

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible_state:
        print(f"No compatible tensors found in checkpoint: {path}")
        return False

    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=False)
    print(f"Warm-started {model.__class__.__name__} from {path} with {len(compatible_state)} compatible tensors.")
    return True


def _total_variation_loss(image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    tv_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
    tv_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    if mask is None:
        return tv_h.mean() + tv_w.mean()
    mask = mask.clamp(0.0, 1.0)
    mask_h = _expand_mask(mask[:, :, 1:, :], image.shape[1])
    mask_w = _expand_mask(mask[:, :, :, 1:], image.shape[1])
    return _masked_average(tv_h, mask_h).mean() + _masked_average(tv_w, mask_w).mean()


def _masked_discriminator_view(image: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_rgb = _expand_mask(_harden_mask(mask), image.shape[1])
    return image * mask_rgb + target.detach() * (1.0 - mask_rgb)


def _compute_reconstruction_losses(
    model: DEGAN,
    recognition_model: InceptionResnetV1,
    masked_inputs: torch.Tensor,
    original_targets: torch.Tensor,
    occlusion_masks: torch.Tensor,
    mask_weight: float,
    full_weight: float,
    identity_weight: float,
    global_identity_weight: float,
    edge_weight: float,
    structure_weight: float,
    perceptual_loss_fn: PerceptualLoss | None = None,
    perceptual_weight: float = 0.0,
    discriminator: PatchDiscriminator | None = None,
    adversarial_weight: float = 0.0,
    refiner: MTRUNet | None = None,
    tv_weight: float = 0.0,
    detail_weight: float = 0.0,
    outside_weight: float = 1.0,
    teacher_identity_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_masks = _harden_mask(occlusion_masks)
    mask_rgb = _expand_mask(hard_masks, masked_inputs.shape[1])
    outside_mask_rgb = 1.0 - mask_rgb
    blended_outputs, raw_outputs = _forward_reconstruction(model, masked_inputs, occlusion_masks, refiner)

    abs_error = (blended_outputs - original_targets).abs()
    sq_error = (blended_outputs - original_targets) ** 2

    masked_l1 = _masked_average(abs_error, mask_rgb).mean()
    masked_mse = _masked_average(sq_error, mask_rgb).mean()
    full_l1 = abs_error.mean()
    full_mse = sq_error.mean()
    raw_abs_error = (raw_outputs - original_targets).abs()
    raw_sq_error = (raw_outputs - original_targets) ** 2
    outside_l1 = _masked_average(raw_abs_error, outside_mask_rgb).mean()
    outside_mse = _masked_average(raw_sq_error, outside_mask_rgb).mean()

    masked_loss = 0.75 * masked_l1 + 0.25 * masked_mse
    full_loss = 0.85 * full_l1 + 0.15 * full_mse
    outside_loss = 0.85 * outside_l1 + 0.15 * outside_mse

    predicted_edges = _gradient_map(blended_outputs)
    target_edges = _gradient_map(original_targets)
    edge_error = (predicted_edges - target_edges).abs()
    masked_edge = _masked_average(edge_error, mask_rgb).mean()
    full_edge = edge_error.mean()
    edge_loss = masked_edge

    predicted_detail = _laplacian_map(blended_outputs)
    target_detail = _laplacian_map(original_targets)
    detail_error = (predicted_detail - target_detail).abs()
    detail_loss = _masked_average(detail_error, mask_rgb).mean()

    masked_structure = _ssim_loss(blended_outputs, original_targets, hard_masks)
    full_structure = _ssim_loss(blended_outputs, original_targets)
    structure_loss = masked_structure

    reconstructed_full_embedding = recognition_model(_recognizer_input_from_normalized_tensor(blended_outputs))
    with torch.no_grad():
        original_full_embedding = recognition_model(_recognizer_input_from_normalized_tensor(original_targets))

    full_identity_similarity = F.cosine_similarity(
        reconstructed_full_embedding, original_full_embedding, dim=1
    ).mean()
    full_identity_loss = 1.0 - full_identity_similarity

    teacher_identity_loss = torch.tensor(0.0, device=masked_inputs.device)
    teacher_identity_similarity = torch.tensor(0.0, device=masked_inputs.device)
    if teacher_identity_weight > 0.0:
        teacher_outputs = masked_inputs * (1.0 - mask_rgb) + raw_outputs.detach() * mask_rgb
        with torch.no_grad():
            teacher_embedding = recognition_model(_recognizer_input_from_normalized_tensor(teacher_outputs))
        teacher_identity_similarity = F.cosine_similarity(
            reconstructed_full_embedding, teacher_embedding, dim=1
        ).mean()
        teacher_identity_loss = 1.0 - teacher_identity_similarity

    # Perceptual loss: feature-level matching for texture/structure quality
    p_loss = torch.tensor(0.0, device=masked_inputs.device)
    if perceptual_loss_fn is not None and perceptual_weight > 0.0:
        masked_perceptual = _masked_perceptual_focus(blended_outputs, original_targets, hard_masks, perceptual_loss_fn)
        p_loss = masked_perceptual

    # Generator adversarial loss: fool the discriminator into accepting
    # the reconstruction as real
    g_adv_loss = torch.tensor(0.0, device=masked_inputs.device)
    if discriminator is not None and adversarial_weight > 0.0:
        fake_pred = discriminator(_masked_discriminator_view(blended_outputs, original_targets, hard_masks))
        g_adv_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )

    tv_loss = torch.tensor(0.0, device=masked_inputs.device)
    if tv_weight > 0.0:
        tv_loss = _total_variation_loss(blended_outputs, hard_masks)

    total_loss = (
        mask_weight * masked_loss
        + full_weight * full_loss
        + outside_weight * outside_loss
        + identity_weight * full_identity_loss
        + global_identity_weight * full_identity_loss
        + edge_weight * edge_loss
        + structure_weight * structure_loss
        + perceptual_weight * p_loss
        + adversarial_weight * g_adv_loss
        + tv_weight * tv_loss
        + detail_weight * detail_loss
        + teacher_identity_weight * teacher_identity_loss
    )
    metrics = {
        "loss": float(total_loss.detach().item()),
        "masked_l1": float(masked_l1.detach().item()),
        "full_l1": float(full_l1.detach().item()),
        "outside_l1": float(outside_l1.detach().item()),
        "outside_preservation": float(outside_loss.detach().item()),
        "edge_loss": float(edge_loss.detach().item()),
        "structure_loss": float(structure_loss.detach().item()),
        "ssim": float((1.0 - masked_structure.detach()).item()),
        "full_identity": float(full_identity_loss.detach().item()),
        "identity_similarity": float(full_identity_similarity.detach().item()),
        "teacher_identity": float(teacher_identity_loss.detach().item()),
        "teacher_identity_similarity": float(teacher_identity_similarity.detach().item()),
        "perceptual": float(p_loss.detach().item()),
        "g_adv": float(g_adv_loss.detach().item()),
        "tv": float(tv_loss.detach().item()),
        "detail": float(detail_loss.detach().item()),
    }
    return total_loss, metrics


def _evaluate_epoch(
    model: DEGAN,
    recognition_model: InceptionResnetV1,
    dataloader: DataLoader,
    device: torch.device,
    mask_weight: float,
    full_weight: float,
    identity_weight: float,
    global_identity_weight: float,
    edge_weight: float,
    structure_weight: float,
    perceptual_loss_fn: PerceptualLoss | None = None,
    perceptual_weight: float = 0.0,
    refiner: MTRUNet | None = None,
    tv_weight: float = 0.0,
    detail_weight: float = 0.0,
    outside_weight: float = 1.0,
    teacher_identity_weight: float = 0.0,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "masked_l1": 0.0,
        "full_l1": 0.0,
        "outside_l1": 0.0,
        "outside_preservation": 0.0,
        "edge_loss": 0.0,
        "structure_loss": 0.0,
        "ssim": 0.0,
        "full_identity": 0.0,
        "identity_similarity": 0.0,
        "teacher_identity": 0.0,
        "teacher_identity_similarity": 0.0,
        "perceptual": 0.0,
        "g_adv": 0.0,
        "tv": 0.0,
        "detail": 0.0,
    }
    batches = 0

    model_was_training = model.training
    refiner_was_training = refiner.training if refiner is not None else False
    model.eval()
    if refiner is not None:
        refiner.eval()
    with torch.no_grad():
        for masked_inputs, original_targets, occlusion_masks in dataloader:
            masked_inputs = masked_inputs.to(device, non_blocking=True)
            original_targets = original_targets.to(device, non_blocking=True)
            occlusion_masks = occlusion_masks.to(device, non_blocking=True)

            _, metrics = _compute_reconstruction_losses(
                model,
                recognition_model,
                masked_inputs,
                original_targets,
                occlusion_masks,
                mask_weight,
                full_weight,
                identity_weight,
                global_identity_weight,
                edge_weight,
                structure_weight,
                perceptual_loss_fn=perceptual_loss_fn,
                perceptual_weight=perceptual_weight,
                refiner=refiner,
                tv_weight=tv_weight,
                detail_weight=detail_weight,
                outside_weight=outside_weight,
                teacher_identity_weight=teacher_identity_weight,
            )
            for key in totals:
                totals[key] += metrics.get(key, 0.0)
            batches += 1

    if refiner is not None:
        if refiner_was_training:
            refiner.train()
        model.eval()
    elif model_was_training:
        model.train()
    if batches == 0:
        return totals
    return {key: value / batches for key, value in totals.items()}


def train(
    masked_dir: str,
    original_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    mask_weight: float,
    full_weight: float,
    identity_weight: float,
    global_identity_weight: float,
    edge_weight: float,
    structure_weight: float,
    mask_threshold: float,
    val_split: float,
    resume: str | None,
    synthetic_probability: float,
    synthetic_mode: str,
    extra_masked_dirs: list[str],
    extra_original_dirs: list[str],
    max_extra_originals: int | None,
    perceptual_weight: float = 1.0,
    adversarial_weight: float = 0.1,
    tv_weight: float = 0.02,
    detail_weight: float = 0.0,
    outside_weight: float = 1.0,
    teacher_identity_weight: float = 0.0,
    disc_lr: float = 1e-4,
    disc_start_epoch: int = 5,
    use_refiner: bool = False,
    refiner_resume: str | None = None,
    refiner_delta_scale: float = 0.75,
    filter_extra_originals: bool = True,
    image_size: int = 112,
    restrict_mouth_chin_mask: bool = MOUTH_CHIN_MASK_RESTRICT_ENABLED_DEFAULT,
    weights_output_dir: str = "weights",
    sample_output_dir_override: str | None = None,
    epoch_offset: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting {'MTR-UNet refiner' if use_refiner else 'DEGAN reconstruction'} training on: {device}")
    print(f"Reconstruction training resolution: {image_size}x{image_size}")

    base_dataset = MaskedFacePairDataset(
        masked_dir, original_dir, mask_threshold=mask_threshold,
        synthetic_probability=0.0, synthetic_mode=synthetic_mode,
        extra_masked_roots=extra_masked_dirs, extra_original_roots=[],
        max_extra_originals=None,
        image_size=image_size,
        restrict_mouth_chin_mask=restrict_mouth_chin_mask,
    )
    val_size = max(1, int(len(base_dataset) * val_split))
    if val_size >= len(base_dataset):
        val_size = max(1, len(base_dataset) // 10)
    train_size = len(base_dataset) - val_size
    generator = torch.Generator().manual_seed(23)
    shuffled_indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]
    train_dataset = MaskedFacePairDataset(
        masked_dir, original_dir, mask_threshold=mask_threshold,
        synthetic_probability=synthetic_probability, synthetic_mode=synthetic_mode,
        extra_masked_roots=extra_masked_dirs, extra_original_roots=extra_original_dirs,
        max_extra_originals=max_extra_originals,
        filter_extra_originals=filter_extra_originals,
        indices=train_indices,
        image_size=image_size,
        restrict_mouth_chin_mask=restrict_mouth_chin_mask,
    )
    val_dataset = MaskedFacePairDataset(
        masked_dir, original_dir, mask_threshold=mask_threshold,
        synthetic_probability=0.0, synthetic_mode=synthetic_mode,
        extra_masked_roots=extra_masked_dirs, extra_original_roots=[],
        max_extra_originals=None, indices=val_indices,
        image_size=image_size,
        restrict_mouth_chin_mask=restrict_mouth_chin_mask,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = DEGAN().to(device)
    _load_partial_checkpoint(model, resume, device)
    refiner = MTRUNet(delta_scale=refiner_delta_scale).to(device) if use_refiner else None
    if refiner is not None:
        _load_partial_checkpoint(refiner, refiner_resume, device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        optimizer_g = optim.Adam(refiner.parameters(), lr=lr)
    else:
        optimizer_g = optim.Adam(model.parameters(), lr=lr)

    # Discriminator for adversarial sharpness feedback
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    # Perceptual loss for texture/structure quality
    perceptual_loss_fn = PerceptualLoss().to(device)

    recognition_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for param in recognition_model.parameters():
        param.requires_grad_(False)

    print(f"Detected {len(base_dataset)} training pairs. Train: {len(train_dataset)}, Val: {len(val_dataset)}.")
    print(f"Loss weights: mask={mask_weight}, full={full_weight}, identity={identity_weight}, "
          f"global_id={global_identity_weight}, edge={edge_weight}, structure={structure_weight}, "
          f"perceptual={perceptual_weight}, adversarial={adversarial_weight}, tv={tv_weight}, "
          f"detail={detail_weight}, outside={outside_weight}, teacher_id={teacher_identity_weight}")
    print(f"Mouth/chin mask restriction: {restrict_mouth_chin_mask}")
    if refiner is not None:
        print(f"MTR-UNet residual delta scale: {refiner_delta_scale}")
    print(f"Discriminator starts at epoch {disc_start_epoch + 1}.")

    if refiner is None:
        model.train()
    else:
        refiner.train()
    best_val_loss = float("inf")
    sample_output_dir = (
        Path(sample_output_dir_override)
        if sample_output_dir_override
        else Path("outputs/training_samples_refiner" if refiner is not None else "outputs/training_samples")
    )
    metrics_log_path = sample_output_dir / "epoch_metrics.jsonl"
    train_step_log_path = sample_output_dir / "train_step_metrics.jsonl"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for epoch in range(epochs):
        display_epoch = epoch + 1 + epoch_offset
        display_total_epochs = epochs + epoch_offset
        running_loss = 0.0
        use_adv = epoch >= disc_start_epoch and adversarial_weight > 0.0
        cur_adv_weight = adversarial_weight if use_adv else 0.0
        running_metrics = {
            "loss": 0.0,
            "discriminator_loss": 0.0,
            "full_identity": 0.0,
            "outside_preservation": 0.0,
            "ssim": 0.0,
            "identity_similarity": 0.0,
            "teacher_identity": 0.0,
            "teacher_identity_similarity": 0.0,
        }
        running_batches = 0

        for i, (masked_inputs, original_targets, occlusion_masks) in enumerate(train_loader):
            masked_inputs = masked_inputs.to(device, non_blocking=True)
            original_targets = original_targets.to(device, non_blocking=True)
            occlusion_masks = occlusion_masks.to(device, non_blocking=True)
            d_loss_value = 0.0

            # --- Discriminator step ---
            if use_adv:
                with torch.no_grad():
                    fake_imgs, _ = _forward_reconstruction(model, masked_inputs, occlusion_masks, refiner)
                optimizer_d.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    real_pred = discriminator(_masked_discriminator_view(original_targets, original_targets, occlusion_masks))
                    fake_pred = discriminator(_masked_discriminator_view(fake_imgs.detach(), original_targets, occlusion_masks))
                    d_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
                        + F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
                    )
                    d_loss_value = float(d_loss.detach().item())
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                scaler.update()

            # --- Generator step ---
            optimizer_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss, train_metrics = _compute_reconstruction_losses(
                    model, recognition_model, masked_inputs, original_targets, occlusion_masks,
                    mask_weight, full_weight, identity_weight, global_identity_weight,
                    edge_weight, structure_weight,
                    perceptual_loss_fn=perceptual_loss_fn, perceptual_weight=perceptual_weight,
                    discriminator=discriminator if use_adv else None, adversarial_weight=cur_adv_weight,
                    refiner=refiner, tv_weight=tv_weight, detail_weight=detail_weight,
                    outside_weight=outside_weight,
                    teacher_identity_weight=teacher_identity_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            running_loss += loss.item()
            running_batches += 1
            running_metrics["loss"] += float(train_metrics["loss"])
            running_metrics["discriminator_loss"] += d_loss_value
            running_metrics["full_identity"] += float(train_metrics["full_identity"])
            running_metrics["outside_preservation"] += float(train_metrics["outside_preservation"])
            running_metrics["ssim"] += float(train_metrics["ssim"])
            running_metrics["identity_similarity"] += float(train_metrics["identity_similarity"])
            running_metrics["teacher_identity"] += float(train_metrics["teacher_identity"])
            running_metrics["teacher_identity_similarity"] += float(train_metrics["teacher_identity_similarity"])
            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                avg_metrics = {key: value / max(1, running_batches) for key, value in running_metrics.items()}
                current_lr = optimizer_g.param_groups[0]["lr"]
                print(
                    f"[Epoch {display_epoch}/{display_total_epochs}, Batch {i + 1}/{len(train_loader)}] "
                    f"G={avg_metrics['loss']:.4f} "
                    f"D={avg_metrics['discriminator_loss']:.4f} "
                    f"id_loss={avg_metrics['full_identity']:.4f} "
                    f"outside={avg_metrics['outside_preservation']:.4f} "
                    f"ssim={avg_metrics['ssim']:.4f} "
                    f"id_sim={avg_metrics['identity_similarity']:.4f} "
                    f"teacher_id={avg_metrics['teacher_identity']:.4f} "
                    f"teacher_sim={avg_metrics['teacher_identity_similarity']:.4f} "
                    f"lr={current_lr:.6g}"
                )
                with train_step_log_path.open("a", encoding="utf-8") as train_log:
                    train_log.write(
                        json.dumps(
                            {
                        "epoch": epoch + 1,
                        "display_epoch": display_epoch,
                        "batch": i + 1,
                                "batches": len(train_loader),
                                "learning_rate": current_lr,
                                "adversarial_active": use_adv,
                                **avg_metrics,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                running_loss = 0.0
                running_batches = 0
                running_metrics = {key: 0.0 for key in running_metrics}

        val_metrics = _evaluate_epoch(
            model, recognition_model, val_loader, device,
            mask_weight, full_weight, identity_weight, global_identity_weight,
            edge_weight, structure_weight, perceptual_loss_fn, perceptual_weight,
            refiner=refiner, tv_weight=tv_weight, detail_weight=detail_weight,
            outside_weight=outside_weight,
            teacher_identity_weight=teacher_identity_weight,
        )
        print(
            f"[Epoch {display_epoch}/{display_total_epochs}] "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_l1={val_metrics['masked_l1']:.4f} "
            f"val_outside={val_metrics['outside_preservation']:.4f} "
            f"val_edge={val_metrics['edge_loss']:.4f} "
            f"val_id={val_metrics['full_identity']:.4f} "
            f"val_id_sim={val_metrics['identity_similarity']:.4f} "
            f"val_teacher_id={val_metrics['teacher_identity']:.4f} "
            f"val_teacher_sim={val_metrics['teacher_identity_similarity']:.4f} "
            f"val_percep={val_metrics['perceptual']:.4f} "
            f"val_tv={val_metrics['tv']:.4f} "
            f"val_detail={val_metrics['detail']:.4f}"
        )
        with metrics_log_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "display_epoch": display_epoch,
                        "image_size": image_size,
                        "restrict_mouth_chin_mask": restrict_mouth_chin_mask,
                        **val_metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        weights_dir = Path(weights_output_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        if refiner is None:
            epoch_checkpoint = weights_dir / f"degan_model_epoch_{display_epoch:03d}.pth"
            torch.save(model.state_dict(), epoch_checkpoint)
            torch.save(discriminator.state_dict(), weights_dir / f"degan_disc_epoch_{display_epoch:03d}.pth")
        else:
            epoch_checkpoint = weights_dir / f"mtr_unet_epoch_{display_epoch:03d}.pth"
            torch.save(refiner.state_dict(), epoch_checkpoint)
            torch.save(discriminator.state_dict(), weights_dir / f"mtr_unet_disc_epoch_{display_epoch:03d}.pth")
        print(f"Saved epoch checkpoint: {epoch_checkpoint}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if refiner is None:
                torch.save(model.state_dict(), weights_dir / "degan_model_best.pth")
            else:
                torch.save(refiner.state_dict(), weights_dir / "mtr_unet_best.pth")
            print(f"Saved improved checkpoint.")

        # Save visual samples every epoch for tracking reconstruction quality
        _save_epoch_samples(
            model,
            val_loader,
            device,
            display_epoch - 1,
            sample_output_dir,
            refiner=refiner,
        )

    weights_dir = Path(weights_output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / "degan_model_final.pth"
    if refiner is None:
        torch.save(model.state_dict(), save_path)
        torch.save(discriminator.state_dict(), weights_dir / "degan_disc_final.pth")
    else:
        save_path = weights_dir / "mtr_unet_final.pth"
        torch.save(refiner.state_dict(), save_path)
        torch.save(discriminator.state_dict(), weights_dir / "mtr_unet_disc_final.pth")
    print(f"Training complete. Weights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DEGAN reconstruction")
    parser.add_argument("--masked", type=str, required=True)
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mask-weight", type=float, default=4.0)
    parser.add_argument(
        "--full-weight",
        type=float,
        default=0.0,
        help="Optional full-image reconstruction term. Keep 0.0 to train reconstruction only inside the mask.",
    )
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--global-identity-weight", type=float, default=1.5)
    parser.add_argument("--edge-weight", type=float, default=1.15)
    parser.add_argument("--structure-weight", type=float, default=0.8)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--detail-weight", type=float, default=0.0)
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=1.0,
        help="Penalty for raw generator changes outside the occlusion mask.",
    )
    parser.add_argument(
        "--teacher-identity-weight",
        type=float,
        default=0.0,
        help="Preserve recognition embedding similarity to the frozen DEGAN/backend blend while training a refiner.",
    )
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--disc-start-epoch", type=int, default=5)
    parser.add_argument("--mask-threshold", type=float, default=0.08)
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Square reconstruction training size. Use 224 for higher-resolution retraining; must be divisible by 8.",
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--synthetic-probability", type=float, default=0.80)
    parser.add_argument("--extra-masked", type=str, nargs="*", default=[])
    parser.add_argument("--extra-original", type=str, nargs="*", default=[])
    parser.add_argument("--max-extra-originals", type=int, default=1500)
    parser.add_argument(
        "--use-refiner", "--use_refiner",
        nargs="?", const=True, default=False, type=_str_to_bool,
        help="Train MTR-UNet after a frozen DEGAN. Accepts true/false or can be used as a flag.",
    )
    parser.add_argument("--refiner-resume", type=str, default=None, help="Optional MTR-UNet checkpoint to resume.")
    parser.add_argument("--refiner-delta-scale", type=float, default=0.75, help="Maximum residual strength for MTR-UNet.")
    parser.add_argument(
        "--weights-output-dir",
        type=str,
        default="weights",
        help="Directory for checkpoints written by this run.",
    )
    parser.add_argument(
        "--sample-output-dir",
        type=str,
        default=None,
        help="Optional directory for visual samples and training metric logs.",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Offset used only for naming/logging resumed continuation epochs.",
    )
    parser.add_argument(
        "--filter-extra-originals", "--filter_extra_originals",
        nargs="?", const=True, default=True, type=_str_to_bool,
        help="Brightness-filter extra-original datasets before sampling. Disable for already cleaned/aligned datasets.",
    )
    parser.add_argument(
        "--restrict-mouth-chin-mask",
        "--restrict_mouth_chin_mask",
        nargs="?",
        const=True,
        default=MOUTH_CHIN_MASK_RESTRICT_ENABLED_DEFAULT,
        type=_str_to_bool,
        help="Restrict training masks to the central mouth/chin band to exclude neck, clothing, and background.",
    )
    parser.add_argument("--synthetic-mode", type=str, default="mixed",
        choices=["mixed", "lower-mask", "sunglasses", "opaque-panel", "side-occlusion", "hand", "scarf"])
    args = parser.parse_args()

    train(
        args.masked, args.original, args.epochs, args.batch, args.lr, args.workers,
        args.mask_weight, args.full_weight, args.identity_weight, args.global_identity_weight,
        args.edge_weight, args.structure_weight, args.mask_threshold, args.val_split,
        args.resume, args.synthetic_probability, args.synthetic_mode,
        args.extra_masked, args.extra_original,
        None if args.max_extra_originals is not None and args.max_extra_originals < 0 else args.max_extra_originals,
        args.perceptual_weight, args.adversarial_weight, args.tv_weight, args.detail_weight,
        args.outside_weight, args.teacher_identity_weight,
        args.disc_lr, args.disc_start_epoch, args.use_refiner, args.refiner_resume,
        args.refiner_delta_scale, args.filter_extra_originals, args.image_size,
        args.restrict_mouth_chin_mask,
        args.weights_output_dir,
        args.sample_output_dir,
        args.epoch_offset,
    )
