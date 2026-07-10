"""Debug visualization script for the DEGAN reconstruction pipeline.

Generates a side-by-side grid at each pipeline stage so you can visually
verify:
  1. Blending/compositing (are white box artifacts gone?)
  2. Normalization consistency (are tensor ranges correct?)
  3. Skip connection quality (is the raw model output clean?)
  4. Mask estimation accuracy (does the mask cover the occluder?)

Usage:
    python backend/debug_pipeline.py --image <path_to_occluded_face>
    python backend/debug_pipeline.py --dataset database/processed_lfw --limit 5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from main import (
        _decode_upload,
        _prepare_face_analysis,
        _reconstruct_face,
        _blend_reconstructed_face,
        _tensor_to_rgb_image,
        _rgb_image_to_face_tensor,
        _occlusion_mask_to_tensor,
        degan_model,
        device,
    )
except ModuleNotFoundError:
    from backend.main import (
        _decode_upload,
        _prepare_face_analysis,
        _reconstruct_face,
        _blend_reconstructed_face,
        _tensor_to_rgb_image,
        _rgb_image_to_face_tensor,
        _occlusion_mask_to_tensor,
        degan_model,
        device,
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a [0,1] float mask to a red-channel heatmap for visualization."""
    mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a label bar above an image."""
    h, w = img.shape[:2]
    bar = np.zeros((22, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack([bar, img])


def debug_single_image(img_bgr: np.ndarray, output_path: Path, name: str = "sample", use_refiner: bool = False):
    """Run the full pipeline on one image and save a diagnostic grid."""
    analysis = _prepare_face_analysis(img_bgr)
    recon_tensor = analysis["reconstruction_tensor"]
    recon_rgb = analysis["reconstruction_rgb"]
    mask = analysis["occlusion_mask"]
    is_occluded = analysis["is_occluded"]
    region = analysis["occlusion_region"]
    ratio = analysis["occlusion_ratio"]

    # --- Stage 1: Input tensor range check ---
    t = recon_tensor
    print(f"[{name}] Input tensor range: [{t.min():.3f}, {t.max():.3f}]  "
          f"occluded={is_occluded}  region={region}  ratio={ratio:.3f}")

    # --- Stage 2: Raw model output (before blending) ---
    raw_output, raw_rgb = _reconstruct_face(recon_tensor, mask)
    print(f"[{name}] Raw model output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")

    # Check for white-box remnants in raw output
    if raw_output.max() > 0.95:
        bright_pixels = (raw_output > 0.9).float().mean().item()
        print(f"  WARNING: {bright_pixels*100:.1f}% of output pixels > 0.9 (possible mask leak)")

    # --- Stage 3: Blended output ---
    if is_occluded:
        blended_tensor, blended_rgb = _blend_reconstructed_face(
            recon_tensor,
            mask,
            enhance_patch=not analysis["detected_face"],
            use_refiner=use_refiner,
        )
    else:
        blended_rgb = recon_rgb

    # --- Build visualization grid ---
    panels = []

    # Row 1: Input / Mask / Raw output / Blended output
    panels.append(_label(recon_rgb, "Input (occluded)"))
    panels.append(_label(_colorize_mask(mask), f"Mask ({ratio:.2f})"))
    panels.append(_label(raw_rgb, "Raw model output"))
    panels.append(_label(blended_rgb, "Blended output"))

    # Row 2: Mask overlay on input / difference map
    mask_overlay = recon_rgb.copy().astype(np.float32)
    red = np.zeros_like(mask_overlay)
    red[:, :, 0] = 255.0
    alpha = np.clip(mask[..., None], 0, 1) * 0.5
    mask_overlay = mask_overlay * (1 - alpha) + red * alpha
    mask_overlay = np.clip(mask_overlay, 0, 255).astype(np.uint8)
    panels.append(_label(mask_overlay, "Mask on input"))

    # Difference: raw output vs input (highlights what the model changed)
    diff = np.abs(raw_rgb.astype(np.float32) - recon_rgb.astype(np.float32))
    diff_vis = np.clip(diff * 3, 0, 255).astype(np.uint8)  # amplify for visibility
    panels.append(_label(diff_vis, "Model change (3x)"))

    # Difference: blended vs input
    diff2 = np.abs(blended_rgb.astype(np.float32) - recon_rgb.astype(np.float32))
    diff2_vis = np.clip(diff2 * 3, 0, 255).astype(np.uint8)
    panels.append(_label(diff2_vis, "Blend change (3x)"))

    # White pixel check: highlight pixels > 240 in raw output
    bright_mask = (raw_rgb.astype(np.float32).mean(axis=2) > 240).astype(np.uint8) * 255
    bright_vis = np.stack([bright_mask, np.zeros_like(bright_mask), np.zeros_like(bright_mask)], axis=2)
    panels.append(_label(bright_vis, "Bright pixels (>240)"))

    # Arrange into 2 rows of 4
    row1 = np.hstack(panels[:4])
    row2 = np.hstack(panels[4:])
    grid = np.vstack([row1, row2])

    out_file = output_path / f"debug_{name}.png"
    cv2.imwrite(str(out_file), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  -> Saved: {out_file}")
    return grid


def debug_dataset(dataset_path: Path, output_path: Path, limit: int = 5, use_refiner: bool = False):
    """Run debug visualization on a few samples from a masked dataset."""
    output_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            name = f"{person_dir.name}_{img_path.stem}"
            debug_single_image(img_bgr, output_path, name, use_refiner=use_refiner)
            count += 1
            if count >= limit:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug DEGAN reconstruction pipeline visually")
    parser.add_argument("--image", type=str, help="Path to a single occluded face image")
    parser.add_argument("--dataset", type=str, help="Path to masked dataset folder")
    parser.add_argument("--limit", type=int, default=5, help="Max samples from dataset")
    parser.add_argument("--output", type=str, default="outputs/debug_pipeline", help="Output directory")
    parser.add_argument(
        "--use-refiner", "--use_refiner",
        nargs="?", const=True, default=False, type=_str_to_bool,
        help="Visualize DEGAN + MTR-UNet refined output. Accepts true/false or can be used as a flag.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.image:
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print(f"Error: Cannot read {args.image}")
        else:
            debug_single_image(img_bgr, out_path, Path(args.image).stem, use_refiner=args.use_refiner)
    elif args.dataset:
        debug_dataset(Path(args.dataset), out_path, args.limit, use_refiner=args.use_refiner)
    else:
        # Default: use processed_lfw
        default_ds = Path("database/processed_lfw")
        if default_ds.exists():
            debug_dataset(default_ds, out_path, args.limit, use_refiner=args.use_refiner)
        else:
            print("Provide --image or --dataset. Run from project root.")
