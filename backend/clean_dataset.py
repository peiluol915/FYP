from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

try:
    from dataset_split import CleanSample, class_distribution, identity_level_split, split_summary
    from face_align import align_face, normalized_augmentation_regions
    from quality_filter import QualityConfig, check_basic_quality, check_pose_quality, is_image_path, load_rgb_image
except ModuleNotFoundError:
    from backend.dataset_split import CleanSample, class_distribution, identity_level_split, split_summary
    from backend.face_align import align_face, normalized_augmentation_regions
    from backend.quality_filter import QualityConfig, check_basic_quality, check_pose_quality, is_image_path, load_rgb_image


def collect_images(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*") if is_image_path(path))


def load_identity_file(identity_file: Path | None) -> dict[str, str]:
    if identity_file is None:
        return {}
    mapping: dict[str, str] = {}
    with identity_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.lower().startswith("image"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                mapping[parts[0].replace("\\", "/")] = parts[1]
    return mapping


def infer_identity(path: Path, input_root: Path, identity_mapping: dict[str, str]) -> str:
    rel = path.relative_to(input_root).as_posix()
    if rel in identity_mapping:
        return identity_mapping[rel]
    if path.name in identity_mapping:
        return identity_mapping[path.name]
    parent = path.parent
    if parent != input_root:
        return parent.name
    return "unknown_identity"


def safe_output_name(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root)
    if len(rel.parts) > 1:
        return rel.name
    return path.name


def detect_single_face(
    mtcnn: MTCNN,
    image_rgb: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str | None]:
    boxes, probs, landmarks = mtcnn.detect(image_rgb, landmarks=True)
    if boxes is None or probs is None or landmarks is None or len(boxes) == 0:
        return None, None, None, "no_face"
    valid_indices = [idx for idx, prob in enumerate(probs) if prob is not None and not np.isnan(prob)]
    if len(valid_indices) != 1 or len(boxes) != 1:
        return None, None, None, "multiple_faces"
    confidence = float(probs[0])
    if confidence < confidence_threshold:
        return None, None, confidence, "low_detection_confidence"
    return boxes[0], landmarks[0], confidence, None


def save_rgb_jpg(image_rgb: np.ndarray, output_path: Path, quality: int = 95) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def process_dataset(
    input_root: Path,
    output_root: Path,
    image_size: int,
    identity_file: Path | None,
    confidence_threshold: float,
    quality_config: QualityConfig,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    limit: int | None = None,
    device_name: str | None = None,
) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    staging_root = output_root / "_staging_clean"

    if staging_root.exists():
        shutil.rmtree(staging_root)
    for split in ("train", "validation", "test"):
        split_path = output_root / split
        if split_path.exists():
            shutil.rmtree(split_path)
    output_root.mkdir(parents=True, exist_ok=True)

    identity_mapping = load_identity_file(identity_file)
    image_paths = collect_images(input_root)
    if limit is not None:
        image_paths = image_paths[:limit]

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    removal_counts: Counter[str] = Counter()
    clean_samples: list[CleanSample] = []
    processed = 0

    for idx, image_path in enumerate(image_paths, start=1):
        processed += 1
        try:
            image_rgb = load_rgb_image(image_path)
        except Exception:
            removal_counts["corrupted"] += 1
            continue

        quality = check_basic_quality(image_rgb, quality_config)
        if not quality.keep:
            for reason in quality.reasons:
                removal_counts[reason] += 1
            continue

        box, landmarks, confidence, face_error = detect_single_face(mtcnn, image_rgb, confidence_threshold)
        if face_error is not None:
            removal_counts[face_error] += 1
            continue

        pose = check_pose_quality(landmarks, quality_config)
        if not pose.keep:
            for reason in pose.reasons:
                removal_counts[reason] += 1
            continue

        try:
            aligned_rgb = align_face(image_rgb, box, landmarks, output_size=image_size)
        except Exception:
            removal_counts["alignment_failed"] += 1
            continue

        aligned_quality = check_basic_quality(aligned_rgb, quality_config)
        if not aligned_quality.keep:
            for reason in aligned_quality.reasons:
                removal_counts[f"aligned_{reason}"] += 1
            continue

        identity = infer_identity(image_path, input_root, identity_mapping)
        output_name = safe_output_name(image_path, input_root)
        relative_output_path = f"{identity}/{output_name}"
        staging_path = staging_root / relative_output_path
        save_rgb_jpg(aligned_rgb, staging_path)

        metrics = {
            **quality.metrics,
            **pose.metrics,
            "detection_confidence": float(confidence),
            "aligned_laplacian_variance": float(aligned_quality.metrics["laplacian_variance"]),
            "normalized_min": 0.0,
            "normalized_max": 1.0,
        }
        clean_samples.append(
            CleanSample(
                staging_path=staging_path,
                identity=identity,
                original_path=str(image_path),
                relative_output_path=relative_output_path,
                metrics=metrics,
                augmentation_regions=normalized_augmentation_regions(image_size),
            )
        )

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)} images. Kept {len(clean_samples)}.")

    splits = identity_level_split(clean_samples, train_ratio, validation_ratio, test_ratio, seed)
    mapping: dict[str, str] = {}
    augmentation_metadata: dict[str, dict] = {}

    for split, samples in splits.items():
        for sample in samples:
            final_path = output_root / split / sample.relative_output_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sample.staging_path), str(final_path))
            rel_final = final_path.relative_to(output_root).as_posix()
            mapping[rel_final] = sample.identity
            augmentation_metadata[rel_final] = {
                "identity": sample.identity,
                "regions": sample.augmentation_regions,
                "image_size": [image_size, image_size],
                "pixel_range": [0.0, 1.0],
            }

    if staging_root.exists():
        shutil.rmtree(staging_root)

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_size": image_size,
        "device": str(device),
        "total_images_before_cleaning": processed,
        "final_image_count": len(clean_samples),
        "removed_images_count": processed - len(clean_samples),
        "removed_by_reason": dict(sorted(removal_counts.items())),
        "blur_removed_count": removal_counts.get("blurry", 0) + removal_counts.get("aligned_blurry", 0),
        "pose_removed_count": removal_counts.get("extreme_yaw", 0),
        "split_summary": split_summary(splits),
        "class_distribution": class_distribution(clean_samples),
        "notes": [
            "Output images are aligned RGB JPG files resized to the requested size.",
            "Training loaders should convert saved uint8 RGB images to float tensors in [0,1] or normalize as needed.",
            "Occlusion metadata gives candidate regions only; no synthetic occlusion is applied here.",
            "Flat datasets need --identity-file for meaningful identity-level splitting.",
        ],
    }

    (output_root / "identity_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (output_root / "augmentation_metadata.json").write_text(json.dumps(augmentation_metadata, indent=2), encoding="utf-8")
    (output_root / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and preprocess face datasets for occluded face recognition.")
    parser.add_argument("--input", required=True, type=str, help="Raw dataset root.")
    parser.add_argument("--output", default="dataset_cleaned", type=str, help="Output dataset root.")
    parser.add_argument("--identity-file", type=str, default=None, help="Optional CelebA-style image-to-identity file.")
    parser.add_argument("--image-size", type=int, default=128, choices=[128, 224], help="Final aligned image size.")
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--blur-threshold", type=float, default=45.0)
    parser.add_argument("--min-width", type=int, default=80)
    parser.add_argument("--min-height", type=int, default=80)
    parser.add_argument("--dark-threshold", type=float, default=35.0)
    parser.add_argument("--overexposed-threshold", type=float, default=225.0)
    parser.add_argument("--yaw-threshold", type=float, default=45.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or omitted for auto.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = QualityConfig(
        min_width=args.min_width,
        min_height=args.min_height,
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        overexposed_threshold=args.overexposed_threshold,
        yaw_threshold=args.yaw_threshold,
    )
    report = process_dataset(
        input_root=Path(args.input),
        output_root=Path(args.output),
        image_size=args.image_size,
        identity_file=Path(args.identity_file) if args.identity_file else None,
        confidence_threshold=args.confidence_threshold,
        quality_config=config,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        limit=args.limit,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
