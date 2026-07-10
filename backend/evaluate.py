import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

try:
    from main import (
        IMAGE_EXTENSIONS,
        _blend_reconstructed_face,
        _prepare_face_analysis,
        _predict_identity,
        _recognition_embedding,
        _recognition_probe_from_reconstruction,
        _reconstruct_face,
        _rgb_image_to_face_tensor,
        gallery_dir,
        refresh_gallery_index,
    )
except ModuleNotFoundError:
    from backend.main import (
        IMAGE_EXTENSIONS,
        _blend_reconstructed_face,
        _prepare_face_analysis,
        _predict_identity,
        _recognition_embedding,
        _recognition_probe_from_reconstruction,
        _reconstruct_face,
        _rgb_image_to_face_tensor,
        gallery_dir,
        refresh_gallery_index,
    )

try:
    import lpips
except Exception:
    lpips = None


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _paired_occlusion_mask(masked_rgb: np.ndarray, original_rgb: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    diff = cv2.absdiff(masked_rgb, original_rgb).mean(axis=2)
    mask = np.where(diff > threshold, 255, 0).astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(mask.astype("float32") / 255.0, (7, 7), 0)


def _mouth_chin_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    roi = np.zeros((h, w), dtype=np.float32)
    y1 = max(0, int(round(0.30 * h)))
    y2 = min(h, int(round(0.84 * h)))
    x1 = max(0, int(round(0.10 * w)))
    x2 = min(w, int(round(0.90 * w)))
    roi[y1:y2, x1:x2] = 1.0
    return roi


def _mask_outside_mouth_chin_fraction(mask: np.ndarray) -> float:
    hard = mask > 0.5
    area = int(np.count_nonzero(hard))
    if area == 0:
        return 0.0
    roi = _mouth_chin_roi_mask(mask.shape[:2]) > 0.5
    return float(np.count_nonzero(hard & ~roi) / area)


def _masked_mse(prediction_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> float:
    mask_3 = np.clip(mask[..., None], 0.0, 1.0).astype("float32")
    weighted_error = ((prediction_rgb - target_rgb) ** 2) * mask_3
    denom = float(mask_3.sum() * prediction_rgb.shape[2])
    if denom <= 1e-6:
        return float(((prediction_rgb - target_rgb) ** 2).mean())
    return float(weighted_error.sum() / denom)


def _lpips_tensor(rgb_float: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.clip(rgb_float, 0, 255).astype("float32")).permute(2, 0, 1).unsqueeze(0)
    return tensor / 127.5 - 1.0


def _arcface_cosine_rgb(prediction_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    pred_tensor = _rgb_image_to_face_tensor(np.clip(prediction_rgb, 0, 255).astype(np.uint8))
    target_tensor = _rgb_image_to_face_tensor(np.clip(target_rgb, 0, 255).astype(np.uint8))
    pred_embedding = _recognition_embedding(pred_tensor)
    target_embedding = _recognition_embedding(target_tensor)
    return float(torch.nn.functional.cosine_similarity(pred_embedding, target_embedding, dim=1).item())


def _label_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    panel = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = panel.shape[:2]
    header = np.full((24, w, 3), 245, dtype=np.uint8)
    cv2.putText(header, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def _save_reconstruction_comparison(
    output_dir: Path,
    index: int,
    original_rgb: np.ndarray,
    masked_rgb: np.ndarray,
    raw_rgb: np.ndarray,
    region_only_rgb: np.ndarray,
    blended_rgb: np.ndarray,
    difference_rgb: np.ndarray,
    stem: str,
    display_size: int = 180,
) -> None:
    comparison_dir = output_dir / "reconstruction_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    def resized_panel(rgb: np.ndarray) -> np.ndarray:
        return cv2.resize(
            np.clip(rgb, 0, 255).astype(np.uint8),
            (display_size, display_size),
            interpolation=cv2.INTER_NEAREST,
        )

    panels = [
        _label_panel(resized_panel(original_rgb), "Original"),
        _label_panel(resized_panel(masked_rgb), "Masked"),
        _label_panel(resized_panel(raw_rgb), "Raw GAN Output"),
        _label_panel(resized_panel(region_only_rgb), "Region Only"),
        _label_panel(resized_panel(blended_rgb), "Final Blended"),
        _label_panel(resized_panel(difference_rgb), "Difference Map"),
    ]
    grid = np.hstack(panels)
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    out_path = comparison_dir / f"{index:03d}_{safe_stem}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def evaluate_dataset(
    masked_root: Path,
    original_root: Path,
    limit: int | None = None,
    output_dir: Path | None = None,
    use_refiner: bool = False,
    compute_lpips: bool = False,
    comparison_limit: int = 8,
) -> dict:
    refresh_gallery_index()

    lpips_model = None
    if compute_lpips and lpips is not None:
        try:
            lpips_model = lpips.LPIPS(net="alex").eval()
        except Exception:
            lpips_model = None

    total = 0
    detected_faces = 0
    final_mse_sum = 0.0
    final_psnr_sum = 0.0
    final_occluded_region_mse_sum = 0.0
    final_occluded_region_psnr_sum = 0.0
    final_ssim_sum = 0.0
    degan_ssim_sum = 0.0
    refined_ssim_sum = 0.0
    arcface_cosine_sum = 0.0
    raw_arcface_cosine_sum = 0.0
    raw_outside_mse_sum = 0.0
    raw_outside_max_delta_sum = 0.0
    mask_outside_mouth_chin_fraction_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    total_time = 0.0

    y_true = []
    y_pred = []
    y_scores = []
    y_match = []
    y_pred_occluded = []
    y_pred_degan = []
    y_pred_refined = []

    for person_dir in sorted(masked_root.iterdir()):
        if not person_dir.is_dir():
            continue

        image_paths = sorted(
            path for path in person_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            rel_path = image_path.relative_to(masked_root)
            original_path = original_root / rel_path
            if not original_path.exists():
                continue

            masked_bgr = cv2.imread(str(image_path))
            original_bgr = cv2.imread(str(original_path))
            if masked_bgr is None or original_bgr is None:
                continue

            start = time.perf_counter()
            analysis = _prepare_face_analysis(masked_bgr)
            face_tensor = analysis["face_tensor"]
            occluded_predicted_name, occluded_similarity, _ = _predict_identity(face_tensor)

            if analysis["is_occluded"]:
                _, raw_rgb = _reconstruct_face(
                    analysis["reconstruction_tensor"],
                    analysis["occlusion_mask"],
                )
                degan_tensor, degan_rgb = _blend_reconstructed_face(
                    analysis["reconstruction_tensor"],
                    analysis["occlusion_mask"],
                    enhance_patch=not analysis["detected_face"],
                    use_refiner=False,
                )
                final_tensor, final_rgb = _blend_reconstructed_face(
                    analysis["reconstruction_tensor"],
                    analysis["occlusion_mask"],
                    enhance_patch=not analysis["detected_face"],
                    use_refiner=use_refiner,
                )
                raw_rgb = raw_rgb.astype("float32")
                degan_rgb = degan_rgb.astype("float32")
                final_rgb = final_rgb.astype("float32")
            else:
                raw_rgb = analysis["reconstruction_rgb"].astype("float32")
                degan_rgb = analysis["reconstruction_rgb"].astype("float32")
                final_rgb = analysis["reconstruction_rgb"].astype("float32")
                degan_tensor = analysis["reconstruction_tensor"]
                final_tensor = analysis["reconstruction_tensor"]

            # Reconstructed-face embeddings currently drift more than the
            # original aligned input embeddings. Evaluate the deployed
            # recognition path using the input face while keeping reconstructed
            # predictions as diagnostics.
            predicted_name = occluded_predicted_name
            similarity = occluded_similarity
            true_name = person_dir.name

            degan_probe_tensor, _, _ = _recognition_probe_from_reconstruction(
                np.clip(degan_rgb, 0, 255).astype(np.uint8),
                degan_tensor,
                np.clip(degan_rgb, 0, 255).astype(np.uint8),
            )
            refined_probe_tensor, _, _ = _recognition_probe_from_reconstruction(
                np.clip(final_rgb, 0, 255).astype(np.uint8),
                final_tensor,
                np.clip(final_rgb, 0, 255).astype(np.uint8),
            )
            degan_predicted_name, _, _ = _predict_identity(degan_probe_tensor)
            refined_predicted_name, _, _ = _predict_identity(refined_probe_tensor)
            elapsed = time.perf_counter() - start
            
            y_true.append(true_name)
            y_pred.append(predicted_name if predicted_name else "Unknown")
            y_scores.append(similarity)
            y_match.append(1 if true_name == predicted_name else 0)
            y_pred_occluded.append(occluded_predicted_name if occluded_predicted_name else "Unknown")
            y_pred_degan.append(degan_predicted_name if degan_predicted_name else "Unknown")
            y_pred_refined.append(refined_predicted_name if refined_predicted_name else "Unknown")

            metric_h, metric_w = final_rgb.shape[:2]
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            original_rgb = cv2.resize(original_rgb, (metric_w, metric_h), interpolation=cv2.INTER_AREA).astype("float32")
            masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
            masked_rgb = cv2.resize(masked_rgb, (metric_w, metric_h), interpolation=cv2.INTER_AREA).astype("float32")
            
            occlusion_mask = _paired_occlusion_mask(masked_rgb, original_rgb)
            mse_pixels = float(((final_rgb - original_rgb) ** 2).mean())
            psnr = _psnr_from_mse(mse_pixels)
            occluded_mse = _masked_mse(final_rgb, original_rgb, occlusion_mask)
            occluded_psnr = _psnr_from_mse(occluded_mse)
            inverse_mask = 1.0 - np.clip(occlusion_mask, 0.0, 1.0)
            raw_outside_mse = _masked_mse(raw_rgb, original_rgb, inverse_mask)
            outside_pixels = inverse_mask > 0.5
            raw_outside_max_delta = 0.0
            if np.any(outside_pixels):
                raw_outside_max_delta = float(
                    np.abs(raw_rgb.astype(np.float32) - original_rgb.astype(np.float32))[outside_pixels].max()
                )
            mask_outside_mouth_chin_fraction = _mask_outside_mouth_chin_fraction(occlusion_mask)
            
            # SSIM calculation
            img_rec_uint8 = np.clip(final_rgb, 0, 255).astype(np.uint8)
            img_degan_uint8 = np.clip(degan_rgb, 0, 255).astype(np.uint8)
            img_orig_uint8 = np.clip(original_rgb, 0, 255).astype(np.uint8)
            current_ssim = ssim(img_orig_uint8, img_rec_uint8, multichannel=True, channel_axis=2, data_range=255)
            degan_ssim = ssim(img_orig_uint8, img_degan_uint8, multichannel=True, channel_axis=2, data_range=255)
            arcface_cosine = _arcface_cosine_rgb(final_rgb, original_rgb)
            raw_arcface_cosine = _arcface_cosine_rgb(raw_rgb, original_rgb)

            if output_dir is not None and total < comparison_limit:
                mask_3 = np.clip(occlusion_mask[..., None], 0.0, 1.0).astype(np.float32)
                region_only_rgb = np.clip(masked_rgb * (1.0 - mask_3) + raw_rgb * mask_3, 0, 255).astype(np.uint8)
                difference_rgb = np.clip(np.abs(final_rgb - original_rgb) * 3.0, 0, 255).astype(np.uint8)
                _save_reconstruction_comparison(
                    output_dir,
                    total + 1,
                    img_orig_uint8,
                    np.clip(masked_rgb, 0, 255).astype(np.uint8),
                    np.clip(raw_rgb, 0, 255).astype(np.uint8),
                    region_only_rgb,
                    img_rec_uint8,
                    difference_rgb,
                    image_path.stem,
                )

            if lpips_model is not None:
                with torch.no_grad():
                    lpips_value = float(lpips_model(_lpips_tensor(final_rgb), _lpips_tensor(original_rgb)).item())
                lpips_sum += lpips_value
                lpips_count += 1

            total += 1
            detected_faces += int(analysis["detected_face"])
            final_mse_sum += mse_pixels
            final_psnr_sum += psnr
            final_occluded_region_mse_sum += occluded_mse
            final_occluded_region_psnr_sum += occluded_psnr
            final_ssim_sum += current_ssim
            degan_ssim_sum += degan_ssim
            refined_ssim_sum += current_ssim
            arcface_cosine_sum += arcface_cosine
            raw_arcface_cosine_sum += raw_arcface_cosine
            raw_outside_mse_sum += raw_outside_mse
            raw_outside_max_delta_sum += raw_outside_max_delta
            mask_outside_mouth_chin_fraction_sum += mask_outside_mouth_chin_fraction
            total_time += elapsed

            if limit is not None and total >= limit:
                break

        if limit is not None and total >= limit:
            break

    if total == 0:
        raise RuntimeError("No evaluation samples were found.")

    # Calculate Sklearn Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate ROC/FAR/TAR only when both positive and negative matches exist.
    # Small sanity samples can legitimately contain all-correct or all-incorrect
    # predictions; sklearn returns NaN in that case, which is not useful JSON.
    fpr = tpr = thresholds = None
    roc_auc = None
    if len(set(y_match)) == 2:
        fpr, tpr, thresholds = roc_curve(y_match, y_scores)
        roc_auc = auc(fpr, tpr)
    
    results = {
        "samples_evaluated": total,
        "use_refiner": use_refiner,
        "reconstruction_strategy": "no-reference-strong-detail-rescue",
        "recognition_strategy": "input-face",
        "accuracy": acc,
        "occluded_input_accuracy": accuracy_score(y_true, y_pred_occluded),
        "degan_reconstructed_accuracy": accuracy_score(y_true, y_pred_degan),
        "refined_reconstructed_accuracy": accuracy_score(y_true, y_pred_refined),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "detected_face_rate": detected_faces / total,
        "avg_reconstruction_mse": final_mse_sum / total,
        "avg_reconstruction_psnr": final_psnr_sum / total,
        "avg_reconstruction_ssim": final_ssim_sum / total,
        "avg_degan_ssim": degan_ssim_sum / total,
        "avg_refined_ssim": refined_ssim_sum / total,
        "avg_occluded_region_mse": final_occluded_region_mse_sum / total,
        "avg_occluded_region_psnr": final_occluded_region_psnr_sum / total,
        "avg_arcface_cosine_similarity": arcface_cosine_sum / total,
        "avg_raw_identity_similarity": raw_arcface_cosine_sum / total,
        "avg_blended_identity_similarity": arcface_cosine_sum / total,
        "avg_raw_outside_mask_mse": raw_outside_mse_sum / total,
        "avg_raw_outside_mask_max_delta": raw_outside_max_delta_sum / total,
        "avg_mask_outside_mouth_chin_fraction": mask_outside_mouth_chin_fraction_sum / total,
        "avg_lpips": (lpips_sum / lpips_count) if lpips_count > 0 else None,
        "avg_fps": total / total_time if total_time > 0 else 0.0,
    }
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC
        if fpr is not None and tpr is not None and roc_auc is not None:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Acceptance Rate (FAR)')
            plt.ylabel('True Acceptance Rate (TAR)')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(output_dir / 'roc_curve.png')
            plt.close()
        
        # Plot Confusion Matrix (limited to top 20 classes if many)
        cm = confusion_matrix(y_true, y_pred)
        if len(set(y_true)) <= 20:
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reconstruction and recognition performance.")
    parser.add_argument("--masked", type=str, default=str(Path("database") / "processed_lfw"), help="Masked dataset root")
    parser.add_argument("--original", type=str, default=str(gallery_dir), help="Original dataset root")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for faster evaluation")
    parser.add_argument("--output", type=str, default=str(Path("outputs") / "evaluation_metrics.json"), help="Path to save metrics JSON")
    parser.add_argument(
        "--use-refiner", "--use_refiner",
        nargs="?", const=True, default=False, type=_str_to_bool,
        help="Evaluate DEGAN + MTR-UNet refined output. Accepts true/false or can be used as a flag.",
    )
    parser.add_argument("--compute-lpips", action="store_true", help="Calculate LPIPS if the package/checkpoint is available.")
    parser.add_argument(
        "--comparison-limit",
        type=int,
        default=8,
        help="Number of Original/Masked/Raw/Blended comparison images to save in the output directory.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    
    results = evaluate_dataset(
        Path(args.masked),
        Path(args.original),
        args.limit,
        output_dir,
        args.use_refiner,
        args.compute_lpips,
        args.comparison_limit,
    )
    output_path.write_text(json.dumps(results, indent=2))

    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"saved_to: {output_path.resolve()}")
