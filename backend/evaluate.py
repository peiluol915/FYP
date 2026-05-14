import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

try:
    from main import (
        IMAGE_EXTENSIONS,
        _blend_reconstructed_face,
        _prepare_face_analysis,
        _predict_identity,
        _reconstructed_region_only_face,
        _tensor_to_rgb_image,
        gallery_dir,
        refresh_gallery_index,
    )
except ModuleNotFoundError:
    from backend.main import (
        IMAGE_EXTENSIONS,
        _blend_reconstructed_face,
        _prepare_face_analysis,
        _predict_identity,
        _reconstructed_region_only_face,
        _tensor_to_rgb_image,
        gallery_dir,
        refresh_gallery_index,
    )


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _paired_occlusion_mask(masked_rgb: np.ndarray, original_rgb: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    diff = cv2.absdiff(masked_rgb, original_rgb).mean(axis=2)
    mask = np.where(diff > threshold, 255, 0).astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(mask.astype("float32") / 255.0, (7, 7), 0)


def _masked_mse(prediction_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> float:
    mask_3 = np.clip(mask[..., None], 0.0, 1.0).astype("float32")
    weighted_error = ((prediction_rgb - target_rgb) ** 2) * mask_3
    denom = float(mask_3.sum() * prediction_rgb.shape[2])
    if denom <= 1e-6:
        return float(((prediction_rgb - target_rgb) ** 2).mean())
    return float(weighted_error.sum() / denom)


def evaluate_dataset(masked_root: Path, original_root: Path, limit: int | None = None, output_dir: Path | None = None) -> dict:
    refresh_gallery_index()

    total = 0
    detected_faces = 0
    reconstruction_mse_sum = 0.0
    reconstruction_psnr_sum = 0.0
    occluded_region_mse_sum = 0.0
    occluded_region_psnr_sum = 0.0
    reconstruction_ssim_sum = 0.0
    total_time = 0.0

    y_true = []
    y_pred = []
    y_scores = []
    y_match = []

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
            if analysis["recognition_is_occluded"]:
                face_for_recognition, _ = _reconstructed_region_only_face(face_tensor, analysis["recognition_mask"])
            else:
                face_for_recognition = face_tensor

            if analysis["is_occluded"]:
                _, reconstructed_rgb = _blend_reconstructed_face(
                    analysis["reconstruction_tensor"],
                    analysis["occlusion_mask"],
                    enhance_patch=not analysis["detected_face"],
                )
                reconstructed_rgb = reconstructed_rgb.astype("float32")
            else:
                reconstructed_rgb = analysis["reconstruction_rgb"].astype("float32")

            predicted_name, similarity, _ = _predict_identity(face_for_recognition)
            elapsed = time.perf_counter() - start
            
            true_name = person_dir.name
            y_true.append(true_name)
            y_pred.append(predicted_name if predicted_name else "Unknown")
            y_scores.append(similarity)
            y_match.append(1 if true_name == predicted_name else 0)

            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            original_rgb = cv2.resize(original_rgb, (112, 112), interpolation=cv2.INTER_AREA).astype("float32")
            masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
            masked_rgb = cv2.resize(masked_rgb, (112, 112), interpolation=cv2.INTER_AREA).astype("float32")
            
            occlusion_mask = _paired_occlusion_mask(masked_rgb, original_rgb)
            mse_pixels = float(((reconstructed_rgb - original_rgb) ** 2).mean())
            psnr = _psnr_from_mse(mse_pixels)
            occluded_mse = _masked_mse(reconstructed_rgb, original_rgb, occlusion_mask)
            occluded_psnr = _psnr_from_mse(occluded_mse)
            
            # SSIM calculation
            img_rec_uint8 = np.clip(reconstructed_rgb, 0, 255).astype(np.uint8)
            img_orig_uint8 = np.clip(original_rgb, 0, 255).astype(np.uint8)
            current_ssim = ssim(img_orig_uint8, img_rec_uint8, multichannel=True, channel_axis=2, data_range=255)

            total += 1
            detected_faces += int(analysis["detected_face"])
            reconstruction_mse_sum += mse_pixels
            reconstruction_psnr_sum += psnr
            occluded_region_mse_sum += occluded_mse
            occluded_region_psnr_sum += occluded_psnr
            reconstruction_ssim_sum += current_ssim
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
    
    # Calculate ROC, FAR (FPR), TAR (TPR)
    fpr, tpr, thresholds = roc_curve(y_match, y_scores)
    roc_auc = auc(fpr, tpr)
    
    results = {
        "samples_evaluated": total,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "detected_face_rate": detected_faces / total,
        "avg_reconstruction_mse": reconstruction_mse_sum / total,
        "avg_reconstruction_psnr": reconstruction_psnr_sum / total,
        "avg_reconstruction_ssim": reconstruction_ssim_sum / total,
        "avg_occluded_region_mse": occluded_region_mse_sum / total,
        "avg_occluded_region_psnr": occluded_region_psnr_sum / total,
        "avg_fps": total / total_time if total_time > 0 else 0.0,
    }
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC
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
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = output_path.parent
    
    results = evaluate_dataset(Path(args.masked), Path(args.original), args.limit, output_dir)
    output_path.write_text(json.dumps(results, indent=2))

    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"saved_to: {output_path.resolve()}")
