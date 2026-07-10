# Submission Code Guide

This file is the cleaned map for the code submission. It answers which code is
used for training, which algorithm is active now, and which files should be kept
or excluded.

## 1. Current Training Code

### Main reconstruction training

Use `backend/train_reconstruction.py`.

This is the current training script for the reconstruction model. It trains:

- DEGAN generator by default.
- Optional MTR-UNet residual refiner when `--use-refiner true` is passed.
- Optional PatchGAN discriminator for adversarial sharpness feedback.

The script creates paired training samples from masked/original face images and
synthetic occlusions. The key objective is to reconstruct only the occluded
region while preserving the original pixels and identity outside the mask.

Important training losses in `backend/train_reconstruction.py`:

- Masked L1/MSE reconstruction loss.
- Outside-mask preservation loss.
- VGGFace2 identity loss through frozen `InceptionResnetV1`.
- VGG16 perceptual loss.
- Gradient edge loss.
- Laplacian detail loss.
- Masked SSIM structure loss.
- Total variation loss.
- Optional PatchGAN adversarial loss.

Latest important training wrappers:

- `backend/run_degan_retrain_vggface2.ps1` - DEGAN retraining with cleaned
  VGGFace2 extra originals.
- `backend/run_refiner_clarity_vggface2.ps1` - MTR-UNet refiner training.

### Older OAN training

Use `backend/train.py` only if describing the older OAN baseline.

It trains an OAN classifier with ArcFace-style classification head and saves:

- `weights/oan_<backbone>_best.pth`
- `weights/oan_<backbone>_final.pth`

For the current submission, do not describe OAN as the active recognition model
unless you are explicitly discussing the earlier baseline.

## 2. Current Active Runtime Code

Use `backend/main.py`.

This is the active FastAPI backend used by the app. The main user-facing endpoint
is:

- `POST /analyze`

Supporting endpoints include:

- `POST /reconstruct`
- `POST /recognize`
- `POST /refresh-gallery`

## 3. Current Algorithm

The current default algorithm is:

`mask-restricted identity-preserving DEGAN reconstruction + VGGFace2 k-NN recognition`

Step by step:

1. MTCNN detects and aligns the uploaded face.
2. The backend estimates the occlusion mask using face-region and panel-mask
   heuristics.
3. The mask is restricted to the central mouth/chin region for lower-face
   occlusions.
4. DEGAN reconstructs the hidden facial content from the occluded face and mask.
5. The backend composites reconstruction only inside the detected mask.
6. Strict guard logic preserves all pixels outside the mask as much as possible.
7. The VGGFace2 FaceNet model extracts an identity embedding.
8. The embedding is matched against `weights/gallery_index.pt` with k-NN scoring.
9. If reconstruction recognition is not clearly better than the original input,
   the backend keeps the safer original-input recognition result.

Important defaults in `backend/main.py`:

- `RECONSTRUCT_ONLY_OCCLUDED_REGION=1`
- `STRICT_OCCLUDER_REMNANT_CLEANUP_ENABLED=1`
- `EXPAND_RECTANGULAR_PANEL_MASK=0`
- `RECOGNITION_GALLERY_SCORING=knn`
- `RECOGNITION_KNN_K=9`
- `ENABLE_DEGAN_OAN=0`

## 4. Model Architecture

### DEGAN

Defined in `backend/models.py`.

DEGAN is a U-Net style generator with:

- Image plus mask input.
- Gated first convolution to suppress corrupted pixels.
- Encoder/decoder skip connections.
- CBAM attention at skip merge points.
- Residual bottleneck blocks.
- Tanh RGB output.

### MTR-UNet

Defined in `backend/models.py`.

MTR-UNet is an optional residual texture refiner that receives:

- Occluded face.
- DEGAN output.
- Occlusion mask.

The current backend loads it, but skips or rejects it for large rectangular masks
when it would damage identity or create artifacts.

### Recognition

The active recognition model is:

- `facenet_pytorch.InceptionResnetV1(pretrained="vggface2")`

The gallery index is built from:

- `database/lfw-deepfunneled`

The gallery cache is:

- `weights/gallery_index.pt`

## 5. Active Checkpoints

The backend currently tries DEGAN weights in this order:

1. `weights/degan_model_active.pth`
2. `weights/degan_model_best.pth`
3. `weights/degan_model_final.pth`
4. `outputs/identity_drift_finetune_4epoch/checkpoints/degan_model_epoch_004.pth`

For the clean repository, use `weights/degan_model_active.pth`. The `outputs/`
fallback path remains in code only for compatibility with old local runs; it is
not required in the cleaned code package.

The backend currently tries MTR-UNet weights in this order:

1. `weights/mtr_unet_final.pth`
2. `weights/mtr_unet_best.pth`

The optional DEGAN-OAN fusion checkpoint exists locally, but it is disabled by
default because `ENABLE_DEGAN_OAN=0`.

## 6. Files To Keep For Code Submission

Keep these as the core submission:

- `README.md`
- `ARCHITECTURE.md`
- `SUBMISSION_CODE_GUIDE.md`
- `.env.example`
- `.gitignore`
- `backend/main.py`
- `backend/models.py`
- `backend/train_reconstruction.py`
- `backend/train.py`
- `backend/dataset_masker.py`
- `backend/clean_dataset.py`
- `backend/dataset_split.py`
- `backend/face_align.py`
- `backend/quality_filter.py`
- `backend/oan_refiner.py`
- `backend/evaluate.py`
- `backend/ablation_evaluate.py`
- `backend/evaluate_checkpoint.py`
- `backend/test_api.py`
- `backend/requirements.txt`
- `backend/run_degan_retrain_vggface2.ps1`
- `backend/run_refiner_clarity_vggface2.ps1`
- `frontend/`

Keep these evaluation scripts only if your marker wants reproduction scripts:

- `evaluate_project.py`
- `fair_evaluate_lfw_15may.py`
- `prepare_lfw15may_original_protocol.py`
- `standardize_lfw_15may.py`

## 7. Files/Folders To Exclude Or Archive

These are experiment history, generated outputs, backups, or local environments.
Do not include them in the clean code package unless the marker specifically
asks for experiment logs.

- `.venv/`
- `venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `dist/`
- `outputs/`
- `evaluation*/`
- `backups/`
- `backend_blend/`
- `frontend_backup/`
- `Source Code/`
- `Source Code.zip`
- `tmp/`
- `tools/`
- `journal_figures/`
- `jiwe_paper_render*/`
- `rendered_jiwe*/`
- `__pycache__/`
- `results.csv`
- Generated `*.pdf`, `*.docx`, and `*.zip` exports unless explicitly requested
- Root-level `debug_*.py`
- Root-level `*_experiment.py`
- Root-level `final_evaluate_*.py`
- Root-level `local_patch_*_experiment.py`
- Root-level `mouth_chin_algorithm_refiner_experiment.py`
- Root-level `mtr_refiner_checkpoint_evaluate.py`
- Root-level `reconstruction_quality_blending_experiment.py`
- Root-level generated report helper files such as `update_final_fyp_report.py`

## 8. Suggested Submission Folder Shape

```text
FYP_submission/
  README.md
  ARCHITECTURE.md
  SUBMISSION_CODE_GUIDE.md
  .env.example
  backend/
  frontend/
  weights/                # optional, if binary weights are allowed
  database/README.txt     # optional note explaining required dataset layout
```

For large artifacts, submit model weights and datasets separately and explain
where they should be placed.

## 9. Important Wording For Report/Presentation

Use this wording for the final system:

> The final system uses a mask-restricted identity-preserving DEGAN
> reconstruction module. It reconstructs the occluded lower-face region only,
> preserves original pixels outside the mask, and performs gallery recognition
> using VGGFace2 FaceNet embeddings with k-NN scoring.

Avoid saying:

- "The final recognizer is OAN" unless discussing the older baseline.
- "The refiner is always used" because the backend guards or skips MTR-UNet for
  large rectangular masks.
- "DEGAN-OAN fusion is active" because it is disabled by default.
