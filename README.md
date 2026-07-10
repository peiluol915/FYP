# Occluded Face Recognition and Reconstruction

This repository contains the final codebase for an occluded face recognition
and reconstruction system. The backend is a PyTorch/FastAPI service, and the
frontend is a React/Vite viewer for uploading an occluded face, comparing the
detected mask and reconstructed output, and inspecting gallery recognition
results.

## Current Default Pipeline

The active runtime entry point is `backend/main.py`.

1. Detect and align the uploaded face with MTCNN.
2. Estimate the occlusion mask from the aligned face.
3. Reconstruct the hidden region with the DEGAN generator.
4. Composite the reconstruction only inside the detected occlusion mask, keeping
   unmasked pixels from the original input.
5. Extract VGGFace2 FaceNet embeddings with
   `InceptionResnetV1(pretrained="vggface2")`.
6. Match identity against the cached gallery index with k-NN scoring.

The current default algorithm is:

`mask-restricted identity-preserving DEGAN reconstruction + VGGFace2 k-NN recognition`

OAN is still present for the older baseline and optional experimental fusion
path, but it is not the default recognizer.

## Repository Layout

- `backend/main.py` - FastAPI inference API, occlusion detection,
  reconstruction, mask-only compositing, gallery indexing, and recognition.
- `backend/models.py` - DEGAN, MTR-UNet, OAN, CBAM, gated convolution, and
  PatchGAN model definitions.
- `backend/oan_refiner.py` - optional DEGAN/OAN fusion module, disabled unless
  explicitly enabled.
- `backend/train_reconstruction.py` - current DEGAN training script with
  optional MTR-UNet refiner and PatchGAN discriminator.
- `backend/train.py` - older OAN classifier/embedding training script.
- `backend/dataset_masker.py` - synthetic occlusion generation.
- `backend/clean_dataset.py`, `backend/dataset_split.py`,
  `backend/face_align.py`, `backend/quality_filter.py` - dataset preparation
  helpers.
- `backend/evaluate.py`, `backend/ablation_evaluate.py`,
  `backend/evaluate_checkpoint.py` - backend evaluation utilities.
- `frontend/` - React/Vite upload and comparison interface.
- `SUBMISSION_CODE_GUIDE.md` - cleaned submission map and final algorithm notes.

Generated reports, paper renders, local export zips, virtual environments,
model weights, datasets, build folders, and experiment output folders are
excluded from version control.

## Required Local Artifacts

Large data and model files are intentionally kept outside Git.

- `database/lfw-deepfunneled/` - gallery/reference images.
- `weights/gallery_index.pt` - cached gallery embeddings. The backend rebuilds
  it automatically from `database/lfw-deepfunneled` when it is missing or stale.
- `weights/degan_model_active.pth` - preferred active DEGAN checkpoint unless
  `DEGAN_WEIGHT_PATH` is set.
- `weights/degan_model_best.pth` and `weights/degan_model_final.pth` - fallback
  DEGAN checkpoints.
- `weights/mtr_unet_final.pth` or `weights/mtr_unet_best.pth` - optional
  MTR-UNet refiner checkpoints.

If a submission platform does not allow large binaries in the source package,
submit weights and datasets separately and place them in the paths above before
running the app.

## Setup

Create and activate a Python environment, then install backend dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

## Run

Start the backend from the project root. Port `8011` is the preferred local
port because the frontend probes it first, then falls back to `8001` and `8000`.

```powershell
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8011
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL printed by the frontend command. The main backend endpoint is
`POST /analyze`; supporting endpoints are `POST /reconstruct`,
`POST /recognize`, and `POST /refresh-gallery`.

## Training

Train the current reconstruction model with:

```powershell
venv\Scripts\python.exe backend\train_reconstruction.py --masked <masked_dataset> --original <original_dataset>
```

Useful wrappers for the latest major reconstruction runs:

- `backend/run_degan_retrain_vggface2.ps1`
- `backend/run_refiner_clarity_vggface2.ps1`

Train the older OAN baseline only when you need to reproduce the baseline:

```powershell
venv\Scripts\python.exe backend\train.py --data <masked_identity_dataset>
```

## Verification

```powershell
venv\Scripts\python.exe -m compileall backend
cd frontend
npm run build
```
