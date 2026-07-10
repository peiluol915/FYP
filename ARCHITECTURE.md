# Architecture Overview

This project implements an end-to-end occluded face reconstruction and
recognition system with a PyTorch/FastAPI backend and a React/Vite frontend.

## 1. Frontend

`frontend/src/App.tsx` is a single-page upload and comparison interface. It
posts images to the backend `POST /analyze` endpoint, then displays:

- the occluded input,
- the detected occlusion mask,
- the raw DEGAN output,
- the final mask-restricted reconstruction,
- input and reconstructed recognition candidates,
- similarity, confidence, and decision metrics.

The frontend probes local backend ports in this order: `8011`, `8001`, `8000`.

## 2. Backend

`backend/main.py` is the active inference service. It provides:

- `POST /analyze` - full reconstruction and recognition pipeline.
- `POST /reconstruct` - reconstruction-only output.
- `POST /recognize` - recognition-only output.
- `POST /refresh-gallery` - rebuilds the cached gallery embeddings.

The backend can also serve the built frontend from `frontend/dist` when that
folder exists.

## 3. Inference Pipeline

1. MTCNN detects and aligns the uploaded face.
2. Heuristic mask estimation identifies lower-face and panel-style occlusions.
3. DEGAN reconstructs the occluded facial region.
4. Guard logic composites reconstruction only inside the accepted mask.
5. VGGFace2 FaceNet embeddings are extracted with
   `InceptionResnetV1(pretrained="vggface2")`.
6. The embedding is matched against `weights/gallery_index.pt` with k-NN
   scoring.
7. If reconstructed recognition is not clearly better, the safer original-input
   recognition result is kept.

## 4. Models

- **DEGAN**: U-Net style generator with gated input convolution, residual
  bottleneck blocks, skip connections, and CBAM attention.
- **MTR-UNet**: Optional residual texture refiner. The backend loads it when
  weights are available but may skip it when guard logic predicts identity or
  artifact risk.
- **InceptionResnetV1 (VGGFace2)**: Active embedding model for gallery
  recognition and identity-preserving reconstruction losses.
- **OAN**: Older occlusion-aware baseline model retained for comparison and
  disabled experimental fusion. It is not the default recognizer.

## 5. Training And Evaluation

- `backend/train_reconstruction.py` trains DEGAN and, optionally, MTR-UNet and a
  PatchGAN discriminator.
- `backend/train.py` reproduces the older OAN baseline.
- `backend/dataset_masker.py` creates synthetic occlusions for reconstruction
  and recognition experiments.
- `backend/evaluate.py`, `backend/ablation_evaluate.py`, and
  `backend/evaluate_checkpoint.py` provide evaluation and checkpoint comparison
  utilities.
