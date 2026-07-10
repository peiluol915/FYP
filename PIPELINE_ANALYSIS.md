# Pipeline Analysis

## 1. Preprocessing And Augmentation

- `backend/dataset_masker.py` creates synthetic occlusions such as surgical
  masks, sunglasses, hands, scarves, and opaque panels.
- `backend/face_align.py` and MTCNN-based routines align faces to the 112x112
  model input format.
- `backend/clean_dataset.py`, `backend/dataset_split.py`, and
  `backend/quality_filter.py` prepare balanced and quality-controlled datasets.
- `MaskedFacePairDataset` in `backend/train_reconstruction.py` can add
  synthetic occlusion during reconstruction training.

## 2. Training Pipeline

### Reconstruction Training

`backend/train_reconstruction.py` is the current training path. It trains DEGAN
by default and can optionally train an MTR-UNet refiner with adversarial
feedback.

Core losses include:

- masked reconstruction loss,
- outside-mask preservation loss,
- VGGFace2 identity loss,
- VGG16 perceptual loss,
- gradient and Laplacian detail losses,
- masked SSIM structure loss,
- total variation regularization,
- optional PatchGAN adversarial loss.

### Baseline OAN Training

`backend/train.py` reproduces the older OAN baseline. It is kept for comparison
and report context, while the active runtime recognition path uses VGGFace2
FaceNet embeddings and gallery k-NN scoring.

## 3. Inference Pipeline

1. The React frontend sends an uploaded image to `POST /analyze`.
2. MTCNN detects and aligns the face.
3. The backend estimates an occlusion mask and restricts reconstruction to the
   accepted lower-face or panel region.
4. DEGAN reconstructs the hidden content.
5. The final display image replaces only pixels inside the detected mask.
6. VGGFace2 FaceNet extracts embeddings from the input and reconstructed probes.
7. Gallery k-NN scoring compares embeddings against `weights/gallery_index.pt`.
8. The API returns recognition decisions, top matches, diagnostic metrics, and
   base64 image views for the frontend.

## 4. Evaluation Strategy

- `backend/evaluate.py` measures reconstruction and identity performance.
- `backend/ablation_evaluate.py` supports ablation comparisons.
- `backend/evaluate_checkpoint.py` compares saved checkpoints against the active
  backend setup.
- Optional root-level evaluation scripts can reproduce the LFW 15 May protocol,
  but they are kept out of the clean submission package unless explicitly
  required.
