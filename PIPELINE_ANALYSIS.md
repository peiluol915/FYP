# Pipeline Analysis

## 1. Preprocessing and Augmentation Pipeline
- **Dataset Preparation (`dataset_masker.py`)**: Takes original datasets (like LFW or CelebA) and applies synthetic occlusions (standard surgical masks, heavy occlusions, opaque panels, hands, scarves, etc.).
- **Data Augmentation**: 
  - Standard facial recognition transforms: Resize (112x112), Random Horizontal Flip, Normalization.
  - On-the-fly synthetic occlusion during reconstruction training (`MaskedFacePairDataset` in `train_reconstruction.py`).

## 2. Training Pipeline

### A. Classifier Training (`train.py`)
- **Input**: Occluded face images.
- **Architecture**: `OAN` (ResNet50 + OAM).
- **Loss**: Standard Cross-Entropy over identities.
- **Current Limitations**: Lacks a validation loop, learning rate scheduling, mixed-precision, and modern face recognition losses (like ArcFace).

### B. Reconstruction Training (`train_reconstruction.py`)
- **Input**: Paired dataset of Masked vs Original faces.
- **Architecture**: `DEGAN`.
- **Losses**:
  - Masked L1 + MSE Loss (focus on occluded regions).
  - Full-face L1 + MSE Loss.
  - Perceptual Edge Loss (Gradient Map difference).
  - Structure Loss (SSIM).
  - Identity Loss (Cosine similarity between reconstructed and original embeddings using InceptionResnetV1).
- **Optimization**: Mixed-precision scaling with Adam.

## 3. Inference Pipeline (`main.py`)
1. **Input**: User uploads an image via the React frontend.
2. **Face Detection**: `MTCNN` detects the face bounding box and landmarks.
3. **Occlusion Estimation**: Heuristics (HSV thresholding, texture energy via Laplacian) determine the occlusion mask.
4. **Reconstruction**: `DEGAN` takes the face tensor and occlusion mask to reconstruct the missing regions. The reconstructed patch is then blended back into the original image.
5. **Embedding Extraction**: The reconstructed face is passed through the recognition model to extract a 512-D embedding.
6. **Matching**: The embedding is compared against a pre-computed gallery (`gallery_index.pt`) using cosine similarity.
7. **Output**: The API returns the matched identity, confidence, and reconstructed images.

## 4. Evaluation Strategy (`evaluate.py`)
- Runs batch inference on a masked dataset and compares against the original unoccluded dataset.
- Tracks Metrics:
  - Top-1 Accuracy.
  - Average Reconstruction MSE and PSNR.
  - Occluded Region MSE and PSNR.
  - Inference Speed (FPS).
- **Current Limitations**: Needs more comprehensive metrics (ROC, FAR/TAR, SSIM, F1, Precision, Recall).
