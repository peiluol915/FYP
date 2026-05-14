# Architecture Overview

This project implements an end-to-end occluded face recognition and reconstruction system, primarily built with PyTorch, FastAPI, and React.

## System Components

### 1. Frontend (React + Vite)
- **App.tsx**: A single-page application that allows users to upload masked images.
- **Integration**: Communicates with the FastAPI backend via REST API (`/analyze`).
- **UI/UX**: Displays a side-by-side comparison of the occluded input, reconstructed face, and original reference face (if identity is recognized).

### 2. Backend (FastAPI)
- **main.py**: The entry point for inference. It exposes an `/analyze` endpoint that:
  1. Detects faces using MTCNN.
  2. Estimates occlusion severity and region.
  3. Reconstructs the unoccluded face using the DEGAN generator.
  4. Extracts embeddings using either the OAN backbone or InceptionResnetV1.
  5. Matches the embedding against a pre-computed gallery (`gallery_index.pt`).

### 3. Deep Learning Models (PyTorch)
- **OAN (Occlusion Aware Network)**: A ResNet-50 based classification backbone equipped with an OAM (Occlusion Aware Module) attention mechanism.
- **DEGAN (Reconstruction GAN)**: A U-Net style architecture with residual bottleneck blocks that reconstructs occluded facial regions.
- **MTCNN**: Used for robust face detection and alignment.
- **InceptionResnetV1 (VGGFace2)**: Used for identity-preserving losses and high-accuracy embedding matching.

### 4. Data Processing & Training
- **dataset_masker.py**: A robust pipeline to synthesize complex occlusions (masks, sunglasses, hands, scarves) on unoccluded datasets.
- **train_reconstruction.py**: The training loop for DEGAN. It implements identity-preserving losses, perceptual edge losses, and SSIM to ensure realistic reconstructions.
- **train.py**: The training loop for the OAN classification model.
- **evaluate.py**: Inference evaluation script measuring MSE, PSNR, and Top-1 Accuracy.
