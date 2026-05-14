# Remaining TODOs

## 1. System Fixes & Completion
- [ ] **Training Loop (`train.py`)**: Add validation loop, metrics tracking, mixed-precision (AMP), learning rate scheduling, and best-checkpoint saving.
- [ ] **Resume Capability**: Implement robust checkpoint resuming in `train.py`.
- [ ] **Configuration Management**: Extract hyperparameters and paths into a configuration system.

## 2. Model Improvements
- [ ] **ArcFace Loss**: Upgrade `train.py` to use ArcFace/CosFace margin-based softmax for better embedding separation.
- [ ] **Lightweight Backbone**: Integrate MobileFaceNet or PartialFace as an alternative to ResNet50 for faster real-time inference.
- [ ] **Advanced Attention**: Replace the basic OAM sigmoid attention with a more robust module (e.g., CBAM or self-attention blocks).

## 3. Evaluation & Metrics
- [ ] **Extended Metrics**: Update `evaluate.py` to calculate Precision, Recall, F1 Score, ROC curves, FAR (False Acceptance Rate), and TAR (True Acceptance Rate).
- [ ] **SSIM Metric**: Add structural similarity index measure to the evaluation script.
- [ ] **Visualization**: Generate confusion matrices, ROC curve plots, and training loss graphs.

## 4. Research & Documentation
- [ ] **Code Quality**: Add comprehensive docstrings to all major functions.
- [ ] **Requirements**: Generate an updated `requirements.txt`.
- [ ] **Setup Guide**: Ensure `README.md` includes clear setup and running instructions.
