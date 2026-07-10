import base64
import os
import time
import warnings
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

try:
    from models import DEGAN, MTRUNet, OAN
except ModuleNotFoundError:
    from backend.models import DEGAN, MTRUNet, OAN

try:
    from oan_refiner import OANGatedFusion
except ModuleNotFoundError:
    from backend.oan_refiner import OANGatedFusion


warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GALLERY_CACHE_VERSION = 6
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_dist_dir = BASE_DIR / "frontend" / "dist"
frontend_assets_dir = frontend_dist_dir / "assets"
frontend_index_path = frontend_dist_dir / "index.html"
weights_dir = BASE_DIR / "weights"
gallery_dir = BASE_DIR / "database" / "lfw-deepfunneled"
gallery_cache_path = weights_dir / "gallery_index.pt"
FINAL_DEGAN_CHECKPOINT = Path(
    os.environ.get(
        "DEGAN_WEIGHT_PATH",
        str(weights_dir / "degan_model_active.pth"),
    )
)
DEGAN_WEIGHT_CANDIDATES = (
    FINAL_DEGAN_CHECKPOINT,
    "degan_model_best.pth",
    "degan_model_final.pth",
    BASE_DIR / "outputs" / "identity_drift_finetune_4epoch" / "checkpoints" / "degan_model_epoch_004.pth",
)
MTR_REFINER_WEIGHT_CANDIDATES = ("mtr_unet_final.pth", "mtr_unet_best.pth")
DEGAN_OAN_CHECKPOINT = Path(
    os.environ.get(
        "DEGAN_OAN_WEIGHT_PATH",
        str(BASE_DIR / "outputs" / "degan_oan_fusion_followup_13June" / "checkpoints" / "gated_degan_oan" / "gated_degan_oan_final.pth"),
    )
)
DEGAN_OAN_WEIGHT_CANDIDATES = (DEGAN_OAN_CHECKPOINT,)
DEGAN_OAN_ENABLED = os.environ.get("ENABLE_DEGAN_OAN", "0").strip().lower() in {"1", "true", "yes"}
DEGAN_OAN_PATCH_SIZE = max(64, int(os.environ.get("DEGAN_OAN_PATCH_SIZE", "192")))
DEGAN_OAN_RESIDUAL_SCALE = float(os.environ.get("DEGAN_OAN_RESIDUAL_SCALE", "0.45"))
DEGAN_OAN_ACCEPT_MIN_SCORE_GAIN = float(os.environ.get("DEGAN_OAN_ACCEPT_MIN_SCORE_GAIN", "0.0"))
DEGAN_OAN_RAW_REGION_COMPOSITING = os.environ.get("DEGAN_OAN_RAW_REGION_COMPOSITING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEGAN_OAN_PATCH_RESIZE_INTERPOLATION = os.environ.get("DEGAN_OAN_PATCH_RESIZE_INTERPOLATION", "cubic").strip().lower()
DEGAN_OAN_GATE_SMOOTH_KERNEL = max(0, int(os.environ.get("DEGAN_OAN_GATE_SMOOTH_KERNEL", "0")))
RECTANGULAR_DETAIL_RESCUE_STRENGTH = 0.55
RECTANGULAR_MODEL_DETAIL_STRENGTH = float(os.environ.get("RECTANGULAR_MODEL_DETAIL_STRENGTH", "0.75"))
DEGAN_PRIOR_PRESERVE_DISPLAY = os.environ.get("DEGAN_PRIOR_PRESERVE_DISPLAY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEGAN_PRIOR_TONE_MATCH_STRENGTH = float(os.environ.get("DEGAN_PRIOR_TONE_MATCH_STRENGTH", "0.16"))
DEGAN_PRIOR_COLOR_ANCHOR_STRENGTH = float(os.environ.get("DEGAN_PRIOR_COLOR_ANCHOR_STRENGTH", "0.18"))
RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD = 40.0
RECTANGULAR_VISUAL_REPAIR_STRENGTH = float(os.environ.get("RECONSTRUCTION_VISUAL_REPAIR_STRENGTH", "0.0"))
HYBRID_RECOGNITION_MARGIN = float(os.environ.get("HYBRID_RECOGNITION_MARGIN", "0.04"))
STRICT_OCCLUSION_BLEND_FEATHER = int(os.environ.get("STRICT_OCCLUSION_BLEND_FEATHER", "0"))
DIRECT_PANEL_MIN_COVERAGE = 0.008
DIRECT_PANEL_MAX_COVERAGE = 0.22
DIRECT_PANEL_BOUNDARY_JUMP_THRESHOLD = float(
    os.environ.get("DIRECT_PANEL_BOUNDARY_JUMP_THRESHOLD", str(RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD))
)
DIRECT_PANEL_DISPLAY_SEAM_PAD = max(0, int(os.environ.get("DIRECT_PANEL_DISPLAY_SEAM_PAD", "1")))
RECONSTRUCT_ONLY_OCCLUDED_REGION = os.environ.get("RECONSTRUCT_ONLY_OCCLUDED_REGION", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
STRICT_DISPLAY_MASK_GUARD = os.environ.get("STRICT_DISPLAY_MASK_GUARD", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
STRICT_DISPLAY_EDGE_CLEANUP = os.environ.get("STRICT_DISPLAY_EDGE_CLEANUP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEGAN_PRIOR_SEAM_RING = os.environ.get("DEGAN_PRIOR_SEAM_RING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
STRICT_DISPLAY_SEAM_INPAINT = os.environ.get("STRICT_DISPLAY_SEAM_INPAINT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
STRICT_DISPLAY_SEAM_WIDTH = max(1, int(os.environ.get("STRICT_DISPLAY_SEAM_WIDTH", "3")))
STRICT_DISPLAY_SEAM_INPAINT_RADIUS = max(1, int(os.environ.get("STRICT_DISPLAY_SEAM_INPAINT_RADIUS", "2")))
DEGAN_PRIOR_SHAPE_STRENGTH = float(os.environ.get("DEGAN_PRIOR_SHAPE_STRENGTH", "0.55"))
DEGAN_PRIOR_SHAPE_MAX_DELTA = float(os.environ.get("DEGAN_PRIOR_SHAPE_MAX_DELTA", "28.0"))
DEGAN_PRIOR_SHAPE_SIGMA = float(os.environ.get("DEGAN_PRIOR_SHAPE_SIGMA", "2.4"))
RESTRICT_RECONSTRUCTION_MASK_TO_MOUTH_CHIN = os.environ.get(
    "RESTRICT_RECONSTRUCTION_MASK_TO_MOUTH_CHIN",
    "1",
).strip().lower() in {
    "1",
    "true",
    "yes",
}
STRICT_OCCLUDER_REMNANT_CLEANUP_ENABLED = os.environ.get(
    "STRICT_OCCLUDER_REMNANT_CLEANUP_ENABLED",
    "1",
).strip().lower() in {
    "1",
    "true",
    "yes",
}
EXPAND_RECTANGULAR_PANEL_MASK = os.environ.get("EXPAND_RECTANGULAR_PANEL_MASK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RECONSTRUCTION_MASK_ERODE_PIXELS = max(0, int(os.environ.get("RECONSTRUCTION_MASK_ERODE_PIXELS", "0")))
PANEL_EDGE_CLEANUP_ENABLED = os.environ.get("PANEL_EDGE_CLEANUP_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_EDGE_CLEANUP_OUTER_PAD = max(0, int(os.environ.get("PANEL_EDGE_CLEANUP_OUTER_PAD", "5")))
PANEL_EDGE_CLEANUP_INNER_PAD = max(0, int(os.environ.get("PANEL_EDGE_CLEANUP_INNER_PAD", "0")))
PANEL_EDGE_CLEANUP_RADIUS = max(1, int(os.environ.get("PANEL_EDGE_CLEANUP_RADIUS", "3")))
PANEL_SUPPORT_CLEANUP_ENABLED = os.environ.get("PANEL_SUPPORT_CLEANUP_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_SUPPORT_CLEANUP_X_SCALE = float(os.environ.get("PANEL_SUPPORT_CLEANUP_X_SCALE", "0.55"))
PANEL_SUPPORT_MODEL_BLEND = float(os.environ.get("PANEL_SUPPORT_MODEL_BLEND", "0.0"))
PANEL_BOUNDARY_BLEND_ENABLED = os.environ.get("PANEL_BOUNDARY_BLEND_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_BOUNDARY_BLEND_RADIUS = max(1, int(os.environ.get("PANEL_BOUNDARY_BLEND_RADIUS", "8")))
PANEL_BOUNDARY_INPAINT_RADIUS = max(1, int(os.environ.get("PANEL_BOUNDARY_INPAINT_RADIUS", "3")))
PANEL_DETAIL_OVAL_SCALE_X = float(os.environ.get("PANEL_DETAIL_OVAL_SCALE_X", "0.75"))
PANEL_DETAIL_OVAL_SCALE_Y = float(os.environ.get("PANEL_DETAIL_OVAL_SCALE_Y", "0.70"))
PANEL_DETAIL_ALPHA_BLUR = max(1, int(os.environ.get("PANEL_DETAIL_ALPHA_BLUR", "13")))
PANEL_DETAIL_ALPHA_STRENGTH = float(os.environ.get("PANEL_DETAIL_ALPHA_STRENGTH", "0.76"))
PANEL_DETAIL_LOCAL_BASE_ENABLED = os.environ.get("PANEL_DETAIL_LOCAL_BASE_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE = float(os.environ.get("PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE", "0.44"))
PANEL_DETAIL_HIGH_PASS_STRENGTH = float(os.environ.get("PANEL_DETAIL_HIGH_PASS_STRENGTH", "0.26"))
PANEL_DETAIL_UNSHARP_AMOUNT = float(os.environ.get("PANEL_DETAIL_UNSHARP_AMOUNT", "0.32"))
PANEL_DETAIL_UNSHARP_SIGMA = float(os.environ.get("PANEL_DETAIL_UNSHARP_SIGMA", "0.90"))
PANEL_DETAIL_TONE_MATCH_STRENGTH = float(os.environ.get("PANEL_DETAIL_TONE_MATCH_STRENGTH", "0.78"))
PANEL_DETAIL_PRESERVE_STRENGTH = float(os.environ.get("PANEL_DETAIL_PRESERVE_STRENGTH", "0.80"))
PANEL_SKIN_BASE_L_PERCENTILE = float(os.environ.get("PANEL_SKIN_BASE_L_PERCENTILE", "75.0"))
PANEL_SKIN_BASE_PLANE_STRENGTH = float(os.environ.get("PANEL_SKIN_BASE_PLANE_STRENGTH", "0.18"))
PANEL_MODEL_DETAIL_TONE_STRENGTH = float(os.environ.get("PANEL_MODEL_DETAIL_TONE_STRENGTH", "0.25"))
PANEL_ENHANCE_DONOR_BLEND_MAX = float(os.environ.get("PANEL_ENHANCE_DONOR_BLEND_MAX", "0.18"))
PANEL_ENHANCE_FINAL_TONE_STRENGTH = float(os.environ.get("PANEL_ENHANCE_FINAL_TONE_STRENGTH", "0.22"))
PANEL_STRUCTURED_REPAIR_ENABLED = os.environ.get("PANEL_STRUCTURED_REPAIR_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_STRUCTURED_REPAIR_SCORE_THRESHOLD = float(os.environ.get("PANEL_STRUCTURED_REPAIR_SCORE_THRESHOLD", "36.0"))
PANEL_STRUCTURED_REPAIR_MARGIN = float(os.environ.get("PANEL_STRUCTURED_REPAIR_MARGIN", "10.0"))
PANEL_SKIN_TONE_MATCH_ENABLED = os.environ.get("PANEL_SKIN_TONE_MATCH_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
PANEL_SKIN_TONE_MATCH_STRENGTH = float(os.environ.get("PANEL_SKIN_TONE_MATCH_STRENGTH", "0.55"))
PANEL_RECTANGULAR_COLOR_ANCHOR_STRENGTH = float(
    os.environ.get("PANEL_RECTANGULAR_COLOR_ANCHOR_STRENGTH", "0.86")
)
PANEL_SUBTLE_STRUCTURE_STRENGTH = float(os.environ.get("PANEL_SUBTLE_STRUCTURE_STRENGTH", "0.0"))
PANEL_RECON_UPSCALE_INTERPOLATION = os.environ.get("PANEL_RECON_UPSCALE_INTERPOLATION", "lanczos").strip().lower()
PANEL_EDGE_INPAINT_ERODE_SIZE = max(3, int(os.environ.get("PANEL_EDGE_INPAINT_ERODE_SIZE", "11")))
PANEL_EDGE_INPAINT_RADIUS = max(1, int(os.environ.get("PANEL_EDGE_INPAINT_RADIUS", "4")))
PANEL_RECON_REGION_SHARPEN_AMOUNT = float(os.environ.get("PANEL_RECON_REGION_SHARPEN_AMOUNT", "0.38"))
PANEL_RECON_REGION_SHARPEN_SIGMA = float(os.environ.get("PANEL_RECON_REGION_SHARPEN_SIGMA", "0.75"))
PANEL_RECON_REGION_SHARPEN_MAX_DELTA = float(os.environ.get("PANEL_RECON_REGION_SHARPEN_MAX_DELTA", "14.0"))
PANEL_MODEL_LUMINANCE_DETAIL_AMOUNT = float(os.environ.get("PANEL_MODEL_LUMINANCE_DETAIL_AMOUNT", "0.24"))
PANEL_MODEL_LUMINANCE_DETAIL_SIGMA = float(os.environ.get("PANEL_MODEL_LUMINANCE_DETAIL_SIGMA", "1.10"))
PANEL_MODEL_LUMINANCE_DETAIL_MAX_DELTA = float(os.environ.get("PANEL_MODEL_LUMINANCE_DETAIL_MAX_DELTA", "9.0"))
PANEL_MODEL_LUMINANCE_STRUCTURE_AMOUNT = float(os.environ.get("PANEL_MODEL_LUMINANCE_STRUCTURE_AMOUNT", "0.20"))
PANEL_MODEL_LUMINANCE_STRUCTURE_SIGMA = float(os.environ.get("PANEL_MODEL_LUMINANCE_STRUCTURE_SIGMA", "2.0"))
PANEL_MODEL_LUMINANCE_STRUCTURE_MAX_DELTA = float(os.environ.get("PANEL_MODEL_LUMINANCE_STRUCTURE_MAX_DELTA", "16.0"))
PANEL_REMNANT_CLEANUP_ENABLED = os.environ.get("PANEL_REMNANT_CLEANUP_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RECOGNITION_GALLERY_SCORING = os.environ.get("RECOGNITION_GALLERY_SCORING", "knn").strip().lower()
RECOGNITION_KNN_K = max(1, int(os.environ.get("RECOGNITION_KNN_K", "9")))
RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD = float(os.environ.get("RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD", "0.45"))
RECOGNITION_ACCEPT_GAP_THRESHOLD = float(os.environ.get("RECOGNITION_ACCEPT_GAP_THRESHOLD", "0.015"))
RECONSTRUCTION_MODEL_IMAGE_SIZE = int(os.environ.get("RECONSTRUCTION_MODEL_IMAGE_SIZE", "112"))
RECOGNITION_ALIGNMENT_IMAGE_SIZE = 112

app = FastAPI()
if frontend_assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=frontend_assets_dir), name="frontend-assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=RECOGNITION_ALIGNMENT_IMAGE_SIZE, margin=0, post_process=False, device=device)


def _resolve_weight_path(weight_ref: str | Path) -> Path:
    path = Path(weight_ref)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return BASE_DIR / path
    return weights_dir / path


def _load_model_weights(model: torch.nn.Module, weight_name: str | Path) -> bool:
    weight_path = _resolve_weight_path(weight_name)
    if not weight_path.exists():
        return False

    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=device)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    if not compatible_state:
        return False
    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=False)
    return True


def _load_first_available_model_weights(
    model: torch.nn.Module,
    weight_names: tuple[str | Path, ...],
) -> tuple[bool, str | None]:
    for weight_name in weight_names:
        if _load_model_weights(model, weight_name):
            return True, str(weight_name)
    return False, None


oan_model = OAN().to(device)
degan_model = DEGAN().to(device)
mtr_refiner = MTRUNet().to(device)
degan_oan_refiner = OANGatedFusion(residual_scale=DEGAN_OAN_RESIDUAL_SCALE).to(device)
degan_oan_refiner.gate_smooth_kernel = DEGAN_OAN_GATE_SMOOTH_KERNEL
recognition_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
oan_loaded = _load_model_weights(oan_model, "oan_model_final.pth")
degan_loaded, degan_weight_name = _load_first_available_model_weights(degan_model, DEGAN_WEIGHT_CANDIDATES)
mtr_refiner_loaded, mtr_refiner_weight_name = _load_first_available_model_weights(
    mtr_refiner,
    MTR_REFINER_WEIGHT_CANDIDATES,
)
degan_oan_loaded, degan_oan_weight_name = _load_first_available_model_weights(
    degan_oan_refiner,
    DEGAN_OAN_WEIGHT_CANDIDATES,
)
oan_model.eval()
degan_model.eval()
mtr_refiner.eval()
degan_oan_refiner.eval()

gallery_embeddings: torch.Tensor | None = None
gallery_names: list[str] = []
gallery_image_paths: list[str] = []
BACKEND_VERSION = "rectangular-structure-sharp-lowblur-seam3d-20260615"


def _runtime_info() -> dict:
    degan_path = _resolve_weight_path(degan_weight_name) if degan_weight_name else _resolve_weight_path(DEGAN_WEIGHT_CANDIDATES[0])
    refiner_path = (
        _resolve_weight_path(mtr_refiner_weight_name)
        if mtr_refiner_weight_name
        else _resolve_weight_path(MTR_REFINER_WEIGHT_CANDIDATES[0])
    )
    degan_oan_path = (
        _resolve_weight_path(degan_oan_weight_name)
        if degan_oan_weight_name
        else _resolve_weight_path(DEGAN_OAN_WEIGHT_CANDIDATES[0])
    )
    degan_modified = None
    refiner_modified = None
    degan_oan_modified = None
    if degan_path.exists():
        degan_modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(degan_path.stat().st_mtime))
    if refiner_path.exists():
        refiner_modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(refiner_path.stat().st_mtime))
    if degan_oan_path.exists():
        degan_oan_modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(degan_oan_path.stat().st_mtime))
    return {
        "backend_version": BACKEND_VERSION,
        "degan_weights_loaded_name": degan_weight_name,
        "degan_weights_path": str(degan_path),
        "degan_weights_modified": degan_modified,
        "refiner_weights_loaded_name": mtr_refiner_weight_name,
        "refiner_weights_path": str(refiner_path),
        "refiner_weights_modified": refiner_modified,
        "degan_oan_enabled": DEGAN_OAN_ENABLED,
        "degan_oan_loaded": degan_oan_loaded,
        "degan_oan_weights_loaded_name": degan_oan_weight_name,
        "degan_oan_weights_path": str(degan_oan_path),
        "degan_oan_weights_modified": degan_oan_modified,
        "degan_oan_patch_size": DEGAN_OAN_PATCH_SIZE,
        "degan_oan_residual_scale": DEGAN_OAN_RESIDUAL_SCALE,
        "degan_oan_accept_min_score_gain": DEGAN_OAN_ACCEPT_MIN_SCORE_GAIN,
        "degan_oan_raw_region_compositing": DEGAN_OAN_RAW_REGION_COMPOSITING,
        "degan_oan_patch_resize_interpolation": DEGAN_OAN_PATCH_RESIZE_INTERPOLATION,
        "degan_oan_gate_smooth_kernel": DEGAN_OAN_GATE_SMOOTH_KERNEL,
        "rectangular_visual_repair_strength": RECTANGULAR_VISUAL_REPAIR_STRENGTH,
        "rectangular_model_detail_strength": RECTANGULAR_MODEL_DETAIL_STRENGTH,
        "degan_prior_preserve_display": DEGAN_PRIOR_PRESERVE_DISPLAY,
        "degan_prior_tone_match_strength": DEGAN_PRIOR_TONE_MATCH_STRENGTH,
        "degan_prior_color_anchor_strength": DEGAN_PRIOR_COLOR_ANCHOR_STRENGTH,
        "hybrid_recognition_margin": HYBRID_RECOGNITION_MARGIN,
        "strict_occlusion_blend_feather": STRICT_OCCLUSION_BLEND_FEATHER,
        "direct_panel_boundary_jump_threshold": DIRECT_PANEL_BOUNDARY_JUMP_THRESHOLD,
        "direct_panel_display_seam_pad": DIRECT_PANEL_DISPLAY_SEAM_PAD,
        "reconstruct_only_occluded_region": RECONSTRUCT_ONLY_OCCLUDED_REGION,
        "strict_display_mask_guard": STRICT_DISPLAY_MASK_GUARD,
        "strict_display_edge_cleanup": STRICT_DISPLAY_EDGE_CLEANUP,
        "degan_prior_seam_ring": DEGAN_PRIOR_SEAM_RING,
        "strict_display_seam_inpaint": STRICT_DISPLAY_SEAM_INPAINT,
        "strict_display_seam_width": STRICT_DISPLAY_SEAM_WIDTH,
        "strict_display_seam_inpaint_radius": STRICT_DISPLAY_SEAM_INPAINT_RADIUS,
        "degan_prior_shape_strength": DEGAN_PRIOR_SHAPE_STRENGTH,
        "degan_prior_shape_max_delta": DEGAN_PRIOR_SHAPE_MAX_DELTA,
        "degan_prior_shape_sigma": DEGAN_PRIOR_SHAPE_SIGMA,
        "restrict_reconstruction_mask_to_mouth_chin": RESTRICT_RECONSTRUCTION_MASK_TO_MOUTH_CHIN,
        "strict_occluder_remnant_cleanup_enabled": STRICT_OCCLUDER_REMNANT_CLEANUP_ENABLED,
        "expand_rectangular_panel_mask": EXPAND_RECTANGULAR_PANEL_MASK,
        "reconstruction_mask_erode_pixels": RECONSTRUCTION_MASK_ERODE_PIXELS,
        "panel_edge_cleanup_enabled": PANEL_EDGE_CLEANUP_ENABLED,
        "panel_edge_cleanup_outer_pad": PANEL_EDGE_CLEANUP_OUTER_PAD,
        "panel_edge_cleanup_inner_pad": PANEL_EDGE_CLEANUP_INNER_PAD,
        "panel_edge_cleanup_radius": PANEL_EDGE_CLEANUP_RADIUS,
        "panel_support_cleanup_enabled": PANEL_SUPPORT_CLEANUP_ENABLED,
        "panel_support_cleanup_x_scale": PANEL_SUPPORT_CLEANUP_X_SCALE,
        "panel_support_model_blend": PANEL_SUPPORT_MODEL_BLEND,
        "panel_boundary_blend_enabled": PANEL_BOUNDARY_BLEND_ENABLED,
        "panel_boundary_blend_radius": PANEL_BOUNDARY_BLEND_RADIUS,
        "panel_boundary_inpaint_radius": PANEL_BOUNDARY_INPAINT_RADIUS,
        "panel_detail_oval_scale_x": PANEL_DETAIL_OVAL_SCALE_X,
        "panel_detail_oval_scale_y": PANEL_DETAIL_OVAL_SCALE_Y,
        "panel_detail_alpha_blur": PANEL_DETAIL_ALPHA_BLUR,
        "panel_detail_alpha_strength": PANEL_DETAIL_ALPHA_STRENGTH,
        "panel_detail_local_base_enabled": PANEL_DETAIL_LOCAL_BASE_ENABLED,
        "panel_detail_local_base_blur_scale": PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE,
        "panel_detail_high_pass_strength": PANEL_DETAIL_HIGH_PASS_STRENGTH,
        "panel_detail_unsharp_amount": PANEL_DETAIL_UNSHARP_AMOUNT,
        "panel_detail_unsharp_sigma": PANEL_DETAIL_UNSHARP_SIGMA,
        "panel_detail_tone_match_strength": PANEL_DETAIL_TONE_MATCH_STRENGTH,
        "panel_detail_preserve_strength": PANEL_DETAIL_PRESERVE_STRENGTH,
        "panel_skin_base_l_percentile": PANEL_SKIN_BASE_L_PERCENTILE,
        "panel_skin_base_plane_strength": PANEL_SKIN_BASE_PLANE_STRENGTH,
        "panel_model_detail_tone_strength": PANEL_MODEL_DETAIL_TONE_STRENGTH,
        "panel_enhance_donor_blend_max": PANEL_ENHANCE_DONOR_BLEND_MAX,
        "panel_enhance_final_tone_strength": PANEL_ENHANCE_FINAL_TONE_STRENGTH,
        "panel_structured_repair_enabled": PANEL_STRUCTURED_REPAIR_ENABLED,
        "panel_structured_repair_score_threshold": PANEL_STRUCTURED_REPAIR_SCORE_THRESHOLD,
        "panel_structured_repair_margin": PANEL_STRUCTURED_REPAIR_MARGIN,
        "panel_skin_tone_match_enabled": PANEL_SKIN_TONE_MATCH_ENABLED,
        "panel_skin_tone_match_strength": PANEL_SKIN_TONE_MATCH_STRENGTH,
        "panel_rectangular_color_anchor_strength": PANEL_RECTANGULAR_COLOR_ANCHOR_STRENGTH,
        "panel_subtle_structure_strength": PANEL_SUBTLE_STRUCTURE_STRENGTH,
        "panel_recon_upscale_interpolation": PANEL_RECON_UPSCALE_INTERPOLATION,
        "panel_edge_inpaint_erode_size": PANEL_EDGE_INPAINT_ERODE_SIZE,
        "panel_edge_inpaint_radius": PANEL_EDGE_INPAINT_RADIUS,
        "panel_recon_region_sharpen_amount": PANEL_RECON_REGION_SHARPEN_AMOUNT,
        "panel_recon_region_sharpen_sigma": PANEL_RECON_REGION_SHARPEN_SIGMA,
        "panel_recon_region_sharpen_max_delta": PANEL_RECON_REGION_SHARPEN_MAX_DELTA,
        "panel_model_luminance_detail_amount": PANEL_MODEL_LUMINANCE_DETAIL_AMOUNT,
        "panel_model_luminance_detail_sigma": PANEL_MODEL_LUMINANCE_DETAIL_SIGMA,
        "panel_model_luminance_detail_max_delta": PANEL_MODEL_LUMINANCE_DETAIL_MAX_DELTA,
        "panel_model_luminance_structure_amount": PANEL_MODEL_LUMINANCE_STRUCTURE_AMOUNT,
        "panel_model_luminance_structure_sigma": PANEL_MODEL_LUMINANCE_STRUCTURE_SIGMA,
        "panel_model_luminance_structure_max_delta": PANEL_MODEL_LUMINANCE_STRUCTURE_MAX_DELTA,
        "panel_remnant_cleanup_enabled": PANEL_REMNANT_CLEANUP_ENABLED,
        "recognition_gallery_scoring": RECOGNITION_GALLERY_SCORING,
        "recognition_knn_k": RECOGNITION_KNN_K,
        "recognition_accept_similarity_threshold": RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD,
        "recognition_accept_gap_threshold": RECOGNITION_ACCEPT_GAP_THRESHOLD,
        "reconstruction_model_image_size": RECONSTRUCTION_MODEL_IMAGE_SIZE,
        "recognition_alignment_image_size": RECOGNITION_ALIGNMENT_IMAGE_SIZE,
    }


def _decode_upload(img_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image.")
    return img_bgr


def _rgb_image_to_face_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _normalize_mtcnn_face_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    tensor = face_tensor.float()
    # MTCNN with post_process=False always returns [0, 255] tensors.
    # Normalize to [-1, 1] unconditionally to avoid silent corruption
    # when a face happens to have pixel values all below 1.0.
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.min() >= 0.0:
        tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _prepare_face_tensor(img_bgr: np.ndarray) -> tuple[torch.Tensor, bool]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    detected_face = face_tensor is not None

    if face_tensor is None:
        resized = cv2.resize(
            img_rgb,
            (RECOGNITION_ALIGNMENT_IMAGE_SIZE, RECOGNITION_ALIGNMENT_IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        face_tensor = _rgb_image_to_face_tensor(resized).squeeze(0)
    else:
        face_tensor = _normalize_mtcnn_face_tensor(face_tensor).squeeze(0)

    return face_tensor.unsqueeze(0).to(device), detected_face


def _landmarks_to_face_coords(
    box: np.ndarray,
    landmarks: np.ndarray,
    size: int = RECOGNITION_ALIGNMENT_IMAGE_SIZE,
) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    mapped = landmarks.astype(np.float32).copy()
    mapped[:, 0] = ((mapped[:, 0] - x1) / width) * (size - 1)
    mapped[:, 1] = ((mapped[:, 1] - y1) / height) * (size - 1)
    return np.clip(mapped, 0, size - 1)


def _scale_landmarks_to_resized_image(
    landmarks: np.ndarray,
    original_shape: tuple[int, int],
    size: int = RECONSTRUCTION_MODEL_IMAGE_SIZE,
) -> np.ndarray:
    original_h, original_w = original_shape
    scaled = landmarks.astype(np.float32).copy()
    scale_x = (size - 1) / max(1.0, float(original_w - 1))
    scale_y = (size - 1) / max(1.0, float(original_h - 1))
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return np.clip(scaled, 0, size - 1)


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    return mask


def _occlusion_mask_to_tensor(
    occlusion_mask: np.ndarray, *, device_target: torch.device, dtype: torch.dtype, batch_size: int
) -> torch.Tensor:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(mask).to(device=device_target, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return tensor.expand(batch_size, -1, -1, -1)


def _harden_mask_tensor(mask_tensor: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    hard_mask = (mask_tensor > threshold).to(mask_tensor.dtype)
    hard_mask = F.max_pool2d(hard_mask, kernel_size=7, stride=1, padding=3)
    hard_mask = F.max_pool2d(hard_mask, kernel_size=5, stride=1, padding=2)
    return hard_mask.clamp(0.0, 1.0)


def _sanitize_face_tensor_for_reconstruction(face_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    hard_mask = _harden_mask_tensor(mask_tensor)
    local_mean = F.avg_pool2d(face_tensor * (1.0 - hard_mask), kernel_size=17, stride=1, padding=8)
    local_count = F.avg_pool2d(1.0 - hard_mask, kernel_size=17, stride=1, padding=8).clamp_min(1e-3)
    fill = local_mean / local_count
    fill = fill.clamp(-1.0, 1.0)
    return face_tensor * (1.0 - hard_mask) + fill * hard_mask


def _harden_mask_array(occlusion_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    mask_u8 = np.where(mask > 0.45, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        x, y, w, h = cv2.boundingRect(largest)
        if area > 24 and w > 5 and h > 5:
            aspect = w / max(1.0, float(h))
            fill_ratio = area / max(1.0, float(w * h))
            if 0.45 <= aspect <= 1.75 and fill_ratio > 0.28:
                pad = max(2, int(0.04 * max(w, h)))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(mask_u8.shape[1], x + w + pad)
                y2 = min(mask_u8.shape[0], y + h + pad)
                if y2 >= int(0.92 * mask_u8.shape[0]):
                    if x1 <= int(0.16 * mask_u8.shape[1]):
                        x1 = 0
                    if x2 >= int(0.84 * mask_u8.shape[1]):
                        x2 = mask_u8.shape[1]
                panel_mask = np.zeros_like(mask_u8)
                panel_mask[y1:y2, x1:x2] = 255
                mask_u8 = np.maximum(mask_u8, panel_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)


def _strict_occlusion_mask_array(occlusion_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    threshold = 0.45
    if mask.max() > 0.0 and np.count_nonzero(mask > threshold) < 12:
        threshold = max(0.15, float(mask.max()) * 0.55)
    mask_u8 = np.where(mask > threshold, 255, 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask_u8)
    for contour in contours:
        if cv2.contourArea(contour) >= 10:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)
    if filtered.max() == 0:
        filtered = mask_u8
    return np.clip(filtered.astype(np.float32) / 255.0, 0.0, 1.0)


def _strict_occlusion_alpha(occlusion_mask: np.ndarray) -> np.ndarray:
    strict_mask = _strict_occlusion_mask_array(occlusion_mask)
    hard = strict_mask > 0.5
    if not np.any(hard):
        return strict_mask

    feather = max(0, int(STRICT_OCCLUSION_BLEND_FEATHER))
    if feather >= 3:
        if feather % 2 == 0:
            feather += 1
        soft = cv2.GaussianBlur(hard.astype(np.float32), (feather, feather), 0)
        return np.where(hard, np.clip(soft, 0.0, 1.0), 0.0).astype(np.float32)
    return hard.astype(np.float32)


def _reconstruction_upscale_interpolation() -> int:
    if PANEL_RECON_UPSCALE_INTERPOLATION == "nearest":
        return cv2.INTER_NEAREST
    if PANEL_RECON_UPSCALE_INTERPOLATION == "linear":
        return cv2.INTER_LINEAR
    if PANEL_RECON_UPSCALE_INTERPOLATION == "cubic":
        return cv2.INTER_CUBIC
    if PANEL_RECON_UPSCALE_INTERPOLATION == "area":
        return cv2.INTER_AREA
    return cv2.INTER_LANCZOS4


def _replace_only_occluded_region(
    original_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
) -> np.ndarray:
    alpha = _strict_occlusion_alpha(occlusion_mask)[..., None]
    return np.clip(
        original_rgb.astype(np.float32) * (1.0 - alpha)
        + candidate_rgb.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _degan_prior_seam_ring_mask(mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    if hard.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return hard
    width = max(1, int(STRICT_DISPLAY_SEAM_WIDTH))
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    expanded = cv2.dilate((hard > 0.5).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    return _strict_occlusion_mask_array(expanded)


def _project_reconstruction_to_upload_resolution(
    upload_rgb: np.ndarray,
    reconstructed_model_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    display_occlusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = upload_rgb.shape[:2]
    use_direct_panel_display = False
    if display_occlusion_mask is not None and _is_usable_direct_panel_mask(display_occlusion_mask):
        mask_full = _strict_occlusion_mask_array(display_occlusion_mask).astype(np.float32)
        if mask_full.shape[:2] != (h, w):
            mask_full = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        use_direct_panel_display = _is_rectangular_panel_mask(mask_full)
    else:
        mask = _strict_occlusion_mask_array(occlusion_mask)
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    mask_full = (
        _strict_occlusion_mask_array(mask_full)
        if RECONSTRUCT_ONLY_OCCLUDED_REGION
        else _expand_mask_for_display_seam(mask_full)
    )
    direct_panel_core = _strict_occlusion_mask_array(_estimate_direct_panel_mask(upload_rgb)).astype(np.float32)
    if direct_panel_core.shape[:2] != (h, w):
        direct_panel_core = cv2.resize(direct_panel_core, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    direct_panel_core = (
        _identity_preserving_occlusion_mask(direct_panel_core)
        if RECONSTRUCT_ONLY_OCCLUDED_REGION
        else _expand_mask_for_display_seam(direct_panel_core)
    )
    if _is_usable_direct_panel_mask(direct_panel_core) and mask_full.max() > 0.5:
        direct_pixels = direct_panel_core > 0.5
        overlap = float(np.count_nonzero((mask_full > 0.5) & direct_pixels)) / max(1, int(np.count_nonzero(direct_pixels)))
        if overlap >= 0.45:
            support_from_fallback_mask = _side_support_components_from_fallback_mask(
                mask_full,
                direct_panel_core,
            )
            mask_full = direct_panel_core
            use_direct_panel_display = True
        else:
            support_from_fallback_mask = np.zeros(mask_full.shape[:2], dtype=np.float32)
    else:
        support_from_fallback_mask = np.zeros(mask_full.shape[:2], dtype=np.float32)
    protected_occlusion_mask = _identity_preserving_occlusion_mask(mask_full).astype(np.float32)
    strict_display_guard_mask = (
        protected_occlusion_mask.copy()
        if RECONSTRUCT_ONLY_OCCLUDED_REGION
        else _strict_occlusion_mask_array(mask_full).astype(np.float32)
    )
    reconstructed_full = cv2.resize(
        reconstructed_model_rgb.astype(np.uint8),
        (w, h),
        interpolation=_reconstruction_upscale_interpolation(),
    )
    edge_cleanup_mask = (
        _panel_boundary_artifact_mask(upload_rgb, mask_full)
        if PANEL_EDGE_CLEANUP_ENABLED
        else np.zeros(mask_full.shape[:2], dtype=np.float32)
    )
    support_cleanup_mask = (
        _panel_support_artifact_mask(upload_rgb, mask_full)
        if PANEL_SUPPORT_CLEANUP_ENABLED and _is_rectangular_panel_mask(mask_full)
        else np.zeros(mask_full.shape[:2], dtype=np.float32)
    )
    if not RECONSTRUCT_ONLY_OCCLUDED_REGION:
        support_cleanup_mask = np.maximum(
            support_cleanup_mask.astype(np.float32),
            support_from_fallback_mask.astype(np.float32),
        )
    if RECONSTRUCT_ONLY_OCCLUDED_REGION:
        if STRICT_OCCLUDER_REMNANT_CLEANUP_ENABLED:
            remnant_mask = _strict_occluder_remnant_mask(
                upload_rgb,
                protected_occlusion_mask,
                np.maximum(edge_cleanup_mask.astype(np.float32), support_cleanup_mask.astype(np.float32)),
            )
            protected_occlusion_mask = _strict_occlusion_mask_array(
                np.maximum(protected_occlusion_mask, remnant_mask)
            ).astype(np.float32)
        edge_cleanup_mask = np.minimum(edge_cleanup_mask.astype(np.float32), protected_occlusion_mask)
        support_cleanup_mask = np.minimum(support_cleanup_mask.astype(np.float32), protected_occlusion_mask)
    core_reconstruction_mask = np.maximum(
        mask_full.astype(np.float32),
        edge_cleanup_mask.astype(np.float32),
    )
    reconstruction_mask_full = np.maximum(
        core_reconstruction_mask,
        support_cleanup_mask.astype(np.float32),
    )
    if RECONSTRUCT_ONLY_OCCLUDED_REGION:
        core_reconstruction_mask = protected_occlusion_mask
        reconstruction_mask_full = protected_occlusion_mask
    if use_direct_panel_display:
        display_rgb = (
            _degan_prior_preserving_panel_display(
                upload_rgb,
                reconstructed_full,
                core_reconstruction_mask,
            )
            if DEGAN_PRIOR_PRESERVE_DISPLAY
            else _model_detail_panel_display(
                upload_rgb,
                reconstructed_full,
                core_reconstruction_mask,
            )
        )
        support_model_rgb = (
            (
                _degan_prior_preserving_panel_display(upload_rgb, reconstructed_full, reconstruction_mask_full)
                if DEGAN_PRIOR_PRESERVE_DISPLAY
                else _model_detail_panel_display(upload_rgb, reconstructed_full, reconstruction_mask_full)
            )
            if support_cleanup_mask.max() > 0.5
            else None
        )
        display_rgb = _inpaint_panel_support_cleanup(
            display_rgb,
            support_cleanup_mask,
            support_model_rgb,
            upload_rgb,
        )
    elif PANEL_BOUNDARY_BLEND_ENABLED:
        display_rgb = _blend_panel_with_inpainted_boundary(
            upload_rgb,
            reconstructed_full,
            reconstruction_mask_full,
        )
    else:
        alpha = reconstruction_mask_full[..., None]
        display_rgb = np.clip(
            upload_rgb.astype(np.float32) * (1.0 - alpha)
            + reconstructed_full.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
    if not DEGAN_PRIOR_PRESERVE_DISPLAY:
        display_rgb = _repair_flat_display_panel(
            upload_rgb,
            display_rgb,
            reconstruction_mask_full,
        )
    display_rgb = _remove_display_panel_edge_artifacts(display_rgb, reconstruction_mask_full, upload_rgb)
    effective_mask = reconstruction_mask_full.astype(np.float32)
    if PANEL_REMNANT_CLEANUP_ENABLED and not RECONSTRUCT_ONLY_OCCLUDED_REGION:
        display_rgb, effective_mask = _cleanup_occlusion_remnants_around_panel(
            upload_rgb,
            display_rgb,
            reconstruction_mask_full,
        )

    changed = np.abs(display_rgb.astype(np.int16) - upload_rgb.astype(np.int16)).max(axis=2) > 1
    if RECONSTRUCT_ONLY_OCCLUDED_REGION:
        effective_mask = protected_occlusion_mask
    else:
        effective_mask = np.maximum(_strict_occlusion_mask_array(effective_mask), changed.astype(np.float32))
    display_rgb = np.where(effective_mask[..., None] > 0.5, display_rgb, upload_rgb).astype(np.uint8)
    if _is_rectangular_panel_mask(reconstruction_mask_full):
        boundary_cleanup_mask = _panel_boundary_artifact_mask(upload_rgb, reconstruction_mask_full)
        if boundary_cleanup_mask.max() > 0.5:
            display_rgb, boundary_cleanup_mask = _cleanup_panel_boundary_artifacts(
                upload_rgb,
                display_rgb,
                reconstruction_mask_full,
            )
            display_rgb = _apply_rectangular_panel_skin_color_anchor(
                display_rgb,
                upload_rgb,
                reconstruction_mask_full,
                strength=(
                    min(0.88, float(np.clip(DEGAN_PRIOR_COLOR_ANCHOR_STRENGTH, 0.0, 1.0)))
                    if DEGAN_PRIOR_PRESERVE_DISPLAY
                    else 0.88
                ),
            )
            effective_mask = np.maximum(effective_mask, boundary_cleanup_mask.astype(np.float32))
        if not DEGAN_PRIOR_PRESERVE_DISPLAY:
            display_rgb = _add_subtle_rectangular_panel_structure(
                display_rgb,
                upload_rgb,
                reconstruction_mask_full,
            )
            display_rgb = _restore_model_luminance_structure(
                display_rgb,
                reconstructed_full,
                upload_rgb,
                reconstruction_mask_full,
            )
        display_rgb = _restore_model_luminance_detail(
            display_rgb,
            reconstructed_full,
            upload_rgb,
            reconstruction_mask_full,
        )
        display_rgb = _sharpen_reconstructed_region_naturally(display_rgb, reconstruction_mask_full)
        if DEGAN_PRIOR_PRESERVE_DISPLAY:
            display_rgb = _restore_degan_luminance_shape(
                display_rgb,
                reconstructed_full,
                reconstruction_mask_full,
            )
    if STRICT_DISPLAY_MASK_GUARD:
        effective_mask = strict_display_guard_mask.astype(np.float32)
        display_rgb = np.where(effective_mask[..., None] > 0.5, display_rgb, upload_rgb).astype(np.uint8)
        if DEGAN_PRIOR_PRESERVE_DISPLAY and DEGAN_PRIOR_SEAM_RING and _is_rectangular_panel_mask(effective_mask):
            seam_ring_mask = _degan_prior_seam_ring_mask(effective_mask)
            if np.count_nonzero(seam_ring_mask > 0.5) > np.count_nonzero(effective_mask > 0.5):
                seam_display_rgb = _degan_prior_preserving_panel_display(
                    upload_rgb,
                    reconstructed_full,
                    seam_ring_mask,
                )
                seam_display_rgb = _restore_model_luminance_detail(
                    seam_display_rgb,
                    reconstructed_full,
                    upload_rgb,
                    seam_ring_mask,
                )
                seam_display_rgb = _restore_degan_luminance_shape(
                    seam_display_rgb,
                    reconstructed_full,
                    seam_ring_mask,
                )
                effective_mask = seam_ring_mask.astype(np.float32)
                display_rgb = np.where(effective_mask[..., None] > 0.5, seam_display_rgb, upload_rgb).astype(np.uint8)
        if _is_rectangular_panel_mask(effective_mask):
            display_rgb = _remove_display_panel_edge_artifacts(display_rgb, effective_mask, upload_rgb)
        if STRICT_DISPLAY_EDGE_CLEANUP and _is_rectangular_panel_mask(effective_mask):
            display_rgb, strict_edge_cleanup_mask = _cleanup_panel_boundary_artifacts(
                upload_rgb,
                display_rgb,
                effective_mask,
            )
            if strict_edge_cleanup_mask.max() > 0.5:
                effective_mask = np.maximum(effective_mask, strict_edge_cleanup_mask.astype(np.float32))
        display_rgb, strict_seam_mask = _inpaint_strict_panel_seam(upload_rgb, display_rgb, effective_mask)
        if strict_seam_mask.max() > 0.5:
            effective_mask = np.maximum(effective_mask, strict_seam_mask.astype(np.float32))
    return display_rgb, effective_mask


def _expand_mask_for_display_seam(mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask)
    if hard.max() <= 0.0:
        return hard
    seam_pad = DIRECT_PANEL_DISPLAY_SEAM_PAD
    if seam_pad <= 0:
        return hard
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate((hard > 0.5).astype(np.uint8), kernel, iterations=seam_pad)
    return expanded.astype(np.float32)


def _expand_rectangular_panel_mask(mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask)
    if not EXPAND_RECTANGULAR_PANEL_MASK:
        return hard
    if not _is_rectangular_panel_mask(hard):
        return hard
    bbox = _mask_bbox(hard)
    if bbox is None:
        return hard

    x1, y1, x2, y2 = bbox
    h, w = hard.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    expand_top = max(2, int(round(0.20 * bh)))
    expand_bottom = max(1, int(round(0.04 * bh)))
    expand_side = max(1, int(round(0.03 * bw)))
    expanded = hard.copy()
    expanded[
        max(0, y1 - expand_top):min(h, y2 + expand_bottom),
        max(0, x1 - expand_side):min(w, x2 + expand_side),
    ] = 1.0
    return _strict_occlusion_mask_array(expanded)


def _identity_preserving_occlusion_mask(mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    if RESTRICT_RECONSTRUCTION_MASK_TO_MOUTH_CHIN and _is_rectangular_panel_mask(hard):
        hard = _restrict_mask_to_mouth_chin_region(hard)
    if hard.max() <= 0.0 or RECONSTRUCTION_MASK_ERODE_PIXELS <= 0:
        return hard

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(
        (hard > 0.5).astype(np.uint8),
        kernel,
        iterations=RECONSTRUCTION_MASK_ERODE_PIXELS,
    ).astype(np.float32)
    original_area = float(np.count_nonzero(hard > 0.5))
    eroded_area = float(np.count_nonzero(eroded > 0.5))
    if eroded_area < 20.0 or eroded_area < original_area * 0.65:
        return hard
    return eroded


def _restrict_mask_to_mouth_chin_region(mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    if hard.max() <= 0.0:
        return hard

    h, w = hard.shape[:2]
    roi = np.zeros_like(hard, dtype=np.float32)
    y1 = max(0, int(round(0.30 * h)))
    y2 = min(h, int(round(0.84 * h)))
    x1 = max(0, int(round(0.10 * w)))
    x2 = min(w, int(round(0.90 * w)))
    roi[y1:y2, x1:x2] = 1.0
    restricted = _strict_occlusion_mask_array(hard * roi).astype(np.float32)

    original_area = float(np.count_nonzero(hard > 0.5))
    restricted_area = float(np.count_nonzero(restricted > 0.5))
    if restricted_area < 12.0:
        return hard
    if original_area > 0.0 and restricted_area < original_area * 0.18:
        return hard
    return restricted


def _strict_occluder_remnant_mask(
    upload_rgb: np.ndarray,
    core_mask: np.ndarray,
    evidence_mask: np.ndarray,
) -> np.ndarray:
    core = _strict_occlusion_mask_array(core_mask) > 0.5
    evidence = _strict_occlusion_mask_array(evidence_mask) > 0.5
    bbox = _mask_bbox(core.astype(np.float32))
    if bbox is None or core.sum() < 20:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    x1, y1, x2, y2 = bbox
    h, w = core.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(0.12 * bw)) | 1), max(3, int(round(0.16 * bh)) | 1)),
    )
    near_core = cv2.dilate(core.astype(np.uint8), ring_kernel, iterations=1).astype(bool) & ~core
    if not np.any(near_core):
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    panel_pixels = upload_rgb[core]
    if panel_pixels.size == 0:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)
    target_rgb = np.median(panel_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    hsv = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    target_ycrcb = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2YCrCb)[0, 0].astype(np.float32)
    gray = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)

    rgb_distance = np.linalg.norm(upload_rgb.astype(np.float32) - target_rgb, axis=2)
    ycrcb_distance = np.linalg.norm(ycrcb.astype(np.float32) - target_ycrcb, axis=2)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    panel_like = (
        (saturation < 118)
        & (value > 78)
        & (texture < 42.0)
        & ((rgb_distance < 92.0) | (ycrcb_distance < 72.0) | ((saturation < 48) & (value > 142)))
    )
    candidate = near_core & panel_like & (evidence | (rgb_distance < 62.0) | ((saturation < 38) & (value > 155)))
    if not np.any(candidate):
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, 8)
    filtered = np.zeros_like(candidate_u8)
    max_area = max(80, int(0.18 * np.count_nonzero(core)))
    for label_idx in range(1, num_labels):
        cx, cy, cw, ch, area = stats[label_idx]
        if area < 2 or area > max_area:
            continue
        touches = np.any(
            cv2.dilate((labels == label_idx).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            & core
        )
        if not touches:
            continue
        center_y = cy + 0.5 * ch
        if center_y < y1 - 0.15 * bh or center_y > y2 + 0.18 * bh:
            continue
        filtered[labels == label_idx] = 255

    return (filtered.astype(np.float32) / 255.0).astype(np.float32)


def _blend_panel_with_inpainted_boundary(
    upload_rgb: np.ndarray,
    reconstructed_full: np.ndarray,
    mask_full: np.ndarray,
) -> np.ndarray:
    """Use DE-GAN in the center and inpainted context at the panel boundary."""
    hard = (_strict_occlusion_mask_array(mask_full) > 0.5).astype(np.uint8)
    if hard.sum() < 20:
        return upload_rgb.copy()
    if _is_rectangular_panel_mask(hard.astype(np.float32)):
        candidate_rgb = _detail_preserving_reconstruction_candidate(
            upload_rgb.astype(np.uint8),
            reconstructed_full.astype(np.uint8),
            hard.astype(np.float32),
        )
        output_rgb = _replace_only_occluded_region(
            upload_rgb.astype(np.uint8),
            candidate_rgb,
            hard.astype(np.float32),
        )
        if PANEL_SKIN_TONE_MATCH_ENABLED:
            output_rgb = _apply_local_skin_tone_match(
                output_rgb,
                upload_rgb,
                hard.astype(np.float32),
                strength=max(PANEL_MODEL_DETAIL_TONE_STRENGTH, 0.72),
            )
            output_rgb = _apply_rectangular_panel_skin_color_anchor(
                output_rgb,
                upload_rgb,
                hard.astype(np.float32),
            )
        return output_rgb

    mask_u8 = hard * 255
    inpainted_bgr = cv2.inpaint(
        cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
        mask_u8,
        PANEL_BOUNDARY_INPAINT_RADIUS,
        cv2.INPAINT_TELEA,
    )
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    context_base_rgb = (
        _smooth_panel_skin_base(upload_rgb, inpainted_rgb, hard.astype(np.float32))
        if PANEL_DETAIL_LOCAL_BASE_ENABLED
        else inpainted_rgb
    )
    detail_source_rgb = (
        _rebase_panel_detail_to_local_context(
            reconstructed_full,
            context_base_rgb,
            hard.astype(np.float32),
        )
        if PANEL_DETAIL_LOCAL_BASE_ENABLED
        else reconstructed_full
    )
    alpha = _soft_panel_detail_alpha(hard)
    output = upload_rgb.astype(np.float32).copy()
    target = hard > 0
    output[target] = (
        context_base_rgb.astype(np.float32)[target] * (1.0 - alpha[target, None])
        + detail_source_rgb.astype(np.float32)[target] * alpha[target, None]
    )
    output_rgb = np.clip(output, 0, 255).astype(np.uint8)
    if PANEL_SKIN_TONE_MATCH_ENABLED:
        output_rgb = _apply_local_skin_tone_match(
            output_rgb,
            upload_rgb,
            hard.astype(np.float32),
            strength=PANEL_SKIN_TONE_MATCH_STRENGTH,
        )
    return output_rgb


def _model_detail_panel_display(
    upload_rgb: np.ndarray,
    reconstructed_full: np.ndarray,
    mask_full: np.ndarray,
) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask_full).astype(np.float32)
    candidate_rgb = _detail_preserving_reconstruction_candidate(
        upload_rgb.astype(np.uint8),
        reconstructed_full.astype(np.uint8),
        hard,
    )
    rectangular_panel = _is_rectangular_panel_mask(hard)
    if rectangular_panel:
        cleanup_mask = _panel_boundary_artifact_mask(upload_rgb, hard)
        repair_mask = _strict_occlusion_mask_array(np.maximum(hard, cleanup_mask))
        repair_base = _skin_prior_repair_region(upload_rgb.astype(np.uint8), repair_mask)
        candidate_rgb = _rebase_panel_detail_to_local_context(candidate_rgb, repair_base, hard)
        candidate_rgb = _apply_local_skin_tone_match(
            candidate_rgb,
            upload_rgb,
            hard,
            strength=max(PANEL_MODEL_DETAIL_TONE_STRENGTH, 0.72),
        )
        candidate_rgb = _boost_reconstructed_panel_detail(candidate_rgb, hard)
        candidate_rgb = _match_panel_boundary_tone_preserving_detail(candidate_rgb, upload_rgb, hard)
    output_rgb = _replace_only_occluded_region(
        upload_rgb.astype(np.uint8),
        candidate_rgb,
        hard,
    )
    if rectangular_panel:
        output_rgb = _apply_rectangular_panel_skin_color_anchor(output_rgb, upload_rgb, hard)
    if PANEL_SKIN_TONE_MATCH_ENABLED and not rectangular_panel:
        output_rgb = _apply_local_skin_tone_match(
            output_rgb,
            upload_rgb,
            hard,
            strength=PANEL_MODEL_DETAIL_TONE_STRENGTH,
        )
    return output_rgb


def _degan_prior_preserving_panel_display(
    upload_rgb: np.ndarray,
    reconstructed_full: np.ndarray,
    mask_full: np.ndarray,
) -> np.ndarray:
    """Paste the DE-GAN prior more directly for diagnosis/identity preservation.

    The default rectangular-panel path rebases model detail onto a local skin
    prior. That improves color continuity but can wash out the identity cues
    already present in DE-GAN. This optional path keeps DE-GAN as the primary
    source inside the hard occlusion mask and applies only mild local tone
    matching plus detail sharpening.
    """
    hard = _strict_occlusion_mask_array(mask_full).astype(np.float32)
    if hard.sum() < 20:
        return upload_rgb.astype(np.uint8).copy()

    candidate_rgb = _suppress_reconstructed_panel_leak(
        upload_rgb.astype(np.uint8),
        reconstructed_full.astype(np.uint8),
        hard,
    )
    candidate_rgb = _boost_reconstructed_panel_detail(candidate_rgb, hard)

    tone_strength = float(np.clip(DEGAN_PRIOR_TONE_MATCH_STRENGTH, 0.0, 1.0))
    if PANEL_SKIN_TONE_MATCH_ENABLED and tone_strength > 0.0:
        candidate_rgb = _apply_local_skin_tone_match(
            candidate_rgb,
            upload_rgb.astype(np.uint8),
            hard,
            strength=tone_strength,
        )

    output_rgb = _replace_only_occluded_region(
        upload_rgb.astype(np.uint8),
        candidate_rgb,
        hard,
    )
    anchor_strength = float(np.clip(DEGAN_PRIOR_COLOR_ANCHOR_STRENGTH, 0.0, 1.0))
    if anchor_strength > 0.0:
        output_rgb = _apply_rectangular_panel_skin_color_anchor(
            output_rgb,
            upload_rgb.astype(np.uint8),
            hard,
            strength=anchor_strength,
        )
    output_rgb = _restore_degan_luminance_shape(output_rgb, reconstructed_full.astype(np.uint8), hard)
    return output_rgb


def _inpaint_panel_support_cleanup(
    display_rgb: np.ndarray,
    support_mask: np.ndarray,
    support_model_rgb: np.ndarray | None = None,
    original_rgb: np.ndarray | None = None,
) -> np.ndarray:
    hard = _strict_occlusion_mask_array(support_mask) > 0.5
    if hard.sum() < 2:
        return display_rgb

    mask_u8 = np.where(hard, 255, 0).astype(np.uint8)
    mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    inpainted_bgr = cv2.inpaint(
        cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
        mask_u8,
        3,
        cv2.INPAINT_TELEA,
    )
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    fill_rgb = inpainted_rgb
    if original_rgb is not None:
        skin_repair = _skin_prior_repair_region(original_rgb.astype(np.uint8), hard.astype(np.float32))
        skin_repair = _apply_local_skin_tone_match(
            skin_repair,
            original_rgb.astype(np.uint8),
            hard.astype(np.float32),
            strength=0.88,
        )
        fill_rgb = skin_repair
    if support_model_rgb is not None:
        model_blend = float(np.clip(PANEL_SUPPORT_MODEL_BLEND, 0.0, 1.0))
        fill_rgb = np.clip(
            support_model_rgb.astype(np.float32) * model_blend
            + fill_rgb.astype(np.float32) * (1.0 - model_blend),
            0,
            255,
        ).astype(np.uint8)
    cleaned = display_rgb.copy()
    cleaned[mask_u8 > 0] = fill_rgb[mask_u8 > 0]
    return cleaned


def _repair_flat_display_panel(
    upload_rgb: np.ndarray,
    display_rgb: np.ndarray,
    mask_full: np.ndarray,
) -> np.ndarray:
    if not PANEL_STRUCTURED_REPAIR_ENABLED:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask_full)
    if not _is_rectangular_panel_mask(hard):
        return display_rgb

    current_score = _rectangular_repair_score(display_rgb, upload_rgb, hard)
    if current_score < PANEL_STRUCTURED_REPAIR_SCORE_THRESHOLD:
        return display_rgb

    candidates = [
        display_rgb,
        _skin_prior_repair_region(upload_rgb.astype(np.uint8), hard),
        _symmetry_repair_region(upload_rgb.astype(np.uint8), hard),
        _inpaint_occlusion_region(upload_rgb.astype(np.uint8), hard),
    ]
    best = min(candidates, key=lambda candidate: _rectangular_repair_score(candidate, upload_rgb, hard))
    best_score = _rectangular_repair_score(best, upload_rgb, hard)
    if best is display_rgb or best_score + PANEL_STRUCTURED_REPAIR_MARGIN >= current_score:
        return display_rgb

    return _blend_repair_inside_mask(upload_rgb.astype(np.uint8), best.astype(np.uint8), hard)


def _blend_repair_inside_mask(original_rgb: np.ndarray, candidate_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    hard = (_strict_occlusion_mask_array(hard_mask) > 0.5).astype(np.float32)
    if hard.sum() < 20:
        return original_rgb.copy()

    alpha = cv2.GaussianBlur(hard, (11, 11), 0)
    alpha = np.clip(alpha, 0.0, 0.98) * hard
    return np.clip(
        original_rgb.astype(np.float32) * (1.0 - alpha[..., None])
        + candidate_rgb.astype(np.float32) * alpha[..., None],
        0,
        255,
    ).astype(np.uint8)


def _smooth_panel_skin_base(
    original_rgb: np.ndarray,
    fallback_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    """Build a smooth skin-colored base without radial inpaint artifacts."""
    hard = _strict_occlusion_mask_array(hard_mask) > 0.5
    if hard.sum() < 20:
        return fallback_rgb

    donor_mask = _panel_skin_context_mask(original_rgb, hard_mask)
    if donor_mask.sum() < 35:
        donor_mask = _local_skin_context_mask(original_rgb, hard_mask)
    if donor_mask.sum() < 35:
        return fallback_rgb

    h, w = hard.shape[:2]
    original_lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(fallback_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    donor_pixels = original_lab[donor_mask]
    donor_y, donor_x = np.where(donor_mask)
    mask_y, mask_x = np.where(hard)
    if donor_pixels.shape[0] < 35 or mask_x.size == 0:
        return fallback_rgb
    donor_l_all = donor_pixels[:, 0]
    bright_donor = donor_l_all >= float(np.percentile(donor_l_all, 55))
    color_donor_pixels = donor_pixels[bright_donor] if bright_donor.sum() >= 20 else donor_pixels

    donor_coords = np.column_stack(
        [
            donor_x.astype(np.float32) / max(1.0, float(w - 1)),
            donor_y.astype(np.float32) / max(1.0, float(h - 1)),
            np.ones(donor_x.shape[0], dtype=np.float32),
        ]
    )
    mask_coords = np.column_stack(
        [
            mask_x.astype(np.float32) / max(1.0, float(w - 1)),
            mask_y.astype(np.float32) / max(1.0, float(h - 1)),
            np.ones(mask_x.shape[0], dtype=np.float32),
        ]
    )

    filled_lab = base_lab.copy()
    for channel_idx in range(3):
        values = donor_pixels[:, channel_idx] if channel_idx == 0 else color_donor_pixels[:, channel_idx]
        low = float(np.percentile(values, 12))
        high = float(np.percentile(values, 92))
        median = float(np.median(values))
        keep = (values >= low) & (values <= high)
        if channel_idx == 0 and keep.sum() >= 12:
            coeffs, *_ = np.linalg.lstsq(donor_coords[keep], values[keep], rcond=None)
            predicted = mask_coords @ coeffs
            target_l = float(np.percentile(values, np.clip(PANEL_SKIN_BASE_L_PERCENTILE, 40.0, 80.0)))
            plane_strength = float(np.clip(PANEL_SKIN_BASE_PLANE_STRENGTH, 0.0, 0.75))
            predicted = predicted * plane_strength + target_l * (1.0 - plane_strength)
            predicted = np.clip(predicted, max(0.0, low - 4.0), min(255.0, high + 10.0))
        else:
            predicted = np.full(mask_x.shape[0], median, dtype=np.float32)
            predicted = np.clip(predicted, max(0.0, low - 6.0), min(255.0, high + 6.0))
        filled_lab[mask_y, mask_x, channel_idx] = predicted.astype(np.float32)

    filled_rgb = cv2.cvtColor(np.clip(filled_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    output = fallback_rgb.copy()
    output[hard] = filled_rgb[hard]
    return output


def _rebase_panel_detail_to_local_context(
    reconstructed_rgb: np.ndarray,
    local_base_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    """Keep DE-GAN detail while replacing its low-frequency face wash."""
    hard = _strict_occlusion_mask_array(hard_mask) > 0.5
    bbox = _mask_bbox(hard.astype(np.float32))
    if bbox is None or hard.sum() < 20:
        return reconstructed_rgb

    x1, y1, x2, y2 = bbox
    span = max(1, x2 - x1, y2 - y1)
    blur_size = int(max(21, round(span * PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE)))
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = min(blur_size, 71)

    reconstructed_f = reconstructed_rgb.astype(np.float32)
    base_f = local_base_rgb.astype(np.float32)
    reconstructed_low = cv2.GaussianBlur(reconstructed_f, (blur_size, blur_size), 0)
    base_low = cv2.GaussianBlur(base_f, (blur_size, blur_size), 0)
    high_pass = reconstructed_f - reconstructed_low
    detail_strength = float(np.clip(PANEL_DETAIL_HIGH_PASS_STRENGTH, 0.0, 1.25))
    rebased = np.clip(base_low + high_pass * detail_strength, 0, 255).astype(np.uint8)
    output = reconstructed_rgb.copy()
    output[hard] = rebased[hard]
    return output


def _boost_reconstructed_panel_detail(candidate_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    hard = (_strict_occlusion_mask_array(hard_mask) > 0.5).astype(np.float32)
    if hard.sum() < 20:
        return candidate_rgb

    amount = float(np.clip(PANEL_DETAIL_UNSHARP_AMOUNT, 0.0, 1.25))
    if amount <= 0.0:
        return candidate_rgb

    sigma = max(0.2, float(PANEL_DETAIL_UNSHARP_SIGMA))
    candidate_f = candidate_rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(candidate_f, (0, 0), sigma)
    sharpened = np.clip(candidate_f + (candidate_f - blurred) * amount, 0, 255)
    alpha = cv2.GaussianBlur(hard, (7, 7), 0) * hard
    return np.clip(candidate_f * (1.0 - alpha[..., None]) + sharpened * alpha[..., None], 0, 255).astype(np.uint8)


def _match_panel_tone_preserving_detail(
    candidate_rgb: np.ndarray,
    original_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    hard = (_strict_occlusion_mask_array(hard_mask) > 0.5).astype(np.float32)
    bbox = _mask_bbox(hard)
    if bbox is None or hard.sum() < 20:
        return candidate_rgb

    tone_strength = float(np.clip(PANEL_DETAIL_TONE_MATCH_STRENGTH, 0.0, 1.0))
    detail_strength = float(np.clip(PANEL_DETAIL_PRESERVE_STRENGTH, 0.0, 1.25))
    if tone_strength <= 0.0:
        return candidate_rgb

    x1, y1, x2, y2 = bbox
    span = max(1, x2 - x1, y2 - y1)
    blur_size = int(max(15, round(span * PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE)))
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = min(blur_size, 51)

    tone_matched = _apply_local_skin_tone_match(
        candidate_rgb,
        original_rgb,
        hard,
        strength=tone_strength,
    )
    candidate_f = candidate_rgb.astype(np.float32)
    tone_f = tone_matched.astype(np.float32)
    candidate_low = cv2.GaussianBlur(candidate_f, (blur_size, blur_size), 0)
    tone_low = cv2.GaussianBlur(tone_f, (blur_size, blur_size), 0)
    detail = candidate_f - candidate_low
    rebuilt = np.clip(tone_low + detail * detail_strength, 0, 255)
    alpha = cv2.GaussianBlur(hard, (7, 7), 0) * hard
    return np.clip(candidate_f * (1.0 - alpha[..., None]) + rebuilt * alpha[..., None], 0, 255).astype(np.uint8)


def _match_panel_boundary_tone_preserving_detail(
    candidate_rgb: np.ndarray,
    original_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    hard = (_strict_occlusion_mask_array(hard_mask) > 0.5).astype(np.float32)
    bbox = _mask_bbox(hard)
    if bbox is None or hard.sum() < 20:
        return candidate_rgb

    donor_mask = _panel_boundary_skin_context_mask(original_rgb, hard)
    if donor_mask.sum() < 35:
        return candidate_rgb

    x1, y1, x2, y2 = bbox
    span = max(1, x2 - x1, y2 - y1)
    blur_size = int(max(15, round(span * PANEL_DETAIL_LOCAL_BASE_BLUR_SCALE)))
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = min(blur_size, 51)

    hard_u8 = hard.astype(np.uint8)
    inner = cv2.erode(hard_u8, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if inner.sum() < 20:
        inner = hard > 0.5

    candidate_lab = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    source_pixels = candidate_lab[inner]
    target_pixels = original_lab[donor_mask]
    if source_pixels.shape[0] < 20 or target_pixels.shape[0] < 35:
        return candidate_rgb

    target_l = target_pixels[:, 0]
    target_core = target_pixels[
        (target_l >= float(np.percentile(target_l, 15)))
        & (target_l <= float(np.percentile(target_l, 90)))
    ]
    if target_core.shape[0] >= 20:
        target_pixels = target_core

    source_center, source_scale = _robust_channel_stats(source_pixels)
    target_center, target_scale = _robust_channel_stats(target_pixels)
    target_center[0] = float(np.percentile(target_pixels[:, 0], 52))
    low = cv2.GaussianBlur(candidate_lab, (blur_size, blur_size), 0)
    matched_low = low.copy()
    channel_strengths = np.array([0.82, 0.96, 0.96], dtype=np.float32)
    for channel_idx, strength in enumerate(channel_strengths):
        normalized = (low[:, :, channel_idx] - source_center[channel_idx]) / source_scale[channel_idx]
        matched = normalized * target_scale[channel_idx] + target_center[channel_idx]
        matched_low[:, :, channel_idx] = low[:, :, channel_idx] * (1.0 - strength) + matched * strength

    target_low = np.percentile(target_pixels, 5, axis=0)
    target_high = np.percentile(target_pixels, 95, axis=0)
    matched_low[:, :, 0] = np.clip(matched_low[:, :, 0], max(0.0, target_low[0] - 10.0), min(255.0, target_high[0] + 14.0))
    matched_low[:, :, 1] = np.clip(matched_low[:, :, 1], max(0.0, target_low[1] - 7.0), min(255.0, target_high[1] + 7.0))
    matched_low[:, :, 2] = np.clip(matched_low[:, :, 2], max(0.0, target_low[2] - 7.0), min(255.0, target_high[2] + 7.0))

    detail = candidate_lab - low
    detail_strength = np.array([0.84, 0.18, 0.18], dtype=np.float32)
    rebuilt = np.clip(matched_low + detail * detail_strength, 0, 255)
    alpha = cv2.GaussianBlur(hard, (7, 7), 0) * hard
    output_lab = candidate_lab * (1.0 - alpha[..., None]) + rebuilt * alpha[..., None]
    return cv2.cvtColor(np.clip(output_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _soft_panel_detail_alpha(hard_mask: np.ndarray) -> np.ndarray:
    bbox = _mask_bbox(hard_mask.astype(np.float32))
    alpha = np.zeros(hard_mask.shape[:2], dtype=np.float32)
    if bbox is None:
        return alpha

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    axes = (
        max(1, int(round(bw * PANEL_DETAIL_OVAL_SCALE_X / 2.0))),
        max(1, int(round(bh * PANEL_DETAIL_OVAL_SCALE_Y / 2.0))),
    )
    cv2.ellipse(alpha, center, axes, 0, 0, 360, 1.0, thickness=-1)
    blur = PANEL_DETAIL_ALPHA_BLUR
    if blur % 2 == 0:
        blur += 1
    if blur > 1:
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)
    alpha = np.clip(alpha, 0.0, 1.0) * (hard_mask > 0).astype(np.float32)
    return np.clip(alpha * PANEL_DETAIL_ALPHA_STRENGTH, 0.0, 1.0)


def _cleanup_panel_boundary_artifacts(
    upload_rgb: np.ndarray,
    display_rgb: np.ndarray,
    core_mask: np.ndarray,
    repair_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint only the synthetic panel seam and straps around the DE-GAN patch."""
    cleanup_mask = _panel_boundary_artifact_mask(upload_rgb, core_mask)
    cleanup_u8 = np.where(cleanup_mask > 0.5, 255, 0).astype(np.uint8)
    if cleanup_u8.max() == 0:
        return display_rgb, cleanup_mask

    replace = cleanup_u8 > 0
    output_rgb = display_rgb.copy()
    if repair_rgb is not None:
        output_rgb[replace] = repair_rgb.astype(np.uint8)[replace]
    else:
        cleaned_bgr = cv2.inpaint(
            cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
            cleanup_u8,
            PANEL_EDGE_CLEANUP_RADIUS,
            cv2.INPAINT_TELEA,
        )
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        output_rgb[replace] = cleaned_rgb[replace]
    return output_rgb, cleanup_mask


def _inpaint_strict_panel_seam(
    original_rgb: np.ndarray,
    display_rgb: np.ndarray,
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not STRICT_DISPLAY_SEAM_INPAINT:
        return display_rgb, np.zeros(core_mask.shape[:2], dtype=np.float32)

    hard = _strict_occlusion_mask_array(core_mask).astype(np.float32)
    hard_bool = hard > 0.5
    if hard_bool.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return display_rgb, np.zeros(core_mask.shape[:2], dtype=np.float32)

    width = max(1, int(STRICT_DISPLAY_SEAM_WIDTH))
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    dilated = cv2.dilate(hard_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded = cv2.erode(hard_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    seam = dilated & ~eroded
    if np.count_nonzero(seam) < 8:
        return display_rgb, np.zeros(core_mask.shape[:2], dtype=np.float32)

    seam_mask = seam.astype(np.float32)
    skin_repair = _skin_prior_repair_region(display_rgb.astype(np.uint8), seam_mask)
    skin_repair = _apply_local_skin_tone_match(
        skin_repair,
        original_rgb.astype(np.uint8),
        seam_mask,
        strength=0.45,
    )
    cleaned = display_rgb.copy()
    cleaned[seam] = skin_repair[seam]
    return cleaned, seam_mask


def _panel_boundary_artifact_mask(upload_rgb: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(core_mask) > 0.5
    if hard.sum() < 20:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    outer_pad = int(PANEL_EDGE_CLEANUP_OUTER_PAD)
    inner_pad = int(PANEL_EDGE_CLEANUP_INNER_PAD)
    if outer_pad <= 0 and inner_pad <= 0:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    panel_pixels = upload_rgb[hard]
    if panel_pixels.size == 0:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    target_rgb = np.median(panel_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    hsv = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    target_ycrcb = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2YCrCb)[0, 0].astype(np.float32)
    rgb_distance = np.linalg.norm(upload_rgb.astype(np.float32) - target_rgb, axis=2)
    ycrcb_distance = np.linalg.norm(ycrcb.astype(np.float32) - target_ycrcb, axis=2)

    if outer_pad > 0:
        outer_kernel = np.ones((outer_pad * 2 + 1, outer_pad * 2 + 1), np.uint8)
        search_region = cv2.dilate(hard.astype(np.uint8), outer_kernel, iterations=1).astype(bool)
    else:
        search_region = hard.copy()
    outer_ring = search_region & ~hard

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    occluder_like = (
        (saturation < 130)
        & (value > 60)
        & ((rgb_distance < 125.0) | (ycrcb_distance < 90.0))
    )
    bright_flat_edge = (saturation < 95) & (value > 120)
    external_cleanup = outer_ring & (occluder_like | bright_flat_edge)

    if inner_pad > 0:
        inner_kernel = np.ones((inner_pad * 2 + 1, inner_pad * 2 + 1), np.uint8)
        eroded = cv2.erode(hard.astype(np.uint8), inner_kernel, iterations=1).astype(bool)
        internal_cleanup = hard & ~eroded
    else:
        internal_cleanup = np.zeros_like(hard, dtype=bool)

    cleanup_u8 = np.where(external_cleanup | internal_cleanup, 255, 0).astype(np.uint8)
    cleanup_u8 = cv2.morphologyEx(cleanup_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cleanup_u8 = np.where((cleanup_u8 > 0) & search_region, 255, 0).astype(np.uint8)
    return cleanup_u8.astype(np.float32) / 255.0


def _panel_support_artifact_mask(upload_rgb: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(core_mask) > 0.5
    bbox = _mask_bbox(hard.astype(np.float32))
    if bbox is None or hard.sum() < 20:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    x1, y1, x2, y2 = bbox
    h, w = hard.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    panel_pixels = upload_rgb[hard]
    if panel_pixels.size == 0:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)

    target_rgb = np.median(panel_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    hsv = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    target_ycrcb = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2YCrCb)[0, 0].astype(np.float32)
    rgb_distance = np.linalg.norm(upload_rgb.astype(np.float32) - target_rgb, axis=2)
    ycrcb_distance = np.linalg.norm(ycrcb.astype(np.float32) - target_ycrcb, axis=2)

    x_scale = float(np.clip(PANEL_SUPPORT_CLEANUP_X_SCALE, 0.15, 0.85))
    search = np.zeros(hard.shape, dtype=bool)
    y_top = max(0, int(y1 - 0.18 * bh))
    y_bottom = min(h, int(y2 + 0.28 * bh))
    search[y_top:y_bottom, max(0, int(x1 - x_scale * bw)):min(w, int(x1 + 0.18 * bw))] = True
    search[y_top:y_bottom, max(0, int(x2 - 0.18 * bw)):min(w, int(x2 + x_scale * bw))] = True

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    candidate = (
        search
        & ~hard
        & (saturation < 105)
        & (value > 85)
        & (
            (rgb_distance < 92.0)
            | (ycrcb_distance < 72.0)
            | ((saturation < 55) & (value > 150))
        )
    )
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(candidate_u8)
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        component_pixels = int(np.count_nonzero(candidate_u8[cy : cy + ch, cx : cx + cw]))
        if component_pixels < 2 or component_pixels > max(220.0, 0.01 * h * w):
            continue
        center_y = cy + 0.5 * ch
        if center_y < y1 + 0.28 * bh or center_y > y1 + 0.85 * bh:
            continue
        near_left = cx <= x1 + max(4, int(0.08 * bw)) and cx + cw >= x1 - int(x_scale * bw)
        near_right = cx + cw >= x2 - max(4, int(0.08 * bw)) and cx <= x2 + int(x_scale * bw)
        if near_left or near_right:
            roi = filtered[cy : cy + ch, cx : cx + cw]
            filtered[cy : cy + ch, cx : cx + cw] = np.maximum(roi, candidate_u8[cy : cy + ch, cx : cx + cw])

    gray = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 35, 100)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1) > 0
    line_like = (saturation < 145) & (value > 105)
    line_thickness = max(3, int(round(0.07 * bh)))
    for start, end in (
        (
            (int(x1), int(y1 + 0.66 * bh)),
            (int(x1 - 0.20 * bw), int(y1 + 0.50 * bh)),
        ),
        (
            (int(x2), int(y1 + 0.66 * bh)),
            (int(x2 + 0.20 * bw), int(y1 + 0.50 * bh)),
        ),
    ):
        line_u8 = np.zeros_like(candidate_u8)
        cv2.line(line_u8, start, end, 255, line_thickness)
        line_strip = (line_u8 > 0) & search & ~hard
        if line_strip.sum() < 4:
            continue
        evidence = line_strip & line_like & edge
        if evidence.sum() < max(8, int(0.08 * line_strip.sum())):
            continue
        filtered[line_strip & line_like] = 255

    if filtered.max() == 0:
        return np.zeros(core_mask.shape[:2], dtype=np.float32)
    filtered = cv2.dilate(filtered, np.ones((3, 3), np.uint8), iterations=1)
    return np.where((filtered > 0) & search, 1.0, 0.0).astype(np.float32)


def _side_support_components_from_fallback_mask(fallback_mask: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    core = _strict_occlusion_mask_array(core_mask) > 0.5
    fallback = _strict_occlusion_mask_array(fallback_mask) > 0.5
    bbox = _mask_bbox(core.astype(np.float32))
    if bbox is None:
        return np.zeros(fallback_mask.shape[:2], dtype=np.float32)

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    core_area = max(1, int(np.count_nonzero(core)))
    extra = fallback & ~core
    if extra.sum() < 2:
        return np.zeros(fallback_mask.shape[:2], dtype=np.float32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
    filtered = np.zeros(extra.shape, dtype=np.uint8)
    max_area = max(260, int(0.16 * core_area))
    for label_idx in range(1, num_labels):
        cx, cy, cw, ch, area = stats[label_idx]
        if area < 2 or area > max_area:
            continue
        center_y = cy + 0.5 * ch
        if center_y < y1 + 0.20 * bh or center_y > y1 + 0.90 * bh:
            continue
        reaches_left_side = cx <= x1 + max(4, int(0.10 * bw))
        reaches_right_side = cx + cw >= x2 - max(4, int(0.10 * bw))
        extends_left = cx < x1 - max(2, int(0.03 * bw))
        extends_right = cx + cw > x2 + max(2, int(0.03 * bw))
        if (reaches_left_side and extends_left) or (reaches_right_side and extends_right):
            filtered[labels == label_idx] = 255

    if filtered.max() == 0:
        return np.zeros(fallback_mask.shape[:2], dtype=np.float32)
    return (filtered.astype(np.float32) / 255.0).astype(np.float32)


def _cleanup_occlusion_remnants_around_panel(
    upload_rgb: np.ndarray,
    display_rgb: np.ndarray,
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove thin synthetic-mask leftovers without expanding the GAN region.

    The direct panel mask is intentionally conservative to preserve identity.
    Some lfw_15may samples have small white straps or JPEG-bright panel edges
    just outside the central panel. Those are still occlusion pixels, but using
    DE-GAN on them tends to replace visible face. Inpaint only those paper-like
    remnants and keep the effective mask updated for debugging.
    """
    hard = _strict_occlusion_mask_array(core_mask) > 0.5
    bbox = _mask_bbox(hard.astype(np.float32))
    if bbox is None or hard.sum() < 20:
        return display_rgb, core_mask

    x1, y1, x2, y2 = bbox
    h, w = hard.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(12, int(0.22 * bw))
    pad_y = max(8, int(0.18 * bh))
    sx1 = max(0, x1 - pad_x)
    sx2 = min(w, x2 + pad_x)
    sy1 = max(0, y1 - pad_y)
    sy2 = min(h, y2 + pad_y)

    search = np.zeros((h, w), dtype=bool)
    side_y1 = max(0, y1 - 4)
    side_y2 = min(h, y2 + 4)
    search[side_y1:side_y2, sx1:min(w, x1 + 4)] = True
    search[side_y1:side_y2, max(0, x2 - 4):sx2] = True
    near_panel = cv2.dilate(hard.astype(np.uint8), np.ones((17, 31), np.uint8), iterations=1).astype(bool)
    search &= near_panel & ~hard
    if not np.any(search):
        return display_rgb, core_mask

    panel_pixels = upload_rgb[hard]
    if panel_pixels.size == 0:
        return display_rgb, core_mask
    target_rgb = np.median(panel_pixels.reshape(-1, 3), axis=0).astype(np.float32)

    hsv = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(upload_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    target_ycrcb = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2YCrCb)[0, 0].astype(np.float32)
    rgb_distance = np.linalg.norm(upload_rgb.astype(np.float32) - target_rgb, axis=2)
    ycrcb_distance = np.linalg.norm(ycrcb.astype(np.float32) - target_ycrcb, axis=2)

    remnant = (
        search
        & (hsv[:, :, 1] < 95)
        & (hsv[:, :, 2] > 82)
        & (rgb_distance < 76.0)
        & (ycrcb_distance < 46.0)
    )
    remnant_u8 = np.where(remnant, 255, 0).astype(np.uint8)
    remnant_u8 = cv2.morphologyEx(remnant_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    filtered = np.zeros_like(remnant_u8)
    contours, _ = cv2.findContours(remnant_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    core_area = float(hard.sum())
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 2.0 or area > max(450.0, core_area * 0.35):
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw > bw * 0.65 and ch > bh * 0.55:
            continue
        touches_panel_band = (
            cx + cw >= x1 - pad_x
            and cx <= x2 + pad_x
            and cy + ch >= y1 - pad_y
            and cy <= y2 + pad_y
        )
        if touches_panel_band:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)

    if filtered.max() == 0:
        return display_rgb, core_mask

    filtered = cv2.dilate(filtered, np.ones((3, 3), np.uint8), iterations=1)
    filtered = np.where((filtered > 0) & search, 255, 0).astype(np.uint8)
    inpainted = cv2.inpaint(display_rgb.astype(np.uint8), filtered, 3, cv2.INPAINT_TELEA)
    cleaned = display_rgb.copy()
    cleanup = filtered > 0
    cleaned[cleanup] = inpainted[cleanup]
    effective_mask = np.maximum(core_mask.astype(np.float32), filtered.astype(np.float32) / 255.0)
    return cleaned, effective_mask


def _remove_display_panel_edge_artifacts(
    display_rgb: np.ndarray,
    mask: np.ndarray,
    original_rgb: np.ndarray | None = None,
) -> np.ndarray:
    hard = _strict_occlusion_mask_array(mask) > 0.5
    if hard.sum() < 20:
        return display_rgb

    rectangular_panel = _is_rectangular_panel_mask(hard.astype(np.float32))
    erode_size = PANEL_EDGE_INPAINT_ERODE_SIZE if rectangular_panel else 13
    if erode_size % 2 == 0:
        erode_size += 1
    eroded = cv2.erode(hard.astype(np.uint8), np.ones((erode_size, erode_size), np.uint8), iterations=1).astype(bool)
    edge_band = hard & ~eroded
    if edge_band.sum() < 8:
        return display_rgb

    hsv = cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_edge = edge_band & (
        ((saturation < 82) & (value > 128))
        | (value > 218)
    )
    if bright_edge.sum() < 4:
        return display_rgb

    if original_rgb is not None and rectangular_panel:
        artifact_u8 = np.where(edge_band, 255, 0).astype(np.uint8)
    else:
        artifact_u8 = np.where(bright_edge, 255, 0).astype(np.uint8)
    artifact_u8 = cv2.dilate(artifact_u8, np.ones((3, 5), np.uint8), iterations=1)
    artifact_u8 = np.where((artifact_u8 > 0) & hard, 255, 0).astype(np.uint8)
    cleaned = display_rgb.copy()
    replace = artifact_u8 > 0
    inpaint_radius = PANEL_EDGE_INPAINT_RADIUS if original_rgb is not None and rectangular_panel else 3
    inpainted = cv2.inpaint(display_rgb.astype(np.uint8), artifact_u8, inpaint_radius, cv2.INPAINT_TELEA)
    cleaned[replace] = inpainted[replace]
    if original_rgb is not None and rectangular_panel:
        cleaned = _apply_rectangular_panel_skin_color_anchor(
            cleaned,
            original_rgb.astype(np.uint8),
            hard.astype(np.float32),
            strength=0.88,
        )
    return cleaned


def _add_subtle_rectangular_panel_structure(
    display_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    strength = float(np.clip(PANEL_SUBTLE_STRUCTURE_STRENGTH, 0.0, 0.65))
    if strength <= 0.0:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    if hard.sum() < 30 or not _is_rectangular_panel_mask(hard):
        return display_rgb

    inner = cv2.erode((hard > 0.5).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(np.float32)
    if inner.sum() < 20:
        return display_rgb
    inner = cv2.GaussianBlur(inner, (17, 17), 0) * hard

    structure_rgb = _skin_prior_repair_region(original_rgb.astype(np.uint8), hard)
    structure_rgb = _apply_rectangular_panel_skin_color_anchor(
        structure_rgb,
        original_rgb.astype(np.uint8),
        hard,
        strength=0.86,
    )

    display_f = display_rgb.astype(np.float32)
    structure_f = structure_rgb.astype(np.float32)
    blur_size = 21
    structure_low = cv2.GaussianBlur(structure_f, (blur_size, blur_size), 0)
    structure_detail = structure_f - structure_low
    candidate = np.clip(display_f + structure_detail * 0.35, 0, 255)

    alpha = (inner[..., None] * strength).astype(np.float32)
    enhanced = np.clip(display_f * (1.0 - alpha) + candidate * alpha, 0, 255).astype(np.uint8)
    return _apply_rectangular_panel_skin_color_anchor(
        enhanced,
        original_rgb.astype(np.uint8),
        hard,
        strength=0.78,
    )


def _sharpen_reconstructed_region_naturally(display_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    amount = float(np.clip(PANEL_RECON_REGION_SHARPEN_AMOUNT, 0.0, 1.25))
    if amount <= 0.0:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    hard_bool = hard > 0.5
    if hard_bool.sum() < 20:
        return display_rgb

    inner = cv2.erode(hard_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    if inner.sum() < 20:
        inner = hard
    alpha = cv2.GaussianBlur(inner, (5, 5), 0) * hard
    if alpha.max() <= 0.0:
        return display_rgb

    sigma = max(0.2, float(PANEL_RECON_REGION_SHARPEN_SIGMA))
    max_delta = max(0.0, float(PANEL_RECON_REGION_SHARPEN_MAX_DELTA))
    ycrcb = cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_channel = ycrcb[:, :, 0]
    blurred_y = cv2.GaussianBlur(y_channel, (0, 0), sigma)
    detail = np.clip((y_channel - blurred_y) * amount, -max_delta, max_delta)
    ycrcb[:, :, 0] = np.clip(y_channel + detail * alpha, 0, 255)
    sharpened_rgb = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    output = display_rgb.copy()
    output[hard_bool] = sharpened_rgb[hard_bool]
    return output


def _restore_degan_luminance_shape(
    display_rgb: np.ndarray,
    model_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    amount = float(np.clip(DEGAN_PRIOR_SHAPE_STRENGTH, 0.0, 0.85))
    if amount <= 0.0:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    hard_bool = hard > 0.5
    if hard_bool.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return display_rgb

    inner = cv2.erode(hard_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    if inner.sum() < 20:
        inner = hard
    inner_bool = inner > 0.5
    alpha = cv2.GaussianBlur(inner, (7, 7), 0) * hard
    if alpha.max() <= 0.0:
        return display_rgb

    display_ycrcb = cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    model_y = cv2.cvtColor(model_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    display_y = display_ycrcb[:, :, 0]

    sigma = max(0.5, float(DEGAN_PRIOR_SHAPE_SIGMA))
    model_structure = cv2.GaussianBlur(model_y, (0, 0), sigma)
    display_structure = cv2.GaussianBlur(display_y, (0, 0), sigma)
    model_center = float(np.percentile(model_structure[inner_bool], 50))
    display_center = float(np.percentile(display_structure[inner_bool], 50))
    shifted_model = model_structure + (display_center - model_center)

    max_delta = max(0.0, float(DEGAN_PRIOR_SHAPE_MAX_DELTA))
    delta = np.clip((shifted_model - display_structure) * amount, -max_delta, max_delta)
    display_ycrcb[:, :, 0] = np.clip(display_y + delta * alpha, 0, 255)
    shaped_rgb = cv2.cvtColor(np.clip(display_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    output = display_rgb.copy()
    output[hard_bool] = shaped_rgb[hard_bool]
    return output


def _restore_model_luminance_detail(
    display_rgb: np.ndarray,
    model_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    amount = float(np.clip(PANEL_MODEL_LUMINANCE_DETAIL_AMOUNT, 0.0, 0.75))
    if amount <= 0.0:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    hard_bool = hard > 0.5
    if hard_bool.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return display_rgb

    inner = cv2.erode(hard_bool.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)
    if inner.sum() < 20:
        inner = hard
    alpha = cv2.GaussianBlur(inner, (7, 7), 0) * hard
    if alpha.max() <= 0.0:
        return display_rgb

    tone_matched_model = _apply_rectangular_panel_skin_color_anchor(
        model_rgb.astype(np.uint8),
        original_rgb.astype(np.uint8),
        hard,
        strength=0.92,
    )
    model_ycrcb = cv2.cvtColor(tone_matched_model.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    display_ycrcb = cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)

    sigma = max(0.3, float(PANEL_MODEL_LUMINANCE_DETAIL_SIGMA))
    model_y = model_ycrcb[:, :, 0]
    display_y = display_ycrcb[:, :, 0]
    model_low = cv2.GaussianBlur(model_y, (0, 0), sigma)
    model_detail = model_y - model_low

    max_delta = max(0.0, float(PANEL_MODEL_LUMINANCE_DETAIL_MAX_DELTA))
    model_detail = np.clip(model_detail * amount, -max_delta, max_delta)
    display_ycrcb[:, :, 0] = np.clip(display_y + model_detail * alpha, 0, 255)
    detailed_rgb = cv2.cvtColor(np.clip(display_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    output = display_rgb.copy()
    output[hard_bool] = detailed_rgb[hard_bool]
    return output


def _restore_model_luminance_structure(
    display_rgb: np.ndarray,
    model_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    amount = float(np.clip(PANEL_MODEL_LUMINANCE_STRUCTURE_AMOUNT, 0.0, 0.75))
    if amount <= 0.0:
        return display_rgb

    hard = _strict_occlusion_mask_array(mask).astype(np.float32)
    hard_bool = hard > 0.5
    if hard_bool.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return display_rgb

    inner = cv2.erode(hard_bool.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(np.float32)
    if inner.sum() < 20:
        inner = hard
    inner_bool = inner > 0.5
    alpha = cv2.GaussianBlur(inner, (9, 9), 0) * hard
    if alpha.max() <= 0.0:
        return display_rgb

    tone_matched_model = _apply_rectangular_panel_skin_color_anchor(
        model_rgb.astype(np.uint8),
        original_rgb.astype(np.uint8),
        hard,
        strength=0.92,
    )
    model_y = cv2.cvtColor(tone_matched_model.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    display_ycrcb = cv2.cvtColor(display_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    display_y = display_ycrcb[:, :, 0]

    sigma = max(0.5, float(PANEL_MODEL_LUMINANCE_STRUCTURE_SIGMA))
    model_structure = cv2.GaussianBlur(model_y, (0, 0), sigma)
    display_structure = cv2.GaussianBlur(display_y, (0, 0), sigma)
    if inner_bool.sum() >= 20:
        model_center = float(np.percentile(model_structure[inner_bool], 50))
        display_center = float(np.percentile(display_structure[inner_bool], 50))
    else:
        model_center = float(np.percentile(model_structure[hard_bool], 50))
        display_center = float(np.percentile(display_structure[hard_bool], 50))

    shifted_model_structure = model_structure + (display_center - model_center)
    max_delta = max(0.0, float(PANEL_MODEL_LUMINANCE_STRUCTURE_MAX_DELTA))
    delta = np.clip((shifted_model_structure - display_structure) * amount, -max_delta, max_delta)
    display_ycrcb[:, :, 0] = np.clip(display_y + delta * alpha, 0, 255)
    structured_rgb = cv2.cvtColor(np.clip(display_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    output = display_rgb.copy()
    output[hard_bool] = structured_rgb[hard_bool]
    return output


def _outside_mask_max_delta(original_rgb: np.ndarray, candidate_rgb: np.ndarray, mask: np.ndarray) -> int:
    outside = mask <= 0.5
    if not np.any(outside):
        return 0
    diff = np.abs(candidate_rgb.astype(np.int16) - original_rgb.astype(np.int16))
    return int(diff[outside].max()) if diff[outside].size else 0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _region_from_mask(mask: np.ndarray) -> str:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "none"
    _, y1, _, y2 = bbox
    h = mask.shape[0]
    center_y = (y1 + y2) / 2.0
    if y1 < h * 0.34 and y2 > h * 0.58:
        return "mixed"
    if center_y < h * 0.43:
        return "eye-region"
    return "lower-face"


def _is_usable_direct_panel_mask(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    hard = _strict_occlusion_mask_array(mask)
    coverage = float(hard.mean())
    if coverage < DIRECT_PANEL_MIN_COVERAGE or coverage > DIRECT_PANEL_MAX_COVERAGE:
        return False
    bbox = _mask_bbox(hard)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    h, w = hard.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (
        0.42 <= aspect <= 2.35
        and w * 0.18 <= center_x <= w * 0.82
        and h * 0.33 <= center_y <= h * 0.82
        and bw <= w * 0.72
        and bh <= h * 0.62
    )


def _estimate_direct_panel_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Detect the synthetic lfw_15may panel directly in the original image space."""
    image_rgb = image_rgb.astype(np.uint8)
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)

    focus = np.zeros((h, w), dtype=np.uint8)
    focus[
        int(h * 0.36):int(h * 0.86),
        int(w * 0.16):int(w * 0.84),
    ] = 255

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    flat_neutral = (
        (focus > 0)
        & (saturation < 72)
        & (value > 122)
        & (texture < 18.0)
        & (cr >= 108)
        & (cr <= 162)
        & (cb >= 92)
        & (cb <= 154)
    )
    paper_distance = np.linalg.norm(
        image_rgb.astype(np.float32) - np.array([230.0, 220.0, 210.0], dtype=np.float32),
        axis=2,
    )
    paper_panel = (
        (focus > 0)
        & (paper_distance < 48.0)
        & (saturation < 92)
        & (value > 116)
        & (texture < 35.0)
        & (cr >= 124)
        & (cr <= 142)
        & (cb >= 112)
        & (cb <= 132)
    )
    candidate = paper_panel
    flat_pixels = lab[flat_neutral]
    if paper_panel.sum() >= max(35, int(h * w * 0.006)):
        candidate = paper_panel
    elif flat_pixels.shape[0] >= max(30, int(h * w * 0.004)):
        quantized = (flat_pixels.astype(np.int16) // np.array([8, 6, 6], dtype=np.int16)).astype(np.int16)
        unique_bins, counts = np.unique(quantized, axis=0, return_counts=True)
        dominant_bin = unique_bins[int(np.argmax(counts))]
        in_dominant_bin = np.all(
            (lab.astype(np.int16) // np.array([8, 6, 6], dtype=np.int16)) == dominant_bin,
            axis=2,
        )
        dominant_pixels = lab[flat_neutral & in_dominant_bin]
        target_lab = np.median(dominant_pixels, axis=0) if dominant_pixels.size else np.median(flat_pixels, axis=0)
        color_distance = np.linalg.norm(lab.astype(np.float32) - target_lab.astype(np.float32), axis=2)
        candidate = flat_neutral & (color_distance < 16.5)
    candidate[:int(h * 0.40), :] = False
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = float("-inf")
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(35.0, h * w * DIRECT_PANEL_MIN_COVERAGE * 0.45):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = float(max(1, bw * bh))
        fill_ratio = area / bbox_area
        aspect = bw / max(1.0, float(bh))
        coverage = area / float(max(1, h * w))
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        if coverage > DIRECT_PANEL_MAX_COVERAGE:
            continue
        if fill_ratio < 0.42 or aspect < 0.42 or aspect > 2.05:
            continue
        if center_x < w * 0.18 or center_x > w * 0.82:
            continue
        if center_y < h * 0.33 or center_y > h * 0.82:
            continue
        if bw > w * 0.72 or bh > h * 0.62:
            continue

        component = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        component_pixels = image_rgb[component > 0]
        color_std = float(component_pixels.reshape(-1, 3).std(axis=0).mean()) if component_pixels.size else 255.0
        target_y = h * 0.58
        center_penalty = abs(center_x - w / 2.0) / max(1.0, w / 2.0)
        target_penalty = abs(center_y - target_y) / max(1.0, h)
        score = (
            area * fill_ratio
            - 1800.0 * center_penalty
            - 1600.0 * target_penalty
            - 45.0 * max(0.0, color_std - 18.0)
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    panel_mask = np.zeros((h, w), dtype=np.uint8)
    if best_contour is None:
        return panel_mask.astype(np.float32)

    selected_contour = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(selected_contour, [best_contour], -1, 255, thickness=-1)
    panel_mask = np.where((candidate_u8 > 0) & (selected_contour > 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(panel_mask) < 20:
        panel_mask = selected_contour
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    panel_mask = _trim_panel_mask_to_dense_core(panel_mask)
    return _strict_occlusion_mask_array(panel_mask.astype(np.float32) / 255.0)


def _trim_panel_mask_to_dense_core(panel_mask: np.ndarray) -> np.ndarray:
    hard = _strict_occlusion_mask_array(panel_mask.astype(np.float32) / 255.0) > 0.5
    bbox = _mask_bbox(hard.astype(np.float32))
    if bbox is None or hard.sum() < 40:
        return panel_mask

    x1, y1, x2, y2 = bbox
    bh = max(1, y2 - y1)
    rows = []
    for y in range(y1, y2):
        xs = np.where(hard[y])[0]
        if xs.size >= 5:
            rows.append((y, int(xs.min()), int(xs.max()) + 1, int(xs.size)))
    if len(rows) < 6:
        return panel_mask

    top_limit = y1 + max(4, int(0.42 * bh))
    top_rows = [row for row in rows if row[0] <= top_limit]
    if len(top_rows) < 4:
        return panel_mask

    top_widths = np.array([row[3] for row in top_rows], dtype=np.float32)
    dense_width = float(np.median(top_widths))
    max_width = float(max(row[3] for row in rows))
    narrow_lower_tail = any(
        row[0] > y1 + 0.45 * bh and row[3] < dense_width * 0.35
        for row in rows
    )
    if dense_width < 8 or (max_width <= dense_width * 1.18 and not narrow_lower_tail):
        return panel_mask

    left = int(np.percentile([row[1] for row in top_rows], 15))
    right = int(np.percentile([row[2] for row in top_rows], 85))
    if right - left < 8:
        return panel_mask

    core_width = right - left
    overlap_threshold = max(5, int(0.42 * core_width))
    valid_rows = []
    for y, _, _, _ in rows:
        overlap = int(np.count_nonzero(hard[y, left:right]))
        if overlap >= overlap_threshold:
            valid_rows.append(y)
    if len(valid_rows) < 6:
        return panel_mask

    top = min(valid_rows)
    bottom = max(valid_rows) + 1
    trimmed = np.zeros_like(hard, dtype=np.uint8)
    trimmed[top:bottom, left:right] = hard[top:bottom, left:right].astype(np.uint8) * 255
    trimmed = cv2.morphologyEx(trimmed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if np.count_nonzero(trimmed) < max(35, int(0.45 * dense_width * (bottom - top))):
        return panel_mask
    return trimmed.astype(np.uint8)


def _is_rectangular_panel_mask(hard_mask: np.ndarray) -> bool:
    mask_u8 = np.where(hard_mask > 0.2, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    x, y, w, h = cv2.boundingRect(largest)
    if area < 30 or w < 7 or h < 7:
        return False
    aspect = w / max(1.0, float(h))
    fill_ratio = area / max(1.0, float(w * h))
    coverage = float(mask_u8.mean() / 255.0)
    return 0.45 <= aspect <= 1.75 and fill_ratio > 0.45 and coverage > 0.018


def _occlusion_boundary_jump(face_rgb: np.ndarray, hard_mask: np.ndarray) -> float:
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 20:
        return 0.0
    inner = cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    boundary = cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool) & ~mask_bool
    if inner.sum() < 20 or boundary.sum() < 20:
        return 0.0
    lab = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.linalg.norm(lab[inner].mean(axis=0) - lab[boundary].mean(axis=0)))


def _has_flat_rectangular_panel_evidence(face_rgb: np.ndarray, hard_mask: np.ndarray) -> bool:
    mask_bool = hard_mask > 0.2
    coverage = float(mask_bool.mean())
    if coverage < 0.012 or mask_bool.sum() < 40:
        return False

    hsv = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    patch_saturation = float(np.median(hsv[:, :, 1][mask_bool]))
    patch_value = float(np.median(hsv[:, :, 2][mask_bool]))
    patch_texture = float(np.median(texture[mask_bool]))

    boundary = cv2.dilate(mask_bool.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool) & ~mask_bool
    boundary_texture = float(np.median(texture[boundary])) if boundary.sum() > 20 else patch_texture

    plausible_panel_brightness = patch_value < 235.0
    neutral_flat_panel = (
        plausible_panel_brightness
        and patch_saturation < 95.0
        and patch_value > 105.0
        and patch_texture < 18.0
    )
    flatter_than_surroundings = patch_texture < max(10.0, boundary_texture * 0.72)
    large_panel = (
        plausible_panel_brightness
        and coverage > 0.045
        and patch_saturation < 115.0
        and patch_texture < 22.0
    )
    return neutral_flat_panel or (large_panel and flatter_than_surroundings)


def _has_visible_face_detail_evidence(face_rgb: np.ndarray, hard_mask: np.ndarray) -> bool:
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 40:
        return False

    gray = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edges = cv2.Canny(gray, 45, 120)

    patch_lap_median = float(np.median(laplacian[mask_bool]))
    patch_edge_density = float((edges[mask_bool] > 0).mean())
    patch_rgb_std = float(face_rgb[mask_bool].astype(np.float32).std(axis=0).mean())
    patch_saturation = float(np.median(hsv[:, :, 1][mask_bool]))
    return patch_edge_density > 0.16 or (patch_edge_density > 0.08 and patch_rgb_std > 35.0)


def _suppress_low_confidence_rectangular_occlusion(
    face_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    is_occluded: bool,
    region: str,
    ratio: float,
) -> tuple[np.ndarray, bool, str, float, float]:
    hard_mask = _harden_mask_array(occlusion_mask)
    boundary_jump = _occlusion_boundary_jump(face_rgb, hard_mask)
    rectangular_mask = _is_rectangular_panel_mask(hard_mask)
    visible_face_detail = _has_visible_face_detail_evidence(face_rgb, hard_mask) if rectangular_mask else False
    if is_occluded and boundary_jump < RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD:
        return np.zeros_like(occlusion_mask, dtype=np.float32), False, "none", 0.0, boundary_jump
    if is_occluded and rectangular_mask and visible_face_detail:
        return np.zeros_like(occlusion_mask, dtype=np.float32), False, "none", 0.0, boundary_jump
    return occlusion_mask, is_occluded, region, ratio, boundary_jump


def _inpaint_occlusion_region(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(hard_mask > 0.05, 255, 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return original_rgb.copy()
    bgr = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_u8, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def _symmetry_repair_region(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(hard_mask > 0.05, 255, 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return original_rgb.copy()

    mirrored_rgb = cv2.flip(original_rgb.astype(np.uint8), 1)
    mirrored_mask = cv2.flip(mask_u8, 1)
    inpainted_rgb = _inpaint_occlusion_region(original_rgb, hard_mask)
    source_valid = mirrored_mask < 8
    target = mask_u8 > 0
    usable = target & source_valid

    repaired = inpainted_rgb.copy().astype(np.float32)
    if usable.sum() > 30:
        mirrored_lab = cv2.cvtColor(mirrored_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        inpaint_lab = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        visible = (hard_mask < 0.05) & source_valid
        if visible.sum() > 50:
            src = mirrored_lab[visible]
            dst = inpaint_lab[visible]
            src_mean = src.mean(axis=0)
            dst_mean = dst.mean(axis=0)
            src_std = src.std(axis=0) + 1e-6
            dst_std = dst.std(axis=0) + 1e-6
            matched = mirrored_lab.copy()
            for channel in range(3):
                matched[:, :, channel] = ((matched[:, :, channel] - src_mean[channel]) / src_std[channel]) * dst_std[channel] + dst_mean[channel]
            mirrored_rgb = cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        alpha = cv2.GaussianBlur(usable.astype(np.float32), (7, 7), 0)[..., None]
        repaired = repaired * (1.0 - alpha) + mirrored_rgb.astype(np.float32) * alpha

    final_alpha = cv2.GaussianBlur((mask_u8 > 0).astype(np.float32), (7, 7), 0)[..., None]
    final_rgb = original_rgb.astype(np.float32) * (1.0 - final_alpha) + repaired * final_alpha
    return np.clip(final_rgb, 0, 255).astype(np.uint8)


def _skin_likelihood_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    warm_hue = (hue <= 32) | (hue >= 170)
    return (
        warm_hue
        & (saturation >= 10)
        & (saturation <= 210)
        & (value >= 35)
        & (cr >= 118)
        & (cr <= 190)
        & (cb >= 70)
        & (cb <= 155)
    )


def _relaxed_skin_likelihood_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    warm_or_neutral = (hue <= 38) | (hue >= 165) | (saturation < 45)
    return (
        warm_or_neutral
        & (saturation >= 6)
        & (saturation <= 225)
        & (value >= 45)
        & (cr >= 108)
        & (cr <= 195)
        & (cb >= 65)
        & (cb <= 165)
    )


def _central_face_prior(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    prior = np.zeros((h, w), dtype=np.uint8)
    center = (int(0.50 * w), int(0.50 * h))
    axes = (max(8, int(0.34 * w)), max(10, int(0.44 * h)))
    cv2.ellipse(prior, center, axes, 0, 0, 360, 255, thickness=-1)
    return prior > 0


def _local_skin_context_mask(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_bool = hard_mask > 0.08
    h, w = hard_mask.shape[:2]
    if mask_bool.sum() < 20:
        return np.zeros((h, w), dtype=bool)

    ys, xs = np.where(mask_bool)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    span = max(y2 - y1, x2 - x1)
    ring_radius = max(9, int(0.26 * span))
    kernel_size = ring_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    outer_ring = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask_bool

    face_prior = _central_face_prior((h, w))
    skin_like = _skin_likelihood_mask(original_rgb)
    relaxed_skin = _relaxed_skin_likelihood_mask(original_rgb)
    donor = outer_ring & face_prior & skin_like
    if donor.sum() >= 35:
        return donor

    y_pad = max(6, int(0.20 * (y2 - y1)))
    x_pad = max(6, int(0.22 * (x2 - x1)))
    local_box = np.zeros((h, w), dtype=bool)
    local_box[max(0, y1 - y_pad):min(h, y2 + y_pad), max(0, x1 - x_pad):min(w, x2 + x_pad)] = True
    donor = local_box & ~mask_bool & face_prior & skin_like
    if donor.sum() >= 35:
        return donor

    donor = face_prior & ~mask_bool & skin_like
    if donor.sum() >= 35:
        return donor

    donor = outer_ring & face_prior & ~mask_bool & relaxed_skin
    if donor.sum() >= 35:
        return donor

    donor = face_prior & ~mask_bool & relaxed_skin
    if donor.sum() >= 35:
        return donor

    return np.zeros((h, w), dtype=bool)


def _panel_skin_context_mask(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_bool = hard_mask > 0.08
    h, w = hard_mask.shape[:2]
    if mask_bool.sum() < 20:
        return np.zeros((h, w), dtype=bool)

    ys, xs = np.where(mask_bool)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    face_prior = _central_face_prior((h, w))
    strict_skin = _skin_likelihood_mask(original_rgb)
    relaxed_skin = _relaxed_skin_likelihood_mask(original_rgb)
    valid_skin = face_prior & ~mask_bool

    upper = np.zeros((h, w), dtype=bool)
    upper[
        max(0, y1 - int(0.72 * bh)):min(h, y1 + int(0.16 * bh)),
        max(0, x1 - int(0.18 * bw)):min(w, x2 + int(0.18 * bw)),
    ] = True

    sides = np.zeros((h, w), dtype=bool)
    side_pad = max(5, int(0.22 * bw))
    sides[
        max(0, y1 - int(0.05 * bh)):min(h, y2 + int(0.03 * bh)),
        max(0, x1 - side_pad):min(w, x1 + int(0.16 * bw)),
    ] = True
    sides[
        max(0, y1 - int(0.05 * bh)):min(h, y2 + int(0.03 * bh)),
        max(0, x2 - int(0.16 * bw)):min(w, x2 + side_pad),
    ] = True

    lower_exclusion = np.zeros((h, w), dtype=bool)
    lower_exclusion[min(h, y2 + max(2, int(0.08 * bh))):, :] = True
    preferred_zone = (upper | sides) & ~lower_exclusion

    for candidate in (
        preferred_zone & valid_skin & strict_skin,
        preferred_zone & valid_skin & relaxed_skin,
        valid_skin & strict_skin,
        valid_skin & relaxed_skin,
    ):
        filtered = _filter_skin_donor_mask(original_rgb, candidate)
        if filtered.sum() >= 35:
            return filtered

    return np.zeros((h, w), dtype=bool)


def _panel_boundary_skin_context_mask(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_bool = hard_mask > 0.08
    h, w = hard_mask.shape[:2]
    if mask_bool.sum() < 20:
        return np.zeros((h, w), dtype=bool)

    ys, xs = np.where(mask_bool)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    face_prior = _central_face_prior((h, w))
    strict_skin = _skin_likelihood_mask(original_rgb)
    relaxed_skin = _relaxed_skin_likelihood_mask(original_rgb)
    hsv = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    gray = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    smooth_visible = (hsv[:, :, 2] > 72) & (texture < 34.0)
    strict_skin &= smooth_visible
    relaxed_skin &= smooth_visible
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    reference_skin = (
        ((hue <= 30) | (hue >= 170))
        & (saturation >= 10)
        & (saturation <= 155)
        & (value > 88)
        & (cr >= 122)
        & (cr <= 182)
        & (cb >= 76)
        & (cb <= 148)
        & (texture < 26.0)
        & face_prior
        & ~mask_bool
    )

    upper = np.zeros((h, w), dtype=bool)
    upper[
        max(0, y1 - int(0.30 * bh)):min(h, y1 + int(0.10 * bh)),
        max(0, x1 - int(0.20 * bw)):min(w, x2 + int(0.20 * bw)),
    ] = True

    cheeks = np.zeros((h, w), dtype=bool)
    cheek_y1 = max(0, y1 - int(0.10 * bh))
    cheek_y2 = min(h, y2 + int(0.04 * bh))
    cheeks[cheek_y1:cheek_y2, max(0, x1 - int(0.34 * bw)):min(w, x1 + int(0.20 * bw))] = True
    cheeks[cheek_y1:cheek_y2, max(0, x2 - int(0.20 * bw)):min(w, x2 + int(0.34 * bw))] = True

    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(7, int(0.20 * bw) | 1), max(7, int(0.22 * bh) | 1)),
    )
    ring = cv2.dilate(mask_bool.astype(np.uint8), ring_kernel, iterations=1).astype(bool) & ~mask_bool
    preferred = (upper | cheeks | ring) & face_prior & ~mask_bool

    for candidate in (
        upper & reference_skin,
        (upper | cheeks) & reference_skin,
        preferred & reference_skin,
        face_prior & ~mask_bool & reference_skin,
        upper & face_prior & ~mask_bool & strict_skin,
        (upper | cheeks) & face_prior & ~mask_bool & strict_skin,
        preferred & strict_skin,
        upper & face_prior & ~mask_bool & relaxed_skin,
        preferred & relaxed_skin,
    ):
        filtered = _filter_skin_donor_mask(original_rgb, candidate)
        if filtered.sum() >= 35:
            return filtered

    return _panel_skin_context_mask(original_rgb, hard_mask)


def _filter_skin_donor_mask(original_rgb: np.ndarray, donor_mask: np.ndarray) -> np.ndarray:
    if donor_mask.sum() < 20:
        return donor_mask
    lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    l_values = lab[:, :, 0][donor_mask]
    sat_values = hsv[:, :, 1][donor_mask]
    chroma_values = np.sqrt((lab[:, :, 1][donor_mask] - 128.0) ** 2 + (lab[:, :, 2][donor_mask] - 128.0) ** 2)
    if l_values.size < 20:
        return donor_mask

    l_low = float(np.percentile(l_values, 12))
    l_high = float(np.percentile(l_values, 94))
    l_mid = float(np.percentile(l_values, 55))
    l_bright = float(np.percentile(l_values, 85))
    adaptive_l_floor = max(35.0, l_low - 4.0, min(l_mid, l_bright - 55.0))
    sat_high = float(np.percentile(sat_values, 96))
    chroma_high = float(np.percentile(chroma_values, 96))
    keep = (
        donor_mask
        & (lab[:, :, 0] >= adaptive_l_floor)
        & (lab[:, :, 0] <= min(255.0, l_high + 8.0))
        & (hsv[:, :, 1] <= min(230.0, sat_high + 12.0))
        & (np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2) <= chroma_high + 8.0)
    )
    return keep if keep.sum() >= 20 else donor_mask


def _robust_channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    p25 = np.percentile(values, 25, axis=0)
    p75 = np.percentile(values, 75, axis=0)
    scale = np.maximum((p75 - p25) / 1.349, 4.0)
    return median.astype(np.float32), scale.astype(np.float32)


def _apply_local_skin_tone_match(
    candidate_rgb: np.ndarray,
    original_rgb: np.ndarray,
    hard_mask: np.ndarray,
    *,
    strength: float = 0.72,
) -> np.ndarray:
    mask_bool = hard_mask > 0.2
    donor_mask = _panel_skin_context_mask(original_rgb, hard_mask)
    if donor_mask.sum() < 35:
        donor_mask = _local_skin_context_mask(original_rgb, hard_mask)
    if mask_bool.sum() < 20 or donor_mask.sum() < 35:
        return candidate_rgb

    cand_lab = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    source_pixels = cand_lab[mask_bool]
    target_pixels = target_lab[donor_mask]
    if source_pixels.shape[0] < 20 or target_pixels.shape[0] < 35:
        return candidate_rgb
    target_l = target_pixels[:, 0]
    mid_target = target_pixels[
        (target_l >= float(np.percentile(target_l, 15)))
        & (target_l <= float(np.percentile(target_l, 90)))
    ]
    if mid_target.shape[0] >= 20:
        target_pixels = mid_target

    source_center, source_scale = _robust_channel_stats(source_pixels)
    target_center, target_scale = _robust_channel_stats(target_pixels)
    remapped = cand_lab.copy()
    channel_strengths = np.array([0.72, 0.92, 0.92], dtype=np.float32) * float(np.clip(strength, 0.0, 1.0))
    for channel_idx, channel_strength in enumerate(channel_strengths):
        normalized = (cand_lab[:, :, channel_idx] - source_center[channel_idx]) / source_scale[channel_idx]
        matched = normalized * target_scale[channel_idx] + target_center[channel_idx]
        remapped[:, :, channel_idx] = (
            cand_lab[:, :, channel_idx] * (1.0 - channel_strength)
            + matched * channel_strength
        )

    target_l_p05 = float(np.percentile(target_pixels[:, 0], 5))
    target_l_p95 = float(np.percentile(target_pixels[:, 0], 95))
    remapped[:, :, 0] = np.clip(remapped[:, :, 0], max(0.0, target_l_p05 - 12.0), min(255.0, target_l_p95 + 16.0))
    matched_rgb = cv2.cvtColor(np.clip(remapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    mask_limit = np.where(hard_mask > 0.2, 1.0, 0.0).astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask_limit, (9, 9), 0) * mask_limit
    alpha = (soft_mask[..., None] ** 0.85) * float(np.clip(strength, 0.0, 1.0))
    output = candidate_rgb.astype(np.float32) * (1.0 - alpha) + matched_rgb * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _apply_rectangular_panel_skin_color_anchor(
    candidate_rgb: np.ndarray,
    original_rgb: np.ndarray,
    hard_mask: np.ndarray,
    *,
    strength: float | None = None,
) -> np.ndarray:
    hard = _strict_occlusion_mask_array(hard_mask).astype(np.float32)
    mask_bool = hard > 0.5
    if mask_bool.sum() < 20 or not _is_rectangular_panel_mask(hard):
        return candidate_rgb

    donor_mask = _panel_boundary_skin_context_mask(original_rgb, hard)
    if donor_mask.sum() < 35:
        donor_mask = _panel_skin_context_mask(original_rgb, hard)
    if donor_mask.sum() < 35:
        donor_mask = _local_skin_context_mask(original_rgb, hard)
    if donor_mask.sum() < 35:
        return candidate_rgb

    cand_lab = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    source_pixels = cand_lab[mask_bool]
    target_pixels = target_lab[donor_mask]
    if source_pixels.shape[0] < 20 or target_pixels.shape[0] < 35:
        return candidate_rgb

    target_l = target_pixels[:, 0]
    target_mid = target_pixels[
        (target_l >= float(np.percentile(target_l, 12)))
        & (target_l <= float(np.percentile(target_l, 88)))
    ]
    if target_mid.shape[0] >= 20:
        target_pixels = target_mid

    source_center, source_scale = _robust_channel_stats(source_pixels)
    target_center, target_scale = _robust_channel_stats(target_pixels)
    target_center[0] = float(np.percentile(target_pixels[:, 0], 50))

    remapped = cand_lab.copy()
    scale_mix = np.array([0.30, 0.40, 0.40], dtype=np.float32)
    channel_strengths = np.array([0.94, 1.0, 1.0], dtype=np.float32) * float(
        np.clip(PANEL_RECTANGULAR_COLOR_ANCHOR_STRENGTH if strength is None else strength, 0.0, 1.0)
    )
    for channel_idx, channel_strength in enumerate(channel_strengths):
        shifted = cand_lab[:, :, channel_idx] + (target_center[channel_idx] - source_center[channel_idx])
        normalized = (cand_lab[:, :, channel_idx] - source_center[channel_idx]) / source_scale[channel_idx]
        scaled = normalized * target_scale[channel_idx] + target_center[channel_idx]
        color_matched = shifted * (1.0 - scale_mix[channel_idx]) + scaled * scale_mix[channel_idx]
        remapped[:, :, channel_idx] = (
            cand_lab[:, :, channel_idx] * (1.0 - channel_strength)
            + color_matched * channel_strength
        )

    target_low = np.percentile(target_pixels, 5, axis=0)
    target_high = np.percentile(target_pixels, 95, axis=0)
    remapped[:, :, 0] = np.clip(remapped[:, :, 0], max(0.0, target_low[0] - 8.0), min(255.0, target_high[0] + 10.0))
    remapped[:, :, 1] = np.clip(remapped[:, :, 1], max(0.0, target_low[1] - 6.0), min(255.0, target_high[1] + 6.0))
    remapped[:, :, 2] = np.clip(remapped[:, :, 2], max(0.0, target_low[2] - 6.0), min(255.0, target_high[2] + 6.0))

    matched_rgb = cv2.cvtColor(np.clip(remapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    soft = cv2.GaussianBlur(hard, (11, 11), 0)
    alpha_strength = float(
        np.clip(PANEL_RECTANGULAR_COLOR_ANCHOR_STRENGTH if strength is None else strength, 0.0, 1.0)
    )
    alpha = ((0.58 + 0.42 * soft) * hard * alpha_strength)[..., None]
    output = candidate_rgb.astype(np.float32) * (1.0 - alpha) + matched_rgb * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _skin_match_region(candidate_rgb: np.ndarray, original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    return _apply_local_skin_tone_match(candidate_rgb, original_rgb, hard_mask, strength=0.74)


def _remove_panel_line_artifacts(candidate_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 30:
        return candidate_rgb
    gray = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    masked_values = gray[mask_bool]
    dark_threshold = max(0.0, float(np.percentile(masked_values, 35) - 12.0))
    edge = cv2.Canny(gray, 35, 100)
    artifact = ((gray < dark_threshold) | (edge > 0)) & mask_bool
    artifact_u8 = artifact.astype(np.uint8) * 255
    artifact_u8 = cv2.morphologyEx(artifact_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    artifact_u8 = cv2.dilate(artifact_u8, np.ones((2, 2), np.uint8), iterations=1)
    if artifact_u8.sum() < 20:
        return candidate_rgb
    repaired = cv2.inpaint(cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR), artifact_u8, 2, cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def _skin_prior_repair_region(original_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    hard_mask = hard_mask.astype(np.float32).copy()
    mask_bool = hard_mask > 0.08
    if mask_bool.sum() < 30:
        return original_rgb.copy()

    ys, xs = np.where(mask_bool)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    h, w = original_rgb.shape[:2]
    if y2 >= h - 3:
        expand_x = max(8, int(0.20 * (x2 - x1)))
        expand_top = max(3, int(0.08 * (y2 - y1)))
        x1 = max(0, x1 - expand_x)
        x2 = min(w, x2 + expand_x)
        y1 = max(0, y1 - expand_top)
        hard_mask[y1:y2, x1:x2] = 1.0
        mask_bool = hard_mask > 0.08

    pad = max(4, int(0.08 * max(y2 - y1, x2 - x1)))
    by1, by2 = max(0, y1 - pad), min(h, y2 + pad)
    bx1, bx2 = max(0, x1 - pad), min(w, x2 + pad)

    visible_band = np.zeros(mask_bool.shape, dtype=np.uint8)
    visible_band[by1:by2, bx1:bx2] = 1
    visible_band = (visible_band > 0) & ~mask_bool
    if visible_band.sum() < 40:
        visible_band = ~mask_bool

    hsv = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    skin_mask = (
        (hsv[:, :, 0] <= 28)
        & (hsv[:, :, 1] >= 12)
        & (hsv[:, :, 1] <= 190)
        & (ycrcb[:, :, 1] >= 125)
        & (ycrcb[:, :, 1] <= 185)
        & (ycrcb[:, :, 2] >= 75)
        & (ycrcb[:, :, 2] <= 145)
        & visible_band
    )
    donor_mask = skin_mask if skin_mask.sum() >= 30 else visible_band
    donor_rgb = original_rgb[donor_mask].astype(np.float32)
    skin_rgb = np.median(donor_rgb, axis=0) if donor_rgb.size else np.array([170.0, 125.0, 105.0], dtype=np.float32)

    repaired = _inpaint_occlusion_region(original_rgb, hard_mask).astype(np.float32)
    patch_h = max(1, y2 - y1)
    patch_w = max(1, x2 - x1)
    yy, xx = np.mgrid[0:patch_h, 0:patch_w].astype(np.float32)
    yn = yy / max(1.0, float(patch_h - 1))
    xn = (xx / max(1.0, float(patch_w - 1))) * 2.0 - 1.0

    skin_patch = np.zeros((patch_h, patch_w, 3), dtype=np.float32)
    vertical = 1.06 - 0.18 * yn
    side_shadow = 1.0 - 0.05 * np.abs(xn)
    nose_column = np.exp(-((xn / 0.17) ** 2)) * np.exp(-((yn - 0.34) / 0.34) ** 2)
    nose_side = (
        np.exp(-(((xn - 0.16) / 0.08) ** 2 + ((yn - 0.42) / 0.30) ** 2))
        + np.exp(-(((xn + 0.16) / 0.08) ** 2 + ((yn - 0.42) / 0.30) ** 2))
    )
    nostril = (
        np.exp(-(((xn - 0.12) / 0.075) ** 2 + ((yn - 0.58) / 0.045) ** 2))
        + np.exp(-(((xn + 0.12) / 0.075) ** 2 + ((yn - 0.58) / 0.045) ** 2))
    )
    mouth_curve = np.exp(-((yn - (0.75 + 0.025 * (xn ** 2))) / 0.045) ** 2) * np.exp(-(xn / 0.58) ** 2)
    lower_lip = np.exp(-((yn - 0.82) / 0.060) ** 2) * np.exp(-(xn / 0.48) ** 2)
    chin_highlight = 1.0 + 0.06 * np.exp(-((yn - 0.93) / 0.12) ** 2) * np.exp(-(xn / 0.70) ** 2)
    shade = vertical * side_shadow * (1.0 + 0.08 * nose_column) * (1.0 - 0.08 * nose_side) * (1.0 - 0.28 * nostril) * (1.0 - 0.20 * mouth_curve) * chin_highlight
    for channel in range(3):
        skin_patch[:, :, channel] = skin_rgb[channel] * shade

    skin_patch[:, :, 0] += 10.0 * lower_lip + 7.0 * mouth_curve
    skin_patch[:, :, 1] -= 3.0 * lower_lip + 6.0 * mouth_curve
    skin_patch[:, :, 2] -= 5.0 * lower_lip + 8.0 * mouth_curve
    skin_patch[:, :, :] -= nostril[..., None] * np.array([18.0, 16.0, 14.0], dtype=np.float32)

    noise = cv2.GaussianBlur(np.random.default_rng(7).normal(0.0, 2.1, skin_patch.shape).astype(np.float32), (3, 3), 0)
    skin_patch = np.clip(skin_patch + noise, 0, 255)

    local_mask = hard_mask[y1:y2, x1:x2].astype(np.float32)
    local_alpha = cv2.GaussianBlur(local_mask, (11, 11), 0)[..., None]
    local_alpha = np.clip(local_alpha, 0.0, 0.96)
    base_patch = repaired[y1:y2, x1:x2]
    repaired[y1:y2, x1:x2] = base_patch * (1.0 - local_alpha) + skin_patch * local_alpha

    full_alpha = cv2.GaussianBlur(hard_mask.astype(np.float32), (31, 31), 0)[..., None]
    full_alpha = np.clip(full_alpha, 0.0, 0.97)
    final_rgb = original_rgb.astype(np.float32) * (1.0 - full_alpha) + repaired * full_alpha
    return np.clip(final_rgb, 0, 255).astype(np.uint8)


def _artifact_score(candidate_rgb: np.ndarray, original_rgb: np.ndarray, hard_mask: np.ndarray) -> float:
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 20:
        return 0.0
    gray = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    grad = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    masked_grad = float(grad[mask_bool].mean())
    boundary = cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool) & ~mask_bool
    boundary_jump = 0.0
    if boundary.sum() > 20:
        candidate_lab = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        boundary_jump = float(
            np.linalg.norm(candidate_lab[mask_bool].mean(axis=0) - candidate_lab[boundary].mean(axis=0))
        )
    patch = candidate_rgb[mask_bool].astype(np.float32)
    patch_std = float(patch.std(axis=0).mean())
    return boundary_jump + masked_grad * 0.35 - patch_std * 0.25


def _postprocess_reconstructed_face(
    original_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    *,
    conservative: bool = False,
) -> np.ndarray:
    hard_mask = _harden_mask_array(occlusion_mask)
    inpainted_rgb = _inpaint_occlusion_region(original_rgb, hard_mask)
    symmetry_rgb = _symmetry_repair_region(original_rgb, hard_mask)
    if _is_rectangular_panel_mask(hard_mask):
        # Keep this branch as a true reconstruction output. Previous versions
        # selected inpaint/symmetry for rectangular panels, which removed the
        # mask but replaced the face with a smooth fill instead of model output.
        repaired = _skin_match_region(candidate_rgb.astype(np.uint8), original_rgb.astype(np.uint8), hard_mask)
        repaired = _remove_panel_line_artifacts(repaired, hard_mask)
        return _replace_only_occluded_region(original_rgb, repaired.astype(np.uint8), occlusion_mask)

    candidate_rgb = _skin_match_region(candidate_rgb.astype(np.uint8), original_rgb.astype(np.uint8), hard_mask)
    candidate_rgb = _remove_panel_line_artifacts(candidate_rgb, hard_mask)
    score = _artifact_score(candidate_rgb, original_rgb, hard_mask)

    if conservative or score > 42.0:
        mix_strength = 0.45
    elif score > 28.0:
        mix_strength = 0.30
    else:
        mix_strength = 0.12

    repaired_model = (
        candidate_rgb.astype(np.float32) * (1.0 - mix_strength)
        + inpainted_rgb.astype(np.float32) * mix_strength
    )
    # Use symmetry only when it scores clearly better; otherwise preserve the
    # model's facial detail and only repair the obvious panel/line artifacts.
    repaired_candidate = np.clip(repaired_model, 0, 255).astype(np.uint8)
    symmetry_score = _candidate_quality_score(symmetry_rgb, original_rgb, occlusion_mask)
    repaired_score = _candidate_quality_score(repaired_candidate, original_rgb, occlusion_mask)
    repaired = symmetry_rgb.astype(np.float32) if symmetry_score + 8.0 < repaired_score else repaired_model
    return _replace_only_occluded_region(original_rgb, np.clip(repaired, 0, 255).astype(np.uint8), occlusion_mask)


def _candidate_quality_score(candidate_rgb: np.ndarray, original_rgb: np.ndarray, occlusion_mask: np.ndarray) -> float:
    hard_mask = _harden_mask_array(occlusion_mask)
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 20:
        return 0.0

    gray = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    candidate_lab = cv2.cvtColor(candidate_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    boundary = cv2.dilate(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool) & ~mask_bool
    boundary_jump = 0.0
    if boundary.sum() > 20:
        boundary_jump = float(np.linalg.norm(candidate_lab[mask_bool].mean(axis=0) - candidate_lab[boundary].mean(axis=0)))

    visible = hard_mask < 0.05
    global_color_shift = 0.0
    if visible.sum() > 50:
        global_color_shift = float(np.abs(candidate_lab[visible] - original_lab[visible]).mean())

    patch_std = float(candidate_rgb[mask_bool].astype(np.float32).std(axis=0).mean())
    masked_lap = float(lap[mask_bool].mean())
    rectangular_penalty = max(0.0, boundary_jump - 18.0) + max(0.0, masked_lap - 18.0)
    flatness_penalty = max(0.0, 14.0 - patch_std)
    return rectangular_penalty + flatness_penalty * 1.8 + global_color_shift * 0.7


def _rectangular_repair_score(candidate_rgb: np.ndarray, original_rgb: np.ndarray, hard_mask: np.ndarray) -> float:
    mask_bool = hard_mask > 0.2
    if mask_bool.sum() < 20:
        return 0.0

    candidate_rgb = candidate_rgb.astype(np.uint8)
    original_rgb = original_rgb.astype(np.uint8)
    candidate_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    candidate_hsv = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2HSV)

    mask_u8 = mask_bool.astype(np.uint8)
    inner = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if inner.sum() < 20:
        inner = mask_bool
    boundary = cv2.dilate(mask_u8, np.ones((7, 7), np.uint8), iterations=1).astype(bool) & ~mask_bool
    donor = _local_skin_context_mask(original_rgb, hard_mask)
    if donor.sum() < 35 and boundary.sum() >= 20:
        donor = boundary
    if donor.sum() < 35:
        donor = hard_mask < 0.05

    patch_lab = candidate_lab[inner]
    donor_lab = original_lab[donor]
    color_jump = 0.0
    if patch_lab.shape[0] > 0 and donor_lab.shape[0] > 0:
        color_jump = float(np.linalg.norm(patch_lab.mean(axis=0) - donor_lab.mean(axis=0)))

    boundary_jump = 0.0
    if boundary.sum() >= 20:
        boundary_jump = float(np.linalg.norm(patch_lab.mean(axis=0) - candidate_lab[boundary].mean(axis=0)))

    patch_lap = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F))
    context_lap = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F))
    patch_texture = float(np.median(patch_lap[inner]))
    context_texture = float(np.median(context_lap[donor])) if donor.sum() >= 35 else 9.0
    target_texture = float(np.clip(context_texture, 7.0, 24.0))
    blur_penalty = max(0.0, target_texture * 0.72 - patch_texture) * 2.5
    noisy_penalty = max(0.0, patch_texture - target_texture * 2.4) * 0.45

    patch_rgb = candidate_rgb[inner].astype(np.float32)
    patch_std = float(patch_rgb.std(axis=0).mean())
    flatness_penalty = max(0.0, 13.0 - patch_std) * 1.6
    patch_saturation = float(np.median(candidate_hsv[:, :, 1][inner]))
    patch_value = float(np.median(candidate_hsv[:, :, 2][inner]))
    neutral_panel_penalty = 18.0 if patch_saturation < 42.0 and patch_value > 135.0 and patch_texture < 9.0 else 0.0

    return (
        boundary_jump * 0.82
        + color_jump * 0.44
        + blur_penalty
        + noisy_penalty
        + flatness_penalty
        + neutral_panel_penalty
    )


def _feather_region_blend(original_rgb: np.ndarray, candidate_rgb: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(hard_mask.astype(np.float32), 0.0, 1.0)
    if mask.max() <= 0.0:
        return original_rgb.astype(np.uint8)

    alpha = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)[..., None]
    alpha = alpha * (mask > 0.5).astype(np.float32)[..., None]
    alpha = np.clip(alpha, 0.0, 0.96)
    return np.clip(
        original_rgb.astype(np.float32) * (1.0 - alpha)
        + candidate_rgb.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _detail_preserving_reconstruction_candidate(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    hard_mask = _strict_occlusion_mask_array(hard_mask)
    reconstructed_rgb = _suppress_reconstructed_panel_leak(
        original_rgb.astype(np.uint8),
        reconstructed_rgb.astype(np.uint8),
        hard_mask,
    )
    detail_strength = float(np.clip(RECTANGULAR_MODEL_DETAIL_STRENGTH, 0.0, 1.0))
    if detail_strength <= 0.0 or not _needs_rectangular_detail_rescue(hard_mask):
        return reconstructed_rgb.astype(np.uint8)

    enhanced_rgb = _enhance_reconstructed_patch(
        original_rgb.astype(np.uint8),
        reconstructed_rgb.astype(np.uint8),
        hard_mask,
    )
    enhanced_rgb = _remove_panel_line_artifacts(enhanced_rgb, hard_mask)
    enhanced_rgb = _apply_local_skin_tone_match(
        enhanced_rgb,
        original_rgb.astype(np.uint8),
        hard_mask,
        strength=0.58,
    )
    detail_alpha = (hard_mask[..., None] ** 1.1) * detail_strength
    return np.clip(
        reconstructed_rgb.astype(np.float32) * (1.0 - detail_alpha)
        + enhanced_rgb.astype(np.float32) * detail_alpha,
        0,
        255,
    ).astype(np.uint8)


def _suppress_reconstructed_panel_leak(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    hard = _strict_occlusion_mask_array(hard_mask) > 0.5
    if hard.sum() < 20:
        return reconstructed_rgb.astype(np.uint8)

    original_u8 = original_rgb.astype(np.uint8)
    reconstructed_u8 = reconstructed_rgb.astype(np.uint8)
    original_hsv = cv2.cvtColor(original_u8, cv2.COLOR_RGB2HSV)
    reconstructed_hsv = cv2.cvtColor(reconstructed_u8, cv2.COLOR_RGB2HSV)
    panel_pixels = original_u8[hard].astype(np.float32)
    if panel_pixels.size == 0:
        return reconstructed_u8

    panel_rgb = np.median(panel_pixels.reshape(-1, 3), axis=0).astype(np.float32)
    panel_distance = np.linalg.norm(reconstructed_u8.astype(np.float32) - panel_rgb, axis=2)
    unchanged_distance = np.abs(reconstructed_u8.astype(np.float32) - original_u8.astype(np.float32)).mean(axis=2)
    original_panel_like = (
        (original_hsv[:, :, 1] < 95)
        & (original_hsv[:, :, 2] > 112)
        & hard
    )
    reconstructed_panel_like = (
        (reconstructed_hsv[:, :, 1] < 96)
        & (reconstructed_hsv[:, :, 2] > 118)
        & (panel_distance < 78.0)
        & hard
    )
    copied_panel_like = original_panel_like & (unchanged_distance < 34.0)
    leak = reconstructed_panel_like | copied_panel_like
    if float(leak.sum()) / max(1.0, float(hard.sum())) < 0.08:
        return reconstructed_u8

    leak_u8 = np.where(leak, 255, 0).astype(np.uint8)
    leak_u8 = cv2.morphologyEx(leak_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    leak_u8 = np.where((leak_u8 > 0) & hard, 255, 0).astype(np.uint8)
    repair_rgb = _skin_prior_repair_region(original_u8, hard.astype(np.float32))
    leak_alpha = cv2.GaussianBlur((leak_u8 > 0).astype(np.float32), (7, 7), 0)
    leak_alpha = np.clip(leak_alpha, 0.0, 1.0) * hard.astype(np.float32)
    return np.clip(
        reconstructed_u8.astype(np.float32) * (1.0 - leak_alpha[..., None])
        + repair_rgb.astype(np.float32) * leak_alpha[..., None],
        0,
        255,
    ).astype(np.uint8)


def _repair_rectangular_reconstruction(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
) -> np.ndarray:
    hard_mask = _strict_occlusion_mask_array(occlusion_mask)
    detail_candidate = _detail_preserving_reconstruction_candidate(
        original_rgb.astype(np.uint8),
        reconstructed_rgb.astype(np.uint8),
        hard_mask,
    )
    legacy_blend = _replace_only_occluded_region(
        original_rgb.astype(np.uint8),
        detail_candidate,
        occlusion_mask,
    )
    legacy_blend = _apply_local_skin_tone_match(legacy_blend, original_rgb.astype(np.uint8), hard_mask, strength=0.64)
    legacy_blend = _apply_rectangular_panel_skin_color_anchor(
        legacy_blend,
        original_rgb.astype(np.uint8),
        hard_mask,
    )
    legacy_blend = _replace_only_occluded_region(original_rgb.astype(np.uint8), legacy_blend, occlusion_mask)

    visual_repair_strength = float(np.clip(RECTANGULAR_VISUAL_REPAIR_STRENGTH, 0.0, 1.0))
    if visual_repair_strength <= 0.0:
        return legacy_blend

    candidates = [
        legacy_blend,
        _symmetry_repair_region(original_rgb.astype(np.uint8), hard_mask),
        _inpaint_occlusion_region(original_rgb.astype(np.uint8), hard_mask),
        _skin_prior_repair_region(original_rgb.astype(np.uint8), hard_mask),
    ]

    legacy_score = _rectangular_repair_score(legacy_blend, original_rgb, hard_mask)
    best = min(candidates, key=lambda candidate: _rectangular_repair_score(candidate, original_rgb, hard_mask))
    best_score = _rectangular_repair_score(best, original_rgb, hard_mask)
    if best is legacy_blend or best_score + 18.0 >= legacy_score:
        return legacy_blend

    repair_strength = float(np.clip((legacy_score - best_score - 18.0) / 42.0, 0.18, 0.46))
    repair_strength *= visual_repair_strength
    if repair_strength <= 0.0:
        return legacy_blend
    repair_alpha = (hard_mask[..., None] ** 1.15) * repair_strength
    softened = np.clip(
        legacy_blend.astype(np.float32) * (1.0 - repair_alpha)
        + best.astype(np.float32) * repair_alpha,
        0,
        255,
    ).astype(np.uint8)
    return _replace_only_occluded_region(original_rgb.astype(np.uint8), softened, occlusion_mask)


def _blend_rgb_from_reconstruction(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    occlusion_mask: np.ndarray,
    *,
    enhance_patch: bool = False,
) -> np.ndarray:
    hard_mask = _harden_mask_array(occlusion_mask)
    if _is_rectangular_panel_mask(hard_mask):
        return _repair_rectangular_reconstruction(
            original_rgb.astype(np.uint8),
            reconstructed_rgb.astype(np.uint8),
            occlusion_mask,
        )

    candidate_rgb = reconstructed_rgb.astype(np.uint8)
    if enhance_patch:
        candidate_rgb = _enhance_reconstructed_patch(
            original_rgb.astype(np.uint8),
            candidate_rgb,
            occlusion_mask,
        )
    candidate_rgb = _postprocess_reconstructed_face(
        original_rgb.astype(np.uint8),
        candidate_rgb.astype(np.uint8),
        occlusion_mask,
        conservative=True,
    ).astype(np.float32)
    return _replace_only_occluded_region(
        original_rgb.astype(np.uint8),
        candidate_rgb.astype(np.uint8),
        occlusion_mask,
    )


def _needs_rectangular_detail_rescue(hard_mask: np.ndarray) -> bool:
    return _is_rectangular_panel_mask(hard_mask)


def _choose_best_repair(original_rgb: np.ndarray, candidates: list[np.ndarray], occlusion_mask: np.ndarray) -> np.ndarray:
    best = candidates[0]
    best_score = _candidate_quality_score(best, original_rgb, occlusion_mask)
    for candidate in candidates[1:]:
        score = _candidate_quality_score(candidate, original_rgb, occlusion_mask)
        if score < best_score:
            best = candidate
            best_score = score
    return best


def _estimate_bright_panel_mask(face_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    grayscale = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    texture_energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(grayscale, cv2.CV_32F)), (5, 5), 0)

    h, w = face_rgb.shape[:2]
    focus_mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, int(w * 0.20))
    x2 = min(w, int(w * 0.80))
    y1 = max(0, int(h * 0.28))
    y2 = min(h, int(h * 0.88))
    focus_mask[y1:y2, x1:x2] = 255

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    neutral_panel = (
        (hsv[:, :, 1] < 58)
        & (hsv[:, :, 2] > 142)
        & (texture_energy < 12.5)
        & (cr >= 116)
        & (cr <= 150)
        & (cb >= 104)
        & (cb <= 142)
    )
    bright_paper = (
        (hsv[:, :, 1] < 46)
        & (hsv[:, :, 2] > 158)
        & (texture_energy < 12.0)
        & (cr <= 155)
        & (cb >= 100)
        & (cb <= 146)
    )
    candidate = (neutral_panel | bright_paper) & (focus_mask > 0)
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    panel_mask = np.zeros_like(candidate_u8)
    contours, _ = cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return panel_mask.astype(np.float32)

    image_center_x = w / 2.0
    image_center_y = h / 2.0
    best_contour = None
    best_score = float("-inf")
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 30:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / max(1.0, float(bh))
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        center_offset_x = abs(center_x - image_center_x) / max(1.0, image_center_x)
        lower_face_target_y = h * 0.58
        center_offset_y = abs(center_y - lower_face_target_y) / max(1.0, h)
        aspect_penalty = abs(np.log(max(aspect_ratio, 1e-6)))
        size_penalty = 0.0
        if bw > w * 0.42 or bh > h * 0.42:
            size_penalty += 1.2
        if aspect_ratio < 0.45 or aspect_ratio > 1.6:
            size_penalty += 1.6
        if center_y < h * 0.38:
            size_penalty += 2.4
        score = (
            area
            - 220.0 * center_offset_x
            - 180.0 * center_offset_y
            - 120.0 * aspect_penalty
            - 180.0 * size_penalty
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return panel_mask.astype(np.float32)

    x, y, bw, bh = cv2.boundingRect(best_contour)
    pad_x = max(2, int(0.02 * w))
    pad_y = max(2, int(0.02 * h))
    panel_mask[max(0, y - pad_y):min(h, y + bh + pad_y), max(0, x - pad_x):min(w, x + bw + pad_x)] = 255
    return cv2.GaussianBlur(panel_mask.astype(np.float32) / 255.0, (9, 9), 0)


def _estimate_skin_occluder_mask(
    face_rgb: np.ndarray,
    focus_region: np.ndarray,
    center_region: np.ndarray,
    eye_region: np.ndarray,
    lower_region: np.ndarray,
    texture_energy: np.ndarray,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    focus_pixels = texture_energy[focus_region > 0]
    adaptive_texture_threshold = 7.0
    if focus_pixels.size > 0:
        adaptive_texture_threshold = float(np.clip(np.percentile(focus_pixels, 32), 4.8, 8.6))

    candidate = (
        (ycrcb[:, :, 1] >= 132)
        & (ycrcb[:, :, 1] <= 182)
        & (ycrcb[:, :, 2] >= 82)
        & (ycrcb[:, :, 2] <= 138)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 1] <= 165)
        & (texture_energy <= adaptive_texture_threshold)
        & (focus_region > 0)
    )
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    h, w = face_rgb.shape[:2]
    best_contour = None
    best_score = float("-inf")
    for contour in cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area < 45:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > w * 0.92 and bh > h * 0.72:
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        center_overlap = float((component_mask[center_region > 0] > 0).mean()) if np.any(center_region > 0) else 0.0
        eye_overlap = float((component_mask[eye_region > 0] > 0).mean()) if np.any(eye_region > 0) else 0.0
        lower_overlap = float((component_mask[lower_region > 0] > 0).mean()) if np.any(lower_region > 0) else 0.0
        if center_overlap < 0.10 and max(eye_overlap, lower_overlap) < 0.12:
            continue

        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        score = (
            area
            + 180.0 * center_overlap
            + 120.0 * max(eye_overlap, lower_overlap)
            - 100.0 * abs(center_x - (w / 2.0)) / max(1.0, w / 2.0)
            - 80.0 * abs(center_y - (h / 2.0)) / max(1.0, h / 2.0)
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return np.zeros((h, w), dtype=np.float32)

    output = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(output, [best_contour], -1, 255, thickness=-1)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(output.astype(np.float32) / 255.0, (9, 9), 0)


def _estimate_landmark_synthetic_panel_mask(
    face_rgb: np.ndarray,
    face_landmarks: np.ndarray,
    texture_energy: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    left_eye, right_eye, nose, left_mouth, right_mouth = face_landmarks
    h, w = face_rgb.shape[:2]
    feature_left = float(min(left_eye[0], left_mouth[0]))
    feature_right = float(max(right_eye[0], right_mouth[0]))
    eye_span = float(abs(right_eye[0] - left_eye[0]))
    mouth_span = float(abs(right_mouth[0] - left_mouth[0]))
    face_width = max(1.0, feature_right - feature_left, eye_span * 2.25, mouth_span * 2.0)

    search = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, int(feature_left - 0.18 * face_width))
    x2 = min(w, int(feature_right + 0.18 * face_width))
    y1 = max(0, int(nose[1] - 0.12 * face_width))
    y2 = min(h, int(max(left_mouth[1], right_mouth[1]) + 0.56 * face_width))
    search[y1:y2, x1:x2] = 255

    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    neutral_panel = (
        (hsv[:, :, 1] < 58)
        & (hsv[:, :, 2] > 142)
        & (texture_energy < 12.5)
        & (cr >= 116)
        & (cr <= 150)
        & (cb >= 104)
        & (cb <= 142)
    )
    bright_paper = (
        (hsv[:, :, 1] < 46)
        & (hsv[:, :, 2] > 158)
        & (texture_energy < 12.0)
        & (cr <= 155)
        & (cb >= 100)
        & (cb <= 146)
    )
    candidate = (neutral_panel | bright_paper) & (search > 0)
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    best_contour = None
    best_score = float("-inf")
    image_area = float(h * w)
    center_x = w / 2.0
    expected_y = float((nose[1] + max(left_mouth[1], right_mouth[1])) / 2.0)
    for contour in cv2.findContours(candidate_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = float(cv2.contourArea(contour))
        if area < max(35.0, 0.0035 * image_area):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 0.16 * w or bh < 0.08 * h:
            continue
        if bw > 0.58 * w or bh > 0.46 * h:
            continue
        aspect = bw / max(1.0, float(bh))
        fill_ratio = area / max(1.0, float(bw * bh))
        if aspect < 0.55 or aspect > 2.25 or fill_ratio < 0.35:
            continue
        if y < int(nose[1] - 0.20 * face_width):
            continue
        if y + bh < int(nose[1] + 0.08 * face_width):
            continue

        component_center_x = x + bw / 2.0
        component_center_y = y + bh / 2.0
        center_penalty = abs(component_center_x - center_x) / max(1.0, center_x)
        y_penalty = abs(component_center_y - expected_y) / max(1.0, h)
        score = area - 180.0 * center_penalty - 130.0 * y_penalty + 60.0 * fill_ratio
        if score > best_score:
            best_score = score
            best_contour = contour

    output = np.zeros((h, w), dtype=np.uint8)
    if best_contour is None:
        return output.astype(np.float32)

    x, y, bw, bh = cv2.boundingRect(best_contour)
    pad_x = max(2, int(0.035 * face_width))
    pad_top = max(3, int(0.08 * face_width))
    pad_bottom = max(2, int(0.04 * face_width))
    x1 = max(0, x - pad_x)
    x2 = min(w, x + bw + pad_x)
    y1 = max(0, y - pad_top)
    y2 = min(h, y + bh + pad_bottom)
    output[y1:y2, x1:x2] = 255
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(output.astype(np.float32) / 255.0, (3, 3), 0)


def _estimate_occlusion(face_rgb: np.ndarray, face_landmarks: np.ndarray | None) -> tuple[np.ndarray, bool, str, float]:
    bright_panel_mask = _estimate_bright_panel_mask(face_rgb)
    if face_landmarks is None:
        coverage = float(bright_panel_mask.mean())
        region = "mixed" if coverage > 0.015 else "none"
        return bright_panel_mask, coverage > 0.015, region, coverage

    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    left_eye, right_eye, nose, left_mouth, right_mouth = face_landmarks
    face_width = max(1.0, right_mouth[0] - left_eye[0])

    lower_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.10 * face_width)), max(0, int(nose[1] - 0.02 * face_width))],
            [min(111, int(right_eye[0] + 0.10 * face_width)), max(0, int(nose[1] - 0.02 * face_width))],
            [min(111, int(right_mouth[0] + 0.18 * face_width)), min(111, int(right_mouth[1] + 0.32 * face_width))],
            [max(0, int(left_mouth[0] - 0.18 * face_width)), min(111, int(left_mouth[1] + 0.32 * face_width))],
        ],
        dtype=np.int32,
    )
    eye_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.20 * face_width))],
            [min(111, int(right_eye[0] + 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.20 * face_width))],
            [min(111, int(right_eye[0] + 0.18 * face_width)), min(111, int(nose[1] + 0.02 * face_width))],
            [max(0, int(left_eye[0] - 0.18 * face_width)), min(111, int(nose[1] + 0.02 * face_width))],
        ],
        dtype=np.int32,
    )
    center_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.16 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.05 * face_width))],
            [min(111, int(right_eye[0] + 0.16 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.05 * face_width))],
            [min(111, int(right_mouth[0] + 0.12 * face_width)), min(111, int(right_mouth[1] + 0.14 * face_width))],
            [max(0, int(left_mouth[0] - 0.12 * face_width)), min(111, int(left_mouth[1] + 0.14 * face_width))],
        ],
        dtype=np.int32,
    )
    focus_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.02 * face_width))],
            [min(111, int(right_eye[0] + 0.22 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.02 * face_width))],
            [min(111, int(right_mouth[0] + 0.16 * face_width)), min(111, int(right_mouth[1] + 0.18 * face_width))],
            [max(0, int(left_mouth[0] - 0.16 * face_width)), min(111, int(left_mouth[1] + 0.18 * face_width))],
        ],
        dtype=np.int32,
    )

    lower_region = _polygon_mask(face_rgb.shape[:2], lower_polygon)
    eye_region = _polygon_mask(face_rgb.shape[:2], eye_polygon)
    center_region = _polygon_mask(face_rgb.shape[:2], center_polygon)
    focus_region = _polygon_mask(face_rgb.shape[:2], focus_polygon)
    upper_reference_polygon = np.array(
        [
            [max(0, int(left_eye[0] - 0.14 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.28 * face_width))],
            [min(111, int(right_eye[0] + 0.14 * face_width)), max(0, int(min(left_eye[1], right_eye[1]) - 0.28 * face_width))],
            [min(111, int(right_eye[0] + 0.08 * face_width)), min(111, int(nose[1] - 0.04 * face_width))],
            [max(0, int(left_eye[0] - 0.08 * face_width)), min(111, int(nose[1] - 0.04 * face_width))],
        ],
        dtype=np.int32,
    )
    upper_reference_region = _polygon_mask(face_rgb.shape[:2], upper_reference_polygon)

    cloth_mask = (
        (hsv[:, :, 1] < 70)
        & (hsv[:, :, 2] > 130)
        & (lower_region > 0)
    )
    upper_values = hsv[:, :, 2][upper_reference_region > 0]
    adaptive_lower_dark_threshold = 75.0
    if upper_values.size > 0:
        adaptive_lower_dark_threshold = float(np.clip(np.percentile(upper_values, 60) - 35.0, 45.0, 120.0))
    lower_dark_mask = (
        (hsv[:, :, 2] < adaptive_lower_dark_threshold)
        & (hsv[:, :, 1] < 135)
        & (lower_region > 0)
    )
    grayscale = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    texture_energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(grayscale, cv2.CV_32F)), (5, 5), 0)
    synthetic_panel_mask = _estimate_landmark_synthetic_panel_mask(face_rgb, face_landmarks, texture_energy)
    synthetic_panel_coverage = float((synthetic_panel_mask > 0.15).mean())
    if synthetic_panel_coverage > 0.012:
        lower_coverage = (
            float((synthetic_panel_mask[lower_region > 0] > 0.15).mean()) if np.any(lower_region > 0) else 0.0
        )
        eye_coverage = (
            float((synthetic_panel_mask[eye_region > 0] > 0.15).mean()) if np.any(eye_region > 0) else 0.0
        )
        region = "mixed" if lower_coverage > 0.08 and eye_coverage > 0.08 else "lower-face"
        return synthetic_panel_mask, True, region, synthetic_panel_coverage

    skin_occluder_mask = _estimate_skin_occluder_mask(
        face_rgb,
        focus_region,
        center_region,
        eye_region,
        lower_region,
        texture_energy,
    )
    bright_neutral_mask = (
        (hsv[:, :, 1] < 60)
        & (hsv[:, :, 2] > 150)
        & (texture_energy < 8.0)
        & (focus_region > 0)
    )
    bright_neutral_mask_u8 = np.where(bright_neutral_mask, 255, 0).astype(np.uint8)
    bright_neutral_mask_u8 = cv2.morphologyEx(bright_neutral_mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    opaque_panel_mask = np.zeros_like(bright_neutral_mask_u8)
    contours, _ = cv2.findContours(bright_neutral_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 35:
            x, y, w, h = cv2.boundingRect(largest)
            pad_x = max(2, int(0.06 * face_width))
            pad_y = max(2, int(0.04 * face_width))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(face_rgb.shape[1], x + w + pad_x)
            y2 = min(face_rgb.shape[0], y + h + pad_y)
            opaque_panel_mask[y1:y2, x1:x2] = 255

    # Rectangular paper/surgical-mask panels often extend below the landmark
    # focus polygon. Detect them in a wider lower-face search area so the
    # final blend replaces the whole occluder instead of only a V-shaped top.
    h_img, w_img = face_rgb.shape[:2]
    panel_search = np.zeros((h_img, w_img), dtype=np.uint8)
    panel_top = max(0, int(nose[1] - 0.10 * face_width))
    panel_left = max(0, int(left_eye[0] - 0.30 * face_width))
    panel_right = min(w_img, int(right_eye[0] + 0.30 * face_width))
    panel_search[panel_top:h_img, panel_left:panel_right] = 255
    wide_panel_candidate = (
        (hsv[:, :, 1] < 82)
        & (hsv[:, :, 2] > 118)
        & (texture_energy < 16.0)
        & (panel_search > 0)
    )
    wide_panel_u8 = np.where(wide_panel_candidate, 255, 0).astype(np.uint8)
    wide_panel_u8 = cv2.morphologyEx(wide_panel_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    wide_panel_u8 = cv2.morphologyEx(wide_panel_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    wide_contours, _ = cv2.findContours(wide_panel_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if wide_contours:
        best_panel = None
        best_panel_score = float("-inf")
        for contour in wide_contours:
            area = float(cv2.contourArea(contour))
            if area < max(70.0, 0.006 * h_img * w_img):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 0.18 * w_img or bh < 0.10 * h_img:
                continue
            center_x = x + bw / 2.0
            center_penalty = abs(center_x - (w_img / 2.0)) / max(1.0, w_img / 2.0)
            bottom_bonus = 0.8 if y + bh > int(0.70 * h_img) else 0.0
            score = area * (1.0 + bottom_bonus) - 180.0 * center_penalty
            if score > best_panel_score:
                best_panel_score = score
                best_panel = (x, y, bw, bh)

        if best_panel is not None:
            x, y, bw, bh = best_panel
            pad_x = max(3, int(0.05 * face_width))
            pad_y = max(2, int(0.04 * face_width))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + bw + pad_x)
            y2 = min(h_img, y + bh + pad_y)
            if y + bh > int(0.78 * h_img):
                y2 = h_img
            opaque_panel_mask[y1:y2, x1:x2] = 255
    dark_mask = (
        (hsv[:, :, 2] < 75)
        & (eye_region > 0)
    )

    panel_present = (float((opaque_panel_mask > 0).mean()) > 0.015) or (float((bright_panel_mask > 0.15).mean()) > 0.015)
    lower_dark_component = np.zeros_like(lower_dark_mask, dtype=bool) if panel_present else lower_dark_mask
    skin_component = np.zeros_like(skin_occluder_mask, dtype=bool) if panel_present else (skin_occluder_mask > 0.18)
    dark_component = np.zeros_like(dark_mask, dtype=bool) if panel_present else dark_mask

    combined = np.where(
        cloth_mask
        | lower_dark_component
        | (opaque_panel_mask > 0)
        | (bright_panel_mask > 0.15)
        | skin_component
        | dark_component,
        255,
        0,
    ).astype(np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coverage = float(combined.mean() / 255.0)
    is_occluded = coverage > 0.012

    region = "none"
    lower_coverage = float((combined[lower_region > 0] > 0).mean()) if np.any(lower_region > 0) else 0.0
    eye_coverage = float((combined[eye_region > 0] > 0).mean()) if np.any(eye_region > 0) else 0.0
    center_coverage = float((combined[center_region > 0] > 0).mean()) if np.any(center_region > 0) else 0.0
    if is_occluded:
        if lower_coverage > 0.08 and eye_coverage > 0.08:
            region = "mixed"
        elif center_coverage > 0.14 and eye_coverage > 0.04:
            region = "mixed"
        elif lower_coverage >= eye_coverage:
            region = "lower-face"
        else:
            region = "eye-region"

    soft_mask = cv2.GaussianBlur(combined.astype(np.float32) / 255.0, (9, 9), 0)
    return soft_mask, is_occluded, region, coverage


def _prepare_face_analysis(img_bgr: np.ndarray) -> dict:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reconstruction_rgb = cv2.resize(
        img_rgb,
        (RECONSTRUCTION_MODEL_IMAGE_SIZE, RECONSTRUCTION_MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    reconstruction_tensor = _rgb_image_to_face_tensor(reconstruction_rgb)
    display_occlusion_mask = _estimate_direct_panel_mask(img_rgb)
    direct_panel_mask_available = _is_usable_direct_panel_mask(display_occlusion_mask)
    boxes, _, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    face_tensor = mtcnn(img_rgb)
    detected_face = face_tensor is not None and boxes is not None and landmarks is not None and len(boxes) > 0

    if detected_face:
        face_tensor = _normalize_mtcnn_face_tensor(face_tensor)
        face_rgb = _tensor_to_rgb_image(face_tensor)
        aligned_boxes, _, aligned_landmarks = mtcnn.detect(face_rgb, landmarks=True)
        if aligned_boxes is not None and aligned_landmarks is not None and len(aligned_boxes) > 0:
            face_landmarks = np.clip(
                aligned_landmarks[0].astype(np.float32),
                0,
                RECOGNITION_ALIGNMENT_IMAGE_SIZE - 1,
            )
        else:
            face_landmarks = _landmarks_to_face_coords(boxes[0], landmarks[0])
        reconstruction_landmarks = _scale_landmarks_to_resized_image(landmarks[0], img_rgb.shape[:2])
    else:
        face_rgb = reconstruction_rgb
        face_tensor = reconstruction_tensor
        face_landmarks = None
        reconstruction_landmarks = None

    recognition_mask, recognition_is_occluded, recognition_region, recognition_ratio = _estimate_occlusion(face_rgb, face_landmarks)
    occlusion_mask, is_occluded, occlusion_region, occlusion_ratio = _estimate_occlusion(
        reconstruction_rgb, reconstruction_landmarks
    )
    (
        recognition_mask,
        recognition_is_occluded,
        recognition_region,
        recognition_ratio,
        recognition_boundary_jump,
    ) = _suppress_low_confidence_rectangular_occlusion(
        face_rgb,
        recognition_mask,
        recognition_is_occluded,
        recognition_region,
        recognition_ratio,
    )
    (
        occlusion_mask,
        is_occluded,
        occlusion_region,
        occlusion_ratio,
        occlusion_boundary_jump,
    ) = _suppress_low_confidence_rectangular_occlusion(
        reconstruction_rgb,
        occlusion_mask,
        is_occluded,
        occlusion_region,
        occlusion_ratio,
    )
    if direct_panel_mask_available:
        direct_occlusion_mask = cv2.resize(
            _strict_occlusion_mask_array(display_occlusion_mask),
            (RECONSTRUCTION_MODEL_IMAGE_SIZE, RECONSTRUCTION_MODEL_IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        direct_occlusion_mask = _strict_occlusion_mask_array(direct_occlusion_mask)
        direct_boundary_jump = _occlusion_boundary_jump(reconstruction_rgb, direct_occlusion_mask)
        direct_flat_panel_evidence = _has_flat_rectangular_panel_evidence(reconstruction_rgb, direct_occlusion_mask)
        direct_visible_face_detail = _has_visible_face_detail_evidence(reconstruction_rgb, direct_occlusion_mask)
        direct_panel_mask_available = (
            direct_boundary_jump >= DIRECT_PANEL_BOUNDARY_JUMP_THRESHOLD
            and direct_flat_panel_evidence
            and not direct_visible_face_detail
        )

    if direct_panel_mask_available:
        direct_occlusion_mask = _identity_preserving_occlusion_mask(
            _expand_rectangular_panel_mask(direct_occlusion_mask)
        )
        occlusion_mask = direct_occlusion_mask
        is_occluded = True
        occlusion_region = _region_from_mask(occlusion_mask)
        occlusion_ratio = float(occlusion_mask.mean())
        occlusion_boundary_jump = direct_boundary_jump
        display_occlusion_mask = _identity_preserving_occlusion_mask(
            _expand_rectangular_panel_mask(display_occlusion_mask)
        )
        occlusion_mask_source = "direct-upload-panel"
    else:
        display_occlusion_mask = cv2.resize(
            _strict_occlusion_mask_array(occlusion_mask),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        if RECONSTRUCT_ONLY_OCCLUDED_REGION:
            occlusion_mask = _identity_preserving_occlusion_mask(occlusion_mask)
            display_occlusion_mask = _identity_preserving_occlusion_mask(display_occlusion_mask)
            occlusion_ratio = float(occlusion_mask.mean())
        occlusion_mask_source = "resized-reconstruction-mask"
    return {
        "face_tensor": face_tensor,
        "detected_face": detected_face,
        "face_rgb": face_rgb,
        "face_landmarks": face_landmarks,
        "occlusion_mask": occlusion_mask,
        "recognition_mask": recognition_mask,
        "recognition_is_occluded": recognition_is_occluded,
        "recognition_region": recognition_region,
        "recognition_ratio": recognition_ratio,
        "recognition_boundary_jump": recognition_boundary_jump,
        "is_occluded": is_occluded,
        "occlusion_region": occlusion_region,
        "occlusion_ratio": occlusion_ratio,
        "occlusion_boundary_jump": occlusion_boundary_jump,
        "display_occlusion_mask": display_occlusion_mask,
        "display_occlusion_ratio": float(display_occlusion_mask.mean()),
        "occlusion_mask_source": occlusion_mask_source,
        "reconstruction_tensor": reconstruction_tensor,
        "reconstruction_rgb": reconstruction_rgb,
    }


def _reconstruction_view_from_analysis(analysis: dict) -> tuple[torch.Tensor, np.ndarray, np.ndarray, bool, str, float, bool]:
    """Use the training-style resized view for reconstruction.

    The reconstruction model was trained on resized LFW/deepfunneled-style
    images, not the tighter MTCNN aligned crop used for recognition. Using the
    recognition crop here makes the output look zoomed and shifts the mask.
    """
    return (
        analysis["reconstruction_tensor"],
        analysis["reconstruction_rgb"],
        analysis["occlusion_mask"],
        analysis["is_occluded"],
        analysis["occlusion_region"],
        float(analysis["occlusion_ratio"]),
        True,
    )


def _reconstruct_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray | None = None) -> tuple[torch.Tensor, np.ndarray]:
    with torch.no_grad():
        mask_tensor = None
        model_input = face_tensor
        if occlusion_mask is not None:
            mask_tensor = _occlusion_mask_to_tensor(
                occlusion_mask, device_target=face_tensor.device, dtype=face_tensor.dtype, batch_size=face_tensor.shape[0]
            )
            mask_tensor = _harden_mask_tensor(mask_tensor)
            model_input = _sanitize_face_tensor_for_reconstruction(face_tensor, mask_tensor)
        reconstructed = degan_model(model_input, mask_tensor)

    reconstructed_rgb = _tensor_to_rgb_image(reconstructed)
    return reconstructed, reconstructed_rgb


def _new_refiner_info(
    *,
    requested: bool,
    status: str | None = None,
    applied: bool = False,
) -> dict:
    return {
        "requested": bool(requested),
        "loaded": bool(mtr_refiner_loaded),
        "applied": bool(applied),
        "status": status or ("disabled" if not requested else "not-evaluated"),
    }


def _finish_refiner_result(
    face_tensor: torch.Tensor,
    face_rgb: np.ndarray,
    refiner_info: dict,
    *,
    return_info: bool,
):
    if return_info:
        return face_tensor, face_rgb, refiner_info
    return face_tensor, face_rgb


def _degan_oan_info(
    *,
    requested: bool,
    status: str | None = None,
    applied: bool = False,
    accepted: bool = False,
    fallback_score: float | None = None,
    candidate_score: float | None = None,
    fallback_similarity: float | None = None,
    candidate_similarity: float | None = None,
    mean_gate: float | None = None,
) -> dict:
    return {
        "requested": bool(requested),
        "enabled": bool(DEGAN_OAN_ENABLED),
        "loaded": bool(degan_oan_loaded),
        "applied": bool(applied),
        "accepted": bool(accepted),
        "status": status or ("disabled" if not requested else "not-evaluated"),
        "fallback_score": fallback_score,
        "candidate_score": candidate_score,
        "fallback_similarity": fallback_similarity,
        "candidate_similarity": candidate_similarity,
        "mean_gate": mean_gate,
    }


def _rgb_patch_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def _context_fill_patch(masked_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    hard = (mask_patch > 0.5).astype(np.uint8) * 255
    if hard.max() == 0:
        return masked_patch.astype(np.uint8).copy()
    bgr = cv2.cvtColor(masked_patch.astype(np.uint8), cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hard, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def _degan_oan_patch_resize_interpolation(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> int:
    mode = DEGAN_OAN_PATCH_RESIZE_INTERPOLATION
    if mode == "nearest":
        return cv2.INTER_NEAREST
    if mode == "linear":
        return cv2.INTER_LINEAR
    if mode == "area":
        return cv2.INTER_AREA
    if mode == "lanczos":
        return cv2.INTER_LANCZOS4
    if mode == "auto":
        src_h, src_w = source_shape[:2]
        dst_h, dst_w = target_shape[:2]
        return cv2.INTER_AREA if dst_h < src_h or dst_w < src_w else cv2.INTER_CUBIC
    return cv2.INTER_CUBIC


def _square_crop_box(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    context: float = 0.55,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    size = max(64, int(round(max(bw, bh) * (1.0 + context))))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    left = int(round(cx - size / 2))
    top = int(round(cy - size / 2))
    right = left + size
    bottom = top + size
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def _recognition_top_similarity_for_rgb(rgb: np.ndarray) -> float:
    try:
        fallback_rgb = cv2.resize(
            rgb.astype(np.uint8),
            (RECOGNITION_ALIGNMENT_IMAGE_SIZE, RECOGNITION_ALIGNMENT_IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        probe_tensor, _, _ = _recognition_probe_from_reconstruction(
            rgb.astype(np.uint8),
            _rgb_image_to_face_tensor(fallback_rgb),
            fallback_rgb,
        )
        matches = _rank_identity_matches(probe_tensor, top_k=1)
    except Exception:
        return float("-inf")
    if not matches:
        return float("-inf")
    return float(matches[0]["similarity"])


def _degan_oan_refine_display(
    masked_rgb: np.ndarray,
    degan_display_rgb: np.ndarray,
    display_mask: np.ndarray,
    *,
    enabled: bool | None = None,
) -> tuple[np.ndarray, dict]:
    requested = DEGAN_OAN_ENABLED if enabled is None else bool(enabled)
    if not requested:
        return degan_display_rgb.astype(np.uint8), _degan_oan_info(requested=False)
    if not degan_oan_loaded:
        return degan_display_rgb.astype(np.uint8), _degan_oan_info(requested=True, status="weights-not-loaded")

    hard_mask = _strict_occlusion_mask_array(display_mask).astype(np.float32)
    bbox = _mask_bbox(hard_mask)
    if bbox is None or hard_mask.max() <= 0.5:
        return degan_display_rgb.astype(np.uint8), _degan_oan_info(requested=True, status="not-needed-no-mask")

    h, w = masked_rgb.shape[:2]
    x1, y1, x2, y2 = _square_crop_box(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return degan_display_rgb.astype(np.uint8), _degan_oan_info(requested=True, status="invalid-crop")

    patch_size = int(DEGAN_OAN_PATCH_SIZE)
    masked_patch = cv2.resize(masked_rgb[y1:y2, x1:x2].astype(np.uint8), (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    degan_patch = cv2.resize(degan_display_rgb[y1:y2, x1:x2].astype(np.uint8), (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    mask_patch = cv2.resize(hard_mask[y1:y2, x1:x2].astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    mask_patch = (mask_patch > 0.5).astype(np.float32)
    context_patch = _context_fill_patch(masked_patch, mask_patch)

    mask_tensor = torch.from_numpy(mask_patch[None, None].astype(np.float32)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        refined_patch_tensor, _raw_patch_tensor, gate_tensor = degan_oan_refiner(
            _rgb_patch_to_tensor(masked_patch),
            mask_tensor,
            _rgb_patch_to_tensor(degan_patch),
            _rgb_patch_to_tensor(context_patch),
            return_gate=True,
        )
    refined_patch = _tensor_to_rgb_image(refined_patch_tensor)
    raw_refined_patch = _tensor_to_rgb_image(_raw_patch_tensor)
    patch_for_resize = raw_refined_patch if DEGAN_OAN_RAW_REGION_COMPOSITING else refined_patch
    resize_interpolation = _degan_oan_patch_resize_interpolation(
        patch_for_resize.shape[:2],
        (y2 - y1, x2 - x1),
    )
    resized_patch = cv2.resize(patch_for_resize, (x2 - x1, y2 - y1), interpolation=resize_interpolation)
    candidate = masked_rgb.astype(np.uint8).copy()
    crop_mask = hard_mask[y1:y2, x1:x2][..., None].astype(np.float32)
    candidate[y1:y2, x1:x2] = np.clip(
        candidate[y1:y2, x1:x2].astype(np.float32) * (1.0 - crop_mask)
        + resized_patch.astype(np.float32) * crop_mask,
        0,
        255,
    ).astype(np.uint8)
    candidate = np.where(hard_mask[..., None] > 0.5, candidate, masked_rgb).astype(np.uint8)

    fallback = np.where(hard_mask[..., None] > 0.5, degan_display_rgb, masked_rgb).astype(np.uint8)
    fallback_score = _candidate_quality_score(fallback, masked_rgb.astype(np.uint8), hard_mask)
    candidate_score = _candidate_quality_score(candidate, masked_rgb.astype(np.uint8), hard_mask)
    fallback_similarity = _recognition_top_similarity_for_rgb(fallback)
    candidate_similarity = _recognition_top_similarity_for_rgb(candidate)
    score_gain = fallback_score - candidate_score
    similarity_gain = candidate_similarity - fallback_similarity
    accept_by_quality = score_gain >= DEGAN_OAN_ACCEPT_MIN_SCORE_GAIN
    accept_by_recognition = similarity_gain >= 0.02 and candidate_score <= fallback_score + 12.0
    accepted = bool(accept_by_quality or accept_by_recognition)
    mean_gate = float(gate_tensor.detach().mean().item())

    if not accepted:
        return fallback, _degan_oan_info(
            requested=True,
            status="rejected-confidence-gate",
            applied=True,
            accepted=False,
            fallback_score=float(fallback_score),
            candidate_score=float(candidate_score),
            fallback_similarity=float(fallback_similarity),
            candidate_similarity=float(candidate_similarity),
            mean_gate=mean_gate,
        )

    return candidate, _degan_oan_info(
        requested=True,
        status="accepted",
        applied=True,
        accepted=True,
        fallback_score=float(fallback_score),
        candidate_score=float(candidate_score),
        fallback_similarity=float(fallback_similarity),
        candidate_similarity=float(candidate_similarity),
        mean_gate=mean_gate,
    )


def _refine_reconstructed_face(
    face_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
    occlusion_mask: np.ndarray,
    fallback_rgb: np.ndarray | None = None,
    *,
    return_info: bool = False,
):
    refiner_info = _new_refiner_info(requested=True, status="applied", applied=True)
    mask_tensor = _occlusion_mask_to_tensor(
        occlusion_mask,
        device_target=face_tensor.device,
        dtype=face_tensor.dtype,
        batch_size=face_tensor.shape[0],
    )
    mask_tensor = _harden_mask_tensor(mask_tensor)
    with torch.no_grad():
        refined_tensor = mtr_refiner(face_tensor, reconstructed_tensor, mask_tensor)
    refined_rgb = _tensor_to_rgb_image(refined_tensor)
    original_rgb = _tensor_to_rgb_image(face_tensor)
    refined_rgb = _postprocess_reconstructed_face(original_rgb, refined_rgb, occlusion_mask)
    if fallback_rgb is not None:
        hard_mask = _harden_mask_array(occlusion_mask)
        # The current MTR-UNet checkpoint is destructive for large rectangular
        # lower-face masks. Keep the pipeline pure by falling back to DE-GAN
        # instead of showing MTR ghosting/panel artifacts.
        if float(hard_mask.mean()) > 0.12 or _is_rectangular_panel_mask(hard_mask):
            refiner_info = _new_refiner_info(
                requested=True,
                status="skipped-large-rectangular-mask-guard",
                applied=False,
            )
            return _finish_refiner_result(
                _rgb_image_to_face_tensor(fallback_rgb),
                fallback_rgb,
                refiner_info,
                return_info=return_info,
            )
        refiner_score = _candidate_quality_score(refined_rgb, original_rgb, occlusion_mask)
        fallback_score = _candidate_quality_score(fallback_rgb, original_rgb, occlusion_mask)
        if refiner_score > fallback_score + 4.0:
            refined_rgb = fallback_rgb
            refiner_info = _new_refiner_info(
                requested=True,
                status="rejected-quality-gate",
                applied=False,
            )
    return _finish_refiner_result(
        _rgb_image_to_face_tensor(refined_rgb),
        refined_rgb,
        refiner_info,
        return_info=return_info,
    )


def _blend_reconstructed_face(
    face_tensor: torch.Tensor,
    occlusion_mask: np.ndarray,
    *,
    enhance_patch: bool = False,
    use_refiner: bool = False,
    original_rgb_override: np.ndarray | None = None,
    return_info: bool = False,
):
    refiner_info = _new_refiner_info(requested=use_refiner)
    reconstructed, reconstructed_rgb = _reconstruct_face(face_tensor, occlusion_mask)

    if original_rgb_override is not None:
        original_rgb = original_rgb_override.astype(np.uint8).astype(np.float32)
    else:
        original_rgb = _tensor_to_rgb_image(face_tensor).astype(np.float32)
    hard_mask = _harden_mask_array(occlusion_mask)
    if _is_rectangular_panel_mask(hard_mask):
        if use_refiner:
            refiner_info = _new_refiner_info(
                requested=True,
                status="skipped-rectangular-panel-guard",
                applied=False,
            )
        blended_rgb = _blend_rgb_from_reconstruction(
            original_rgb.astype(np.uint8),
            reconstructed_rgb.astype(np.uint8),
            occlusion_mask,
            enhance_patch=enhance_patch,
        )
        return _finish_refiner_result(
            _rgb_image_to_face_tensor(blended_rgb),
            blended_rgb,
            refiner_info,
            return_info=return_info,
        )

    degan_candidate_rgb = _blend_rgb_from_reconstruction(
        original_rgb.astype(np.uint8),
        reconstructed_rgb.astype(np.uint8),
        occlusion_mask,
        enhance_patch=enhance_patch,
    )
    if use_refiner:
        if mtr_refiner_loaded:
            return _refine_reconstructed_face(
                face_tensor,
                reconstructed,
                occlusion_mask,
                fallback_rgb=degan_candidate_rgb,
                return_info=return_info,
            )
        refiner_info = _new_refiner_info(
            requested=True,
            status="weights-not-loaded",
            applied=False,
        )

    blended_rgb = degan_candidate_rgb.astype(np.uint8)
    blended_tensor = _rgb_image_to_face_tensor(blended_rgb)
    return _finish_refiner_result(blended_tensor, blended_rgb, refiner_info, return_info=return_info)


def _extract_region_only_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray, fill_value: float = 0.0) -> tuple[torch.Tensor, np.ndarray]:
    mask = np.clip(occlusion_mask, 0.0, 1.0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask).to(device=device, dtype=face_tensor.dtype).unsqueeze(0).unsqueeze(0)
    mask_tensor = mask_tensor.expand(face_tensor.shape[0], face_tensor.shape[1], -1, -1)
    region_tensor = face_tensor * mask_tensor + fill_value * (1.0 - mask_tensor)
    region_rgb = _tensor_to_rgb_image(region_tensor)
    return region_tensor, region_rgb


def _reconstructed_region_only_face(face_tensor: torch.Tensor, occlusion_mask: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    reconstructed, _ = _reconstruct_face(face_tensor, occlusion_mask)
    return _extract_region_only_face(reconstructed, occlusion_mask)


def _load_image_tensor_from_path(image_path: Path) -> torch.Tensor | None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    tensor, _ = _prepare_face_tensor(img_bgr)
    return tensor


def _load_rgb_image_from_path(image_path: Path) -> np.ndarray | None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _recognition_probe_from_reconstruction(
    reconstructed_rgb: np.ndarray,
    fallback_tensor: torch.Tensor,
    fallback_rgb: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, bool]:
    reconstructed_bgr = cv2.cvtColor(reconstructed_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    probe_tensor, aligned = _prepare_face_tensor(reconstructed_bgr)
    if aligned:
        return probe_tensor, _tensor_to_rgb_image(probe_tensor), True
    return fallback_tensor, fallback_rgb, False


def _tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _enhance_reconstructed_patch(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(occlusion_mask.astype(np.float32), 0.0, 1.0)
    mask_u8 = (mask * 255).astype(np.uint8)
    if mask_u8.max() == 0:
        return reconstructed_rgb

    nonzero = np.argwhere(mask_u8 > 16)
    if nonzero.size == 0:
        return reconstructed_rgb

    y1, x1 = nonzero.min(axis=0)
    y2, x2 = nonzero.max(axis=0) + 1
    if (y2 - y1) < 8 or (x2 - x1) < 8:
        return reconstructed_rgb

    source_patch = reconstructed_rgb[y1:y2, x1:x2].copy()
    local_mask = mask[y1:y2, x1:x2]
    h, w = original_rgb.shape[:2]
    pad_y = max(8, int(0.14 * (y2 - y1)))
    pad_x = max(8, int(0.14 * (x2 - x1)))
    sample_y1 = max(0, y1 - pad_y)
    sample_y2 = min(h, y2 + pad_y)
    sample_x1 = max(0, x1 - pad_x)
    sample_x2 = min(w, x2 + pad_x)

    source_gray = cv2.cvtColor(source_patch, cv2.COLOR_RGB2GRAY)
    source_texture = float(cv2.Laplacian(source_gray, cv2.CV_32F).var())
    target_texture = np.clip(70.0 / max(source_texture, 12.0), 0.9, 1.65)

    denoised_bgr = cv2.bilateralFilter(cv2.cvtColor(source_patch, cv2.COLOR_RGB2BGR), 3, 14, 14)
    denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    upscaled = cv2.resize(denoised, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up_blur = cv2.GaussianBlur(upscaled, (0, 0), 0.7)
    up_sharp = cv2.addWeighted(upscaled, 1.22 * target_texture, up_blur, -0.22 * target_texture, 1.0)
    sharpened = cv2.resize(np.clip(up_sharp, 0, 255).astype(np.uint8), (source_patch.shape[1], source_patch.shape[0]), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(sharpened, cv2.CV_32F, ksize=3)
    sharpened = np.clip(sharpened.astype(np.float32) + 0.14 * target_texture * laplacian, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    l_channel = cv2.convertScaleAbs(l_channel, alpha=1.04, beta=2)
    sharpened = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)

    # Match the reconstructed patch color to nearby visible skin so the patch
    # returns closer to the person's original complexion.
    sample_region = original_rgb[sample_y1:sample_y2, sample_x1:sample_x2].copy()
    sample_mask = mask[sample_y1:sample_y2, sample_x1:sample_x2]
    inpaint_mask = (sample_mask > 0.12).astype(np.uint8) * 255
    donor_patch = None
    if inpaint_mask.max() > 0:
        donor_bgr = cv2.inpaint(
            cv2.cvtColor(sample_region, cv2.COLOR_RGB2BGR),
            inpaint_mask,
            3,
            cv2.INPAINT_TELEA,
        )
        donor_rgb = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2RGB)
        donor_patch = donor_rgb[y1 - sample_y1:y2 - sample_y1, x1 - sample_x1:x2 - sample_x1].copy()
    hsv = cv2.cvtColor(sample_region, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(sample_region, cv2.COLOR_RGB2YCrCb)
    skin_mask = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 18)
        & (hsv[:, :, 1] <= 185)
        & (ycrcb[:, :, 1] >= 132)
        & (ycrcb[:, :, 1] <= 180)
        & (ycrcb[:, :, 2] >= 80)
        & (ycrcb[:, :, 2] <= 135)
        & (sample_mask < 0.08)
    )
    if skin_mask.sum() >= 40:
        source_lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(sample_region, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_pixels = target_lab[skin_mask]
        source_pixels = source_lab.reshape(-1, 3)

        target_mean = target_pixels.mean(axis=0)
        source_mean = source_pixels.mean(axis=0)
        target_std = target_pixels.std(axis=0) + 1e-6
        source_std = source_pixels.std(axis=0) + 1e-6

        remapped = source_lab.copy()
        target_l_p90 = float(np.percentile(target_pixels[:, 0], 90))
        target_l_p97 = float(np.percentile(target_pixels[:, 0], 97))
        source_l_mean = float(source_mean[0])
        extra_l_strength = 0.92 if source_l_mean > target_mean[0] + 6.0 else 0.72
        for channel_idx, strength in ((0, extra_l_strength), (1, 0.86), (2, 0.86)):
            normalized = (source_lab[:, :, channel_idx] - source_mean[channel_idx]) / source_std[channel_idx]
            matched = normalized * target_std[channel_idx] + target_mean[channel_idx]
            remapped[:, :, channel_idx] = (
                source_lab[:, :, channel_idx] * (1.0 - strength)
                + matched * strength
            )
        # Compress high luminance outliers so the patch stays within the local
        # complexion range instead of keeping a washed-out rectangular highlight.
        remapped[:, :, 0] = np.where(
            remapped[:, :, 0] > target_l_p90,
            target_l_p90 + (remapped[:, :, 0] - target_l_p90) * 0.28,
            remapped[:, :, 0],
        )
        remapped[:, :, 0] = np.clip(remapped[:, :, 0], 0.0, target_l_p97 + 4.0)
        sharpened = cv2.cvtColor(np.clip(remapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Run a second lightweight Y-channel match against the nearby visible
        # face region to reduce exposure jumps inside the reconstructed patch.
        sharpened_ycrcb = cv2.cvtColor(sharpened, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        target_ycrcb = cv2.cvtColor(sample_region, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        target_y = target_ycrcb[:, :, 0][skin_mask]
        patch_y = sharpened_ycrcb[:, :, 0]
        if target_y.size >= 40:
            target_y_mean = float(target_y.mean())
            target_y_std = float(target_y.std() + 1e-6)
            patch_y_mean = float(patch_y.mean())
            patch_y_std = float(patch_y.std() + 1e-6)
            matched_y = (patch_y - patch_y_mean) * (target_y_std / patch_y_std) + target_y_mean
            blend_strength = 0.72 if patch_y_mean > target_y_mean + 5.0 else 0.5
            sharpened_ycrcb[:, :, 0] = patch_y * (1.0 - blend_strength) + matched_y * blend_strength
            high_clip = float(np.percentile(target_y, 97) + 5.0)
            sharpened_ycrcb[:, :, 0] = np.clip(sharpened_ycrcb[:, :, 0], 0.0, high_clip)
            sharpened = cv2.cvtColor(np.clip(sharpened_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    visible_context = original_rgb[sample_y1:sample_y2, sample_x1:sample_x2].copy()
    context_mask = mask[sample_y1:sample_y2, sample_x1:sample_x2] < 0.08
    if context_mask.sum() >= 50:
        context_ycrcb = cv2.cvtColor(visible_context, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        context_y = context_ycrcb[:, :, 0][context_mask]
        context_hsv = cv2.cvtColor(visible_context, cv2.COLOR_RGB2HSV).astype(np.float32)
        context_s = context_hsv[:, :, 1][context_mask]
        patch_ycrcb = cv2.cvtColor(sharpened, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        patch_y = patch_ycrcb[:, :, 0]
        patch_hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV).astype(np.float32)
        patch_s = patch_hsv[:, :, 1]
        context_mean = float(context_y.mean())
        context_p95 = float(np.percentile(context_y, 95))
        patch_mean = float(patch_y.mean())
        context_sat_mean = float(context_s.mean()) if context_s.size > 0 else 0.0
        patch_sat_mean = float(patch_s.mean())
        highlight_fraction = float((patch_y > context_p95 + 6.0).mean())
        overbright = patch_mean > context_mean + 7.0 or highlight_fraction > 0.08
        washed_out = patch_sat_mean < max(18.0, context_sat_mean * 0.62)
        if overbright:
            matched_y = patch_y - (patch_mean - context_mean) * 0.88
            matched_y = np.where(
                matched_y > context_p95,
                context_p95 + (matched_y - context_p95) * 0.18,
                matched_y,
            )
            patch_ycrcb[:, :, 0] = np.clip(matched_y, 0.0, context_p95 + 4.0)
            sharpened = cv2.cvtColor(np.clip(patch_ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        if washed_out:
            patch_hsv[:, :, 1] = np.clip(
                patch_s * 0.35 + context_sat_mean * 0.65,
                0.0,
                max(255.0, context_sat_mean + 20.0),
            )
            sharpened = cv2.cvtColor(np.clip(patch_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        if donor_patch is not None and (overbright or washed_out):
            donor_rgb = donor_patch.astype(np.float32)
            current_rgb = sharpened.astype(np.float32)
            donor_y = cv2.cvtColor(donor_patch, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
            donor_mean = float(donor_y.mean())
            severity = 0.35
            if patch_mean > context_mean + 11.0 or highlight_fraction > 0.16:
                severity = 0.55
            if washed_out and context_sat_mean > patch_sat_mean + 10.0:
                severity += 0.10
            severity = float(np.clip(severity, 0.0, max(0.0, min(0.65, PANEL_ENHANCE_DONOR_BLEND_MAX))))
            if severity > 0.0:
                blend = (local_mask[..., None] ** 1.25) * severity
                sharpened = np.clip(current_rgb * (1.0 - blend) + donor_rgb * blend, 0, 255).astype(np.uint8)

    if context_mask.sum() >= 50:
        context_gray = cv2.cvtColor(visible_context, cv2.COLOR_RGB2GRAY)
        context_texture = float(cv2.Laplacian(context_gray, cv2.CV_32F).var())
        patch_gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        patch_texture = float(cv2.Laplacian(patch_gray, cv2.CV_32F).var())
        if patch_texture > 1e-6 and context_texture > patch_texture:
            extra_gain = np.clip(context_texture / patch_texture, 1.0, 1.35)
            blur_small = cv2.GaussianBlur(sharpened, (0, 0), 0.6)
            sharpened = cv2.addWeighted(sharpened, 1.08 * extra_gain, blur_small, -0.08 * extra_gain, 0.5)

    enhanced = original_rgb.copy().astype(np.float32)
    alpha = (local_mask[..., None] ** 1.35) * 0.88
    enhanced[y1:y2, x1:x2] = (
        enhanced[y1:y2, x1:x2] * (1.0 - alpha)
        + sharpened.astype(np.float32) * alpha
    )
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    final_tone_strength = float(np.clip(PANEL_ENHANCE_FINAL_TONE_STRENGTH, 0.0, 1.0))
    if final_tone_strength <= 0.0:
        return enhanced
    return _apply_local_skin_tone_match(
        enhanced,
        original_rgb.astype(np.uint8),
        _harden_mask_array(mask),
        strength=final_tone_strength,
    )


def _encode_rgb_to_base64(img_rgb: np.ndarray) -> str:
    image = Image.fromarray(img_rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _encode_mask_to_base64(mask: np.ndarray) -> str:
    mask_u8 = np.clip(mask.astype(np.float32), 0.0, 1.0)
    mask_u8 = (mask_u8 * 255.0).astype(np.uint8)
    mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
    return _encode_rgb_to_base64(mask_rgb)


def _embed_faces(face_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mask_tensor = torch.zeros((face_tensor.shape[0], 1, face_tensor.shape[2], face_tensor.shape[3]), device=device, dtype=face_tensor.dtype)
        reconstructed = degan_model(face_tensor, mask_tensor)
        embedding = oan_model(reconstructed)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def _recognizer_input_from_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    rgb_image = _tensor_to_rgb_image(face_tensor)
    resized = cv2.resize(rgb_image, (160, 160), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    tensor = (tensor - 127.5) / 128.0
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _recognition_embedding(face_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        recognizer_input = _recognizer_input_from_tensor(face_tensor)
        embedding = recognition_model(recognizer_input)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def _iter_gallery_reference_images() -> list[tuple[str, Path]]:
    references: list[tuple[str, Path]] = []
    if not gallery_dir.exists():
        return references

    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        image_paths = sorted(
            path for path in person_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            references.append((person_dir.name, image_path))
    return references


def _build_gallery_index() -> tuple[torch.Tensor | None, list[str], list[str]]:
    names: list[str] = []
    embeddings: list[torch.Tensor] = []
    image_paths: list[str] = []

    references = _iter_gallery_reference_images()
    total_references = len(references)
    for index, (person_name, image_path) in enumerate(references, start=1):
        face_tensor = _load_image_tensor_from_path(image_path)
        if face_tensor is None:
            continue
        embedding = _recognition_embedding(face_tensor)
        names.append(person_name)
        embeddings.append(embedding.squeeze(0).cpu())
        image_paths.append(str(image_path))
        if index % 500 == 0 or index == total_references:
            print(
                f"  gallery index: {index}/{total_references} references processed, "
                f"{len(embeddings)} embeddings kept",
                flush=True,
            )

    if not embeddings:
        return None, [], []

    return torch.stack(embeddings), names, image_paths


def refresh_gallery_index() -> int:
    global gallery_embeddings, gallery_names, gallery_image_paths
    if gallery_cache_path.exists():
        try:
            cache = torch.load(gallery_cache_path, map_location="cpu")
            cached_names = cache.get("names", [])
            cached_embeddings = cache.get("embeddings")
            cached_image_paths = cache.get("image_paths", [])
            cached_gallery_dir = cache.get("gallery_dir")
            if (
                cache.get("version") == GALLERY_CACHE_VERSION
                and cached_gallery_dir == str(gallery_dir)
                and isinstance(cached_names, list)
                and isinstance(cached_embeddings, torch.Tensor)
                and isinstance(cached_image_paths, list)
                and len(cached_names) == cached_embeddings.shape[0]
                and len(cached_image_paths) == len(cached_names)
            ):
                gallery_embeddings = cached_embeddings
                gallery_names = cached_names
                gallery_image_paths = cached_image_paths
                print(f"Loaded gallery cache: {gallery_cache_path} ({len(gallery_names)} embeddings)", flush=True)
                return len(gallery_names)
        except Exception:
            gallery_cache_path.unlink(missing_ok=True)

    print(f"Building gallery cache from: {gallery_dir}", flush=True)
    gallery_embeddings, gallery_names, gallery_image_paths = _build_gallery_index()
    if gallery_embeddings is not None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": GALLERY_CACHE_VERSION,
                "gallery_dir": str(gallery_dir),
                "names": gallery_names,
                "embeddings": gallery_embeddings,
                "image_paths": gallery_image_paths,
            },
            gallery_cache_path,
        )
    return 0 if gallery_embeddings is None else len(gallery_names)


def _rank_identity_matches(face_tensor: torch.Tensor, top_k: int = 5) -> list[dict[str, object]]:
    if gallery_embeddings is None or not gallery_names:
        refresh_gallery_index()
    if gallery_embeddings is None or not gallery_names:
        return []

    probe_embedding = _recognition_embedding(face_tensor).squeeze(0).cpu()
    similarities = torch.matmul(gallery_embeddings, probe_embedding)
    best_similarity_by_name: dict[str, float] = {}
    best_index_by_name: dict[str, int] = {}
    for index, value in enumerate(similarities.tolist()):
        name = gallery_names[index]
        similarity = float(value)
        if name not in best_similarity_by_name or similarity > best_similarity_by_name[name]:
            best_similarity_by_name[name] = similarity
            best_index_by_name[name] = index

    ranking_score_by_name = dict(best_similarity_by_name)
    if RECOGNITION_GALLERY_SCORING in {"knn", "topk", "vote", "knn_vote"} and RECOGNITION_KNN_K > 1:
        k = min(RECOGNITION_KNN_K, int(similarities.numel()))
        values, indices = torch.topk(similarities, k=k)
        score_by_name: dict[str, float] = {}
        count_by_name: dict[str, int] = {}
        for value, index_tensor in zip(values.tolist(), indices.tolist()):
            index = int(index_tensor)
            name = gallery_names[index]
            similarity = float(value)
            score_by_name[name] = score_by_name.get(name, 0.0) + similarity
            count_by_name[name] = count_by_name.get(name, 0) + 1
        ranking_score_by_name = {
            name: score / max(1, count_by_name.get(name, 1))
            for name, score in score_by_name.items()
        }
        ranking_score_by_name.update(
            {
                name: similarity
                for name, similarity in best_similarity_by_name.items()
                if name not in ranking_score_by_name
            }
        )

        ranked_names = sorted(
            ranking_score_by_name,
            key=lambda name: (ranking_score_by_name[name], best_similarity_by_name[name]),
            reverse=True,
        )
    else:
        ranked_names = [
            name for name, _similarity in sorted(best_similarity_by_name.items(), key=lambda item: item[1], reverse=True)
        ]

    matches: list[dict[str, object]] = []
    for name in ranked_names[:top_k]:
        index = best_index_by_name[name]
        matched_image_path = Path(gallery_image_paths[index]) if index < len(gallery_image_paths) else None
        similarity = float(best_similarity_by_name[name])
        matches.append(
            {
                "name": name,
                "similarity": similarity,
                "rank_score": float(ranking_score_by_name.get(name, similarity)),
                "confidence": (similarity + 1.0) / 2.0,
                "matched_image_path": str(matched_image_path) if matched_image_path is not None else None,
            }
        )
    return matches


def _recognition_decision(matches: list[dict[str, object]]) -> tuple[bool, str | None, float]:
    if not matches:
        return False, "no-gallery-match", 0.0
    top_similarity = float(matches[0]["similarity"])
    top_score = float(matches[0].get("rank_score", top_similarity))
    second_score = float(matches[1].get("rank_score", matches[1]["similarity"])) if len(matches) > 1 else -1.0
    gap = top_score - second_score
    if top_similarity < RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD:
        return False, "low-similarity", gap
    if gap < RECOGNITION_ACCEPT_GAP_THRESHOLD:
        return False, "ambiguous-gallery-match", gap
    return True, None, gap


def _match_path(match: dict[str, object] | None) -> Path | None:
    if not match:
        return None
    value = match.get("matched_image_path")
    return Path(str(value)) if value else None


def _serialize_matches(matches: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "name": str(match["name"]),
            "similarity": round(float(match["similarity"]), 4),
            "rank_score": round(float(match.get("rank_score", match["similarity"])), 4),
            "confidence": round(float(match["confidence"]), 4),
            "matched_image_path": match.get("matched_image_path"),
        }
        for match in matches
    ]


def _predict_identity(face_tensor: torch.Tensor) -> tuple[str | None, float, Path | None]:
    matches = _rank_identity_matches(face_tensor, top_k=1)
    if not matches:
        return None, 0.0, None
    matched_path = matches[0].get("matched_image_path")
    return (
        str(matches[0]["name"]),
        float(matches[0]["similarity"]),
        Path(str(matched_path)) if matched_path else None,
    )


def _name_from_dataset_filename(filename: str | None) -> str | None:
    if not filename:
        return None

    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None

    candidate = parts[0]
    candidate_dir = gallery_dir / candidate
    if candidate_dir.is_dir():
        return candidate
    return None


def _original_image_from_match(matched_image_path: Path | None) -> np.ndarray | None:
    if matched_image_path is None or not matched_image_path.is_file():
        return None
    return _load_rgb_image_from_path(matched_image_path)


def _serve_frontend_index() -> FileResponse:
    if not frontend_index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run `npm run build` in the frontend folder.",
        )
    return FileResponse(frontend_index_path)


def _serve_frontend_dist_file(filename: str) -> FileResponse:
    path = frontend_dist_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Frontend asset {filename} not found.")
    return FileResponse(path)


@app.get("/")
def read_root():
    return {
        "message": "Occluded face recognition API is running smoothly.",
        "ui": "/index" if frontend_index_path.is_file() else None,
        "device": str(device),
        "oan_weights_loaded": oan_loaded,
        "degan_weights_loaded": degan_loaded,
        "refiner_weights_loaded": mtr_refiner_loaded,
        "gallery_size": len(gallery_names),
        **_runtime_info(),
    }


@app.get("/index", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
@app.get("/analyze", include_in_schema=False)
@app.get("/analyse", include_in_schema=False)
def read_frontend_app():
    return _serve_frontend_index()


@app.get("/favicon.svg", include_in_schema=False)
def read_frontend_favicon():
    return _serve_frontend_dist_file("favicon.svg")


@app.get("/icons.svg", include_in_schema=False)
def read_frontend_icons():
    return _serve_frontend_dist_file("icons.svg")


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...), use_refiner: bool = False):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    upload_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    analysis = _prepare_face_analysis(img_bgr)
    (
        reconstruction_tensor,
        original_rgb,
        reconstruction_mask,
        is_occluded,
        occlusion_region,
        occlusion_ratio,
        enhance_patch,
    ) = _reconstruction_view_from_analysis(analysis)

    if is_occluded:
        _, raw_reconstructed_rgb = _reconstruct_face(reconstruction_tensor, reconstruction_mask)
        _, reconstructed_rgb, refiner_info = _blend_reconstructed_face(
            reconstruction_tensor,
            reconstruction_mask,
            enhance_patch=enhance_patch,
            use_refiner=use_refiner,
            original_rgb_override=original_rgb,
            return_info=True,
        )
    else:
        raw_reconstructed_rgb = original_rgb
        reconstructed_rgb = original_rgb
        refiner_info = _new_refiner_info(
            requested=use_refiner,
            status="not-needed-no-occlusion",
            applied=False,
        )
    display_reconstructed_rgb, display_mask = _project_reconstruction_to_upload_resolution(
        upload_rgb,
        raw_reconstructed_rgb,
        reconstruction_mask,
        analysis.get("display_occlusion_mask"),
    )
    display_reconstructed_rgb, degan_oan_info = _degan_oan_refine_display(
        upload_rgb,
        display_reconstructed_rgb,
        display_mask,
    )
    display_outside_max_delta = _outside_mask_max_delta(upload_rgb, display_reconstructed_rgb, display_mask)
    mse = float(((reconstructed_rgb.astype(np.float32) - original_rgb.astype(np.float32)) ** 2).mean())
    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": is_occluded,
        "occlusion_region": occlusion_region,
        "occlusion_ratio": round(float(occlusion_ratio), 4),
        "display_occlusion_ratio": round(float(display_mask.mean()), 4),
        "occlusion_mask_source": analysis["occlusion_mask_source"],
        "occlusion_boundary_jump": round(float(analysis["occlusion_boundary_jump"]), 4),
        "display_outside_max_delta": display_outside_max_delta,
        "rectangular_occlusion_boundary_jump_threshold": RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD,
        "reconstruction_model_loaded": degan_loaded,
        "refiner_model_loaded": mtr_refiner_loaded,
        "refiner_requested": refiner_info["requested"],
        "refiner_used": refiner_info["applied"],
        "refiner_applied": refiner_info["applied"],
        "refiner_status": refiner_info["status"],
        "degan_oan_requested": degan_oan_info["requested"],
        "degan_oan_enabled": degan_oan_info["enabled"],
        "degan_oan_loaded": degan_oan_info["loaded"],
        "degan_oan_applied": degan_oan_info["applied"],
        "degan_oan_accepted": degan_oan_info["accepted"],
        "degan_oan_status": degan_oan_info["status"],
        "reconstruction_mse": round(mse, 6),
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "input_image_base64": _encode_rgb_to_base64(upload_rgb),
        "occlusion_mask_base64": _encode_mask_to_base64(display_mask),
        "raw_reconstructed_image_base64": _encode_rgb_to_base64(raw_reconstructed_rgb),
        "reconstructed_image_base64": _encode_rgb_to_base64(display_reconstructed_rgb),
        "reconstructed_model_view_base64": _encode_rgb_to_base64(reconstructed_rgb),
        **_runtime_info(),
    }
    return response


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    analysis = _prepare_face_analysis(img_bgr)
    face_for_recognition = analysis["face_tensor"]
    recognition_region_rgb = analysis["face_rgb"]

    recognition_matches = _rank_identity_matches(face_for_recognition, top_k=5)
    predicted_name = str(recognition_matches[0]["name"]) if recognition_matches else None
    similarity = float(recognition_matches[0]["similarity"]) if recognition_matches else 0.0
    recognition_accepted, recognition_rejection_reason, recognition_similarity_gap = _recognition_decision(recognition_matches)
    matched_image_path = _match_path(recognition_matches[0] if recognition_matches else None)
    gallery_original_rgb = _original_image_from_match(matched_image_path)
    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": analysis["is_occluded"],
        "occlusion_region": analysis["occlusion_region"],
        "occlusion_ratio": round(float(analysis["occlusion_ratio"]), 4),
        "occlusion_boundary_jump": round(float(analysis["occlusion_boundary_jump"]), 4),
        "rectangular_occlusion_boundary_jump_threshold": RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD,
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "match": predicted_name is not None,
        "confidence": round((similarity + 1.0) / 2.0, 4),
        "similarity": round(similarity, 4),
        "predicted_name": predicted_name,
        "recognition_accepted": recognition_accepted,
        "recognition_rejection_reason": recognition_rejection_reason,
        "recognition_similarity_gap": round(float(recognition_similarity_gap), 4),
        "recognition_similarity_threshold": RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD,
        "recognition_margin_threshold": RECOGNITION_ACCEPT_GAP_THRESHOLD,
        "recognition_top_matches": _serialize_matches(recognition_matches),
        "branch": "Hybrid",
        "reconstruction_model_loaded": degan_loaded,
        "refiner_model_loaded": mtr_refiner_loaded,
        "recognition_model_loaded": True,
        "gallery_size": len(gallery_names),
        "recognition_mode": "full-face",
        "recognition_probe_image_base64": _encode_rgb_to_base64(recognition_region_rgb),
        **_runtime_info(),
    }
    if gallery_original_rgb is not None:
        response["original_image_base64"] = _encode_rgb_to_base64(gallery_original_rgb)
    return response


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), use_refiner: bool = False):
    start = time.perf_counter()
    img_bytes = await file.read()
    img_bgr = _decode_upload(img_bytes)
    upload_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    analysis = _prepare_face_analysis(img_bgr)
    recognition_probe_tensor = analysis["face_tensor"]
    recognition_region_rgb = analysis["face_rgb"]
    recognition_probe_source = "input"
    (
        reconstruction_tensor,
        reconstruction_input_rgb,
        reconstruction_mask,
        is_occluded,
        occlusion_region,
        occlusion_ratio,
        enhance_patch,
    ) = _reconstruction_view_from_analysis(analysis)

    if is_occluded:
        _, raw_reconstructed_rgb = _reconstruct_face(reconstruction_tensor, reconstruction_mask)
        _, reconstructed_rgb, refiner_info = _blend_reconstructed_face(
            reconstruction_tensor,
            reconstruction_mask,
            enhance_patch=enhance_patch,
            use_refiner=use_refiner,
            original_rgb_override=reconstruction_input_rgb,
            return_info=True,
        )
    else:
        raw_reconstructed_rgb = reconstruction_input_rgb
        reconstructed_rgb = reconstruction_input_rgb
        refiner_info = _new_refiner_info(
            requested=use_refiner,
            status="not-needed-no-occlusion",
            applied=False,
        )

    recognition_matches = _rank_identity_matches(recognition_probe_tensor, top_k=5)
    predicted_name = str(recognition_matches[0]["name"]) if recognition_matches else None
    similarity = float(recognition_matches[0]["similarity"]) if recognition_matches else 0.0
    recognition_accepted, recognition_rejection_reason, recognition_similarity_gap = _recognition_decision(recognition_matches)
    matched_image_path = _match_path(recognition_matches[0] if recognition_matches else None)
    gallery_original_rgb = _original_image_from_match(matched_image_path)

    reconstructed_probe_aligned = False
    reconstructed_predicted_name = None
    reconstructed_similarity = 0.0
    reconstructed_matches: list[dict[str, object]] = []
    reconstructed_matched_image_path = None
    if is_occluded:
        reconstructed_probe_tensor, _, reconstructed_probe_aligned = _recognition_probe_from_reconstruction(
            reconstructed_rgb,
            analysis["face_tensor"],
            analysis["face_rgb"],
        )
        reconstructed_matches = _rank_identity_matches(reconstructed_probe_tensor, top_k=5)
        reconstructed_predicted_name = str(reconstructed_matches[0]["name"]) if reconstructed_matches else None
        reconstructed_similarity = float(reconstructed_matches[0]["similarity"]) if reconstructed_matches else 0.0
        reconstructed_matched_image_path = _match_path(reconstructed_matches[0] if reconstructed_matches else None)
    else:
        reconstructed_probe_aligned = bool(analysis["detected_face"])
        reconstructed_matches = recognition_matches
        reconstructed_predicted_name = predicted_name
        reconstructed_similarity = similarity
        reconstructed_matched_image_path = matched_image_path
    reconstructed_gallery_original_rgb = _original_image_from_match(reconstructed_matched_image_path)
    (
        reconstructed_recognition_accepted,
        reconstructed_recognition_rejection_reason,
        reconstructed_similarity_gap,
    ) = _recognition_decision(reconstructed_matches)

    system_predicted_name = predicted_name
    system_similarity = similarity
    system_probe_source = "input"
    if (
        is_occluded
        and reconstructed_predicted_name is not None
        and reconstructed_similarity > similarity + HYBRID_RECOGNITION_MARGIN
    ):
        system_predicted_name = reconstructed_predicted_name
        system_similarity = reconstructed_similarity
        system_probe_source = "reconstructed"

    display_reconstructed_rgb, display_mask = _project_reconstruction_to_upload_resolution(
        upload_rgb,
        raw_reconstructed_rgb,
        reconstruction_mask,
        analysis.get("display_occlusion_mask"),
    )
    display_reconstructed_rgb, degan_oan_info = _degan_oan_refine_display(
        upload_rgb,
        display_reconstructed_rgb,
        display_mask,
    )
    display_outside_max_delta = _outside_mask_max_delta(upload_rgb, display_reconstructed_rgb, display_mask)

    elapsed = time.perf_counter() - start

    response = {
        "status": "success",
        "detected_face": analysis["detected_face"],
        "is_occluded": is_occluded,
        "occlusion_region": occlusion_region,
        "occlusion_ratio": round(float(occlusion_ratio), 4),
        "display_occlusion_ratio": round(float(display_mask.mean()), 4),
        "occlusion_mask_source": analysis["occlusion_mask_source"],
        "occlusion_boundary_jump": round(float(analysis["occlusion_boundary_jump"]), 4),
        "display_outside_max_delta": display_outside_max_delta,
        "rectangular_occlusion_boundary_jump_threshold": RECTANGULAR_OCCLUSION_BOUNDARY_JUMP_THRESHOLD,
        "predicted_name": predicted_name,
        "similarity": round(similarity, 4),
        "confidence": round((similarity + 1.0) / 2.0, 4),
        "fps": round(1.0 / max(elapsed, 1e-6), 2),
        "recognition_mode": "input-face" if is_occluded else "full-face",
        "recognition_probe_source": recognition_probe_source,
        "reconstructed_probe_aligned": reconstructed_probe_aligned,
        "recognition_accepted": recognition_accepted,
        "recognition_rejection_reason": recognition_rejection_reason,
        "recognition_similarity_gap": round(float(recognition_similarity_gap), 4),
        "recognition_similarity_threshold": RECOGNITION_ACCEPT_SIMILARITY_THRESHOLD,
        "recognition_margin_threshold": RECOGNITION_ACCEPT_GAP_THRESHOLD,
        "recognition_top_matches": _serialize_matches(recognition_matches),
        "reconstructed_predicted_name": reconstructed_predicted_name,
        "reconstructed_similarity": round(reconstructed_similarity, 4),
        "reconstructed_recognition_accepted": reconstructed_recognition_accepted,
        "reconstructed_recognition_rejection_reason": reconstructed_recognition_rejection_reason,
        "reconstructed_recognition_similarity_gap": round(float(reconstructed_similarity_gap), 4),
        "reconstructed_top_matches": _serialize_matches(reconstructed_matches),
        "system_predicted_name": system_predicted_name,
        "system_similarity": round(system_similarity, 4),
        "system_probe_source": system_probe_source,
        "hybrid_recognition_margin": HYBRID_RECOGNITION_MARGIN,
        "reconstruction_model_loaded": degan_loaded,
        "refiner_model_loaded": mtr_refiner_loaded,
        "refiner_requested": refiner_info["requested"],
        "refiner_used": refiner_info["applied"],
        "refiner_applied": refiner_info["applied"],
        "refiner_status": refiner_info["status"],
        "degan_oan_requested": degan_oan_info["requested"],
        "degan_oan_enabled": degan_oan_info["enabled"],
        "degan_oan_loaded": degan_oan_info["loaded"],
        "degan_oan_applied": degan_oan_info["applied"],
        "degan_oan_accepted": degan_oan_info["accepted"],
        "degan_oan_status": degan_oan_info["status"],
        "input_image_base64": _encode_rgb_to_base64(upload_rgb),
        "occlusion_mask_base64": _encode_mask_to_base64(display_mask),
        "raw_reconstructed_image_base64": _encode_rgb_to_base64(raw_reconstructed_rgb),
        "reconstructed_image_base64": _encode_rgb_to_base64(display_reconstructed_rgb),
        "reconstructed_model_view_base64": _encode_rgb_to_base64(reconstructed_rgb),
        "recognition_probe_image_base64": _encode_rgb_to_base64(recognition_region_rgb),
        **_runtime_info(),
    }
    if gallery_original_rgb is not None:
        response["original_image_base64"] = _encode_rgb_to_base64(gallery_original_rgb)
    if reconstructed_gallery_original_rgb is not None:
        response["reconstructed_original_image_base64"] = _encode_rgb_to_base64(reconstructed_gallery_original_rgb)
    return response


@app.post("/analyse")
async def analyse(file: UploadFile = File(...), use_refiner: bool = False):
    return await analyze(file=file, use_refiner=use_refiner)


@app.post("/refresh-gallery")
def refresh_gallery():
    gallery_size = refresh_gallery_index()
    return {"status": "success", "gallery_size": gallery_size}
