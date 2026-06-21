"""
predictor.py — ML-based document dewarping inference wrapper.

Wraps DewarpTextlineMaskGuide with:
  - Lazy singleton model loading (loaded once, reused across requests)
  - Automatic device selection: CUDA → MPS → CPU
  - Simple run_auto_dewarp(bgr_image) → bgr_image API

Model path resolution order:
  1. DEWARP_MODEL_PATH environment variable
  2. Default: test dewarp pretrained_models/30.pt in the project
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Default model path — points to the pretrained model in backend/pretrained_models
# Adjust via DEWARP_MODEL_PATH env var if needed.
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = os.environ.get(
    "DEWARP_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__),            # …/utils/dewarp_ml/
        "..", "..", "pretrained_models", "30.pt",
    )
)

_MODEL_INPUT_SIZE = 224   # fixed input size the model was trained on

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_model = None
_device = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model() -> None:
    """Load DewarpTextlineMaskGuide once; subsequent calls are no-ops."""
    global _model, _device

    if _model is not None:
        return  # already loaded

    model_path = os.path.normpath(_DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[AutoDewarp] Pretrained model not found at: {model_path}\n"
            "Set DEWARP_MODEL_PATH env variable to the correct path of 30.pt"
        )

    # Import here to avoid loading torch at module import time
    from utils.dewarp_ml.model import DewarpTextlineMaskGuide

    _device = _get_device()
    print(f"[AutoDewarp] Loading model on {_device} …")

    model = DewarpTextlineMaskGuide(image_size=_MODEL_INPUT_SIZE)

    state_dict = torch.load(model_path, map_location="cpu")

    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(_device)
    model.eval()

    _model = model
    print(f"[AutoDewarp] Model loaded — ready for inference.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_auto_dewarp(bgr_image: np.ndarray) -> np.ndarray:
    """
    Apply ML-based dewarping to a BGR OpenCV image.

    Parameters
    ----------
    bgr_image : np.ndarray  (H, W, 3)  uint8  BGR

    Returns
    -------
    dewarped  : np.ndarray  (H, W, 3)  uint8  BGR
    """
    _load_model()

    img_h, img_w = bgr_image.shape[:2]

    # BGR → RGB float32 [0, 1]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img_f32 = (rgb.astype(np.float32) / 255.0)

    # Resize to model input size
    resized = cv2.resize(img_f32, (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE))

    # HWC → CHW tensor, add batch dim
    input_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor = input_tensor.to(_device)

    with torch.no_grad():
        # bm shape: (1, 2, H, W) — backward mapping in pixel coords [0, 223]
        bm = _model(input_tensor.float())

        # Normalise to [-1, 1] for grid_sample
        bm = (2 * (bm / (_MODEL_INPUT_SIZE - 1.0)) - 1) * 0.99

    bm = bm.detach().cpu()

    # Resize BM back to original image size
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))   # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))   # y flow

    # Smooth slightly to reduce block artefacts
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))

    # Build sampling grid  (1, H, W, 2)
    lbl = torch.from_numpy(
        np.stack([bm0, bm1], axis=2)
    ).unsqueeze(0).float()   # (1, H, W, 2)

    # Source tensor: original BGR as float, CHW
    src_rgb = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(src_rgb, lbl, align_corners=True)

    # Convert back to uint8 BGR
    out_np = (out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    return result_bgr


def is_model_available() -> bool:
    """Return True if the pretrained model file exists."""
    path = os.path.normpath(_DEFAULT_MODEL_PATH)
    return os.path.exists(path)
