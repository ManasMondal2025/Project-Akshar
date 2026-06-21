"""
CNN-based Corner Detection Utility
=====================================
Ports the OptimizedMobileNetwork inference pipeline from the akshar_ai project.

Algorithm, preprocessing, inference, and postprocessing logic are preserved
exactly as in the reference implementation (akshar_ai-main 2/bbox.py +
akshar_ai-main 2/bbox_cnn.py).

Public API:
  - detect_corners_cnn(image_path) -> [[x,y], [x,y], [x,y], [x,y]]  (TL, TR, BR, BL)
    Returns corner coordinates in the same format as detect_corners() in
    edge_detect_corners.py so the rest of the system needs no adaptation.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model weights path — inside pretrained_models/ relative to this file
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(
    _THIS_DIR, "..", "pretrained_models", "best_model_fold_5.pth"
)

# Target canvas dimension (same as reference implementation)
_TARGET_DIM = 2048

# ---------------------------------------------------------------------------
# Model Architecture (ported verbatim from bbox_cnn.py)
# ---------------------------------------------------------------------------

class _SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(_SqueezeExcitation, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        return x * self.se(x)


class _Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, use_se, activation_layer):
        super(_Bottleneck, self).__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        padding = kernel_size // 2
        layers = []

        # Pointwise Expansion
        if exp_size != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(exp_size),
                activation_layer(inplace=True)
            ])

        # Depthwise Convolution
        layers.extend([
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            activation_layer(inplace=True)
        ])

        # Squeeze and Excitation
        if use_se:
            layers.append(_SqueezeExcitation(exp_size))

        # Pointwise Linear Projection
        layers.extend([
            nn.Conv2d(exp_size, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class _OptimizedMobileNetwork(nn.Module):
    def __init__(self, in_channels=1, num_targets=8):
        super(_OptimizedMobileNetwork, self).__init__()

        # Stage 1: Standard Convolutional Downsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stage 2: 3x3 Bottleneck Blocks
        self.stage2 = nn.Sequential(
            _Bottleneck(in_channels=128, out_channels=256, exp_size=256,
                        kernel_size=3, stride=2, use_se=False, activation_layer=nn.ReLU),
            _Bottleneck(in_channels=256, out_channels=256, exp_size=512,
                        kernel_size=3, stride=1, use_se=False, activation_layer=nn.ReLU)
        )

        # Stage 3: 5x5 Bottleneck Blocks (with SE)
        self.stage3 = nn.Sequential(
            _Bottleneck(in_channels=256, out_channels=512, exp_size=512,
                        kernel_size=5, stride=2, use_se=True, activation_layer=nn.Hardswish),
            _Bottleneck(in_channels=512, out_channels=512, exp_size=1024,
                        kernel_size=5, stride=1, use_se=True, activation_layer=nn.Hardswish),
            _Bottleneck(in_channels=512, out_channels=1024, exp_size=1024,
                        kernel_size=5, stride=2, use_se=True, activation_layer=nn.Hardswish),
            _Bottleneck(in_channels=1024, out_channels=1024, exp_size=2048,
                        kernel_size=5, stride=2, use_se=True, activation_layer=nn.Hardswish),
            _Bottleneck(in_channels=1024, out_channels=1024, exp_size=2048,
                        kernel_size=5, stride=2, use_se=True, activation_layer=nn.Hardswish)
        )

        # Stage 4: Optimized Final Layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Hardswish(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final Linear mapping to 8 targets (4 corners × 2 coords)
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_targets),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = self.regression_head(x)
        return x


# ---------------------------------------------------------------------------
# Lazy model singleton (avoid reloading weights on every request)
# ---------------------------------------------------------------------------

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL_INSTANCE = None


def _get_model():
    """Lazy initializer — loads model weights once and caches the instance."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        weights_path = os.path.normpath(_WEIGHTS_PATH)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"CNN corner detection model weights not found at: '{weights_path}'. "
                f"Ensure best_model_fold_5.pth is present in pretrained_models/."
            )
        model = _OptimizedMobileNetwork(in_channels=1, num_targets=8).to(_DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=_DEVICE))
        model.eval()
        _MODEL_INSTANCE = model
    return _MODEL_INSTANCE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_corners_cnn(image_path: str) -> list[list[int]]:
    """
    Run CNN inference to detect the four document corners in an image.

    Preprocessing, inference, and postprocessing logic is preserved exactly
    as in the akshar_ai reference implementation (bbox.py).

    Parameters
    ----------
    image_path : str
        Absolute path to the source image file.

    Returns
    -------
    list[list[int]]
        Four corner coordinates [[x,y], ...] in TL, TR, BR, BL order,
        mapped back to the original image pixel space.
        Falls back to full-image boundary on inference failure.
    """
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h_orig, w_orig = raw_image.shape[:2]
    max_dim = max(h_orig, w_orig)

    try:
        # 1. Preprocessing: Scale down if max(H, W) > 2048, maintaining aspect ratio
        image = raw_image.copy()
        scale = 1.0
        if max_dim > _TARGET_DIM:
            scale = _TARGET_DIM / max_dim
            new_w = int(round(w_orig * scale))
            new_h = int(round(h_orig * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h_scaled, w_scaled = image.shape[:2]

        # Pad bottom and right edges to construct a uniform 2048×2048 canvas
        pad_bottom = _TARGET_DIM - h_scaled
        pad_right = _TARGET_DIM - w_scaled
        padded_image = cv2.copyMakeBorder(
            image, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Convert to single-channel grayscale
        gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)

        # 2. Neural Network Forward Pass Preparation
        image_tensor = gray_image.astype('float32') / 255.0
        image_tensor = np.expand_dims(image_tensor, axis=(0, 1))  # (1, 1, 2048, 2048)

        device_tensor = torch.from_numpy(image_tensor).to(_DEVICE)
        model = _get_model()

        with torch.no_grad():
            output = model(device_tensor)

        # Extract flat output: [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
        predictions = output.squeeze().cpu().numpy()

        # 3. Post-processing: Map coordinates back to original image space
        # Step A: Scale sigmoid [0,1] outputs up to 2048px canvas dimensions
        canvas_coords = predictions * float(_TARGET_DIM)

        # Step B: Reshape to 4 distinct (x, y) points
        points = canvas_coords.reshape(4, 2)

        # Step C: Invert the aspect-ratio scaling to restore original pixel coords
        original_scale_points = points / scale

        # 4. Clip to image bounds and convert to [[x,y], ...] list format
        corners = []
        for pt in original_scale_points:
            x = int(max(0, min(w_orig - 1, round(pt[0]))))
            y = int(max(0, min(h_orig - 1, round(pt[1]))))
            corners.append([x, y])

        return corners  # [TL, TR, BR, BL]

    except Exception as e:
        # Fallback: return full image boundary if inference fails
        print(f"[CNN Corner Detection] Inference failed ({e}). Returning image boundary.")
        ix, iy = int(w_orig * 0.08), int(h_orig * 0.08)
        return [
            [ix, iy],
            [w_orig - ix, iy],
            [w_orig - ix, h_orig - iy],
            [ix, h_orig - iy],
        ]
