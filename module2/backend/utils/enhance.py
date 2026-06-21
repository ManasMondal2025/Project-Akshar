"""
Module 1 - Interactive Image Workbench: Image Enhancement Utilities
Provides Otsu thresholding, adaptive Gaussian threshold, and CLAHE
contrast enhancement for document image preprocessing.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class EnhancementConfig:
    """Controls for the digital color document look."""

    illumination_strength: float = 0.72
    paper_whiten_strength: float = 0.62
    contrast_gain: float = 1.10
    saturation_gain: float = 1.22
    sharpen_amount: float = 0.32
    sharpen_radius: float = 1.10
    denoise_mix: float = 0.20
    denoise_median_size: int = 3
    background_blur_radius: Optional[float] = None
    jpeg_quality: int = 95


DEFAULT_CONFIG = EnhancementConfig()


def _clip_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high must be greater than low")
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _odd_filter_size(size: int) -> int:
    size = max(3, int(size))
    return size if size % 2 else size + 1


def _background_radius(height: int, width: int, config: EnhancementConfig) -> float:
    if config.background_blur_radius is not None:
        return max(1.0, float(config.background_blur_radius))
    return max(18.0, min(height, width) * 0.025)


def _estimate_background_luma(luma_u8: np.ndarray, radius: float) -> np.ndarray:
    """Estimate broad lighting without learning characters or photo details."""
    height, width = luma_u8.shape
    luma_image = Image.fromarray(luma_u8, mode="L")
    scale = min(1.0, 900.0 / max(height, width))

    if scale < 1.0:
        preview_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        preview = luma_image.resize(preview_size, Image.Resampling.BILINEAR)
        background = preview.filter(
            ImageFilter.GaussianBlur(radius=max(1.0, radius * scale))
        )
        background = background.resize((width, height), Image.Resampling.BILINEAR)
    else:
        background = luma_image.filter(ImageFilter.GaussianBlur(radius=radius))

    return np.asarray(background, dtype=np.float32)


def _light_denoise(rgb_image: Image.Image, config: EnhancementConfig) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32)
    if config.denoise_mix <= 0:
        return rgb

    median = rgb_image.filter(
        ImageFilter.MedianFilter(size=_odd_filter_size(config.denoise_median_size))
    )
    median_rgb = np.asarray(median, dtype=np.float32)
    mix = float(np.clip(config.denoise_mix, 0.0, 1.0))
    return rgb * (1.0 - mix) + median_rgb * mix


def enhance_document_color(
    image: Image.Image,
    config: EnhancementConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """Enhance a cropped document page while preserving colored content.

    The paper and text contrast are improved through luminance processing.
    Original chroma is kept for photos, diagrams, highlights, and colored ink.
    """
    image = ImageOps.exif_transpose(image)
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    rgb_image = image.convert("RGB")
    base_rgb = _clip_u8(_light_denoise(rgb_image, config))

    ycbcr = Image.fromarray(base_rgb, mode="RGB").convert("YCbCr")
    ycbcr_values = np.asarray(ycbcr, dtype=np.float32)
    luma = ycbcr_values[..., 0]
    cb = ycbcr_values[..., 1]
    cr = ycbcr_values[..., 2]

    height, width = luma.shape
    background = _estimate_background_luma(
        _clip_u8(luma),
        radius=_background_radius(height, width, config),
    )
    background = np.maximum(background, 8.0)

    # Keep lighting correction smooth so real strokes are not expanded.
    target_background = 238.0
    normalized_luma = luma * (target_background / background)
    illumination_mix = float(np.clip(config.illumination_strength, 0.0, 1.0))
    scan_luma = luma * (1.0 - illumination_mix) + normalized_luma * illumination_mix
    scan_luma = np.clip(scan_luma, 0.0, 255.0)

    contrast_gain = max(0.0, float(config.contrast_gain))
    scan_luma = np.clip(128.0 + (scan_luma - 128.0) * contrast_gain, 0.0, 255.0)

    cb_offset = cb - 128.0
    cr_offset = cr - 128.0
    chroma = np.sqrt(cb_offset * cb_offset + cr_offset * cr_offset)
    colored_content = _smoothstep(chroma, 8.0, 34.0)
    neutral_content = 1.0 - colored_content

    # Whiten only bright, low-chroma regions. Faint gray show-through fades,
    # while dark text and colored photos keep their tonal range.
    paper_brightness = _smoothstep(scan_luma, 142.0, 228.0)
    paper_mask = neutral_content * paper_brightness
    paper_mix = paper_mask * float(np.clip(config.paper_whiten_strength, 0.0, 1.0))
    scan_luma = scan_luma + (255.0 - scan_luma) * paper_mix

    if config.sharpen_amount > 0:
        blurred = Image.fromarray(_clip_u8(scan_luma), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(0.1, config.sharpen_radius))
        )
        blurred_luma = np.asarray(blurred, dtype=np.float32)
        detail = np.clip(scan_luma - blurred_luma, -28.0, 28.0)
        scan_luma = np.clip(scan_luma + detail * config.sharpen_amount, 0.0, 255.0)

    # Keep a clean white paper background but strengthen genuine color.
    saturation_gain = max(0.0, float(config.saturation_gain))
    saturation_scale = 1.0 + (saturation_gain - 1.0) * colored_content
    paper_neutral_scale = 1.0 - 0.86 * paper_mix
    chroma_scale = saturation_scale * paper_neutral_scale

    enhanced_ycbcr = np.empty_like(ycbcr_values, dtype=np.uint8)
    enhanced_ycbcr[..., 0] = _clip_u8(scan_luma)
    enhanced_ycbcr[..., 1] = _clip_u8(128.0 + cb_offset * chroma_scale)
    enhanced_ycbcr[..., 2] = _clip_u8(128.0 + cr_offset * chroma_scale)

    enhanced = Image.fromarray(enhanced_ycbcr, mode="YCbCr").convert("RGB")
    if alpha is not None:
        enhanced.putalpha(alpha)
    return enhanced


def enhance_rgb_array(
    rgb: np.ndarray,
    config: EnhancementConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Enhance a uint8 RGB NumPy image and return a uint8 RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        rgb = _clip_u8(rgb.astype(np.float32))
    enhanced = enhance_document_color(Image.fromarray(rgb, mode="RGB"), config)
    return np.asarray(enhanced, dtype=np.uint8)


def enhance_bgr_array(
    bgr: np.ndarray,
    config: EnhancementConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Enhance an OpenCV-style BGR NumPy image without requiring OpenCV."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must have shape (height, width, 3)")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    enhanced_rgb = enhance_rgb_array(rgb, config)
    return np.ascontiguousarray(enhanced_rgb[..., ::-1])


def _remove_shadow(image: np.ndarray) -> np.ndarray:
    """
    Remove shadow / uneven illumination from a scanned document.
    Scales parameters proportionally with resolution so that it works
    optimally on high-resolution images as well as standard images.
    """
    h, w = image.shape[:2]
    scale = w / 800.0

    k_dil = int(max(3, round(7 * scale)))
    k_blur = int(max(5, round(21 * scale)))
    if k_blur % 2 == 0:
        k_blur += 1

    rgb_planes  = cv2.split(image)
    result_norm = []
    dil_kernel = np.ones((k_dil, k_dil), np.uint8)
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, dil_kernel)
        bg      = cv2.medianBlur(dilated, k_blur)
        diff    = 255 - cv2.absdiff(plane, bg)
        norm    = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_norm.append(norm)
    return cv2.merge(result_norm)


def _remove_black_border(image: np.ndarray) -> np.ndarray:
    """
    Match Module 1's post-extraction cleanup: crop away residual dark edges by
    finding the bounding box of the bright page region.

    Pages arrive already perspective-cropped and do not need SAM whitening.
    pages. If lighting is gray or uneven, a simple bright-pixel threshold can
    find only a small highlight and accidentally crop away most of the page.
    Keep this conservative: only crop when the bright region still covers almost
    the whole image, which is the signature of a thin border.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image

    x, y, box_w, box_h = cv2.boundingRect(coords)
    width_ratio = box_w / float(w)
    height_ratio = box_h / float(h)
    area_ratio = (box_w * box_h) / float(w * h)

    if width_ratio < 0.85 or height_ratio < 0.85 or area_ratio < 0.80:
        return image

    pad = max(4, int(min(w, h) * 0.01))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + box_w + pad)
    y2 = min(h, y + box_h + pad)

    if (x2 - x1) < w * 0.85 or (y2 - y1) < h * 0.85:
        return image

    return image[y1:y2, x1:x2]


def _resize_for_pipeline(image: np.ndarray, width: int = 800) -> np.ndarray:
    """Match Module 1's fixed-width working resolution."""
    h, w = image.shape[:2]
    if w == width:
        return image

    ratio = width / float(w)
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, int(h * ratio)), interpolation=interpolation)


def _remove_shadow_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Original Module 1 shadow removal (shadow.py) with hardcoded kernel sizes.
    Works per-channel: dilate → medianBlur → diff → normalize.
    """
    rgb_planes = cv2.split(image)
    result_norm = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_norm.append(norm)
    return cv2.merge(result_norm)


def _enhance_bw_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Original Module 1 B/W enhancement pipeline (enhance.py, mode='bw'):
      1. Shadow removal with hardcoded kernel sizes
      2. Grayscale conversion
      3. Background normalization with medianBlur(gray, 31)
      4. Sharpen with laplacian kernel
      5. Otsu threshold
      6. Morphological open with (2,2) kernel to clean noise
    Returns a 3-channel BGR image for consistency.
    """
    image = _remove_shadow_pipeline(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------- Step 1: Background normalization ----------
    bg = cv2.medianBlur(gray, 31)   # estimate background (original hardcoded size)
    norm = cv2.divide(gray, bg, scale=255)

    # --------- Step 2: Sharpen (VERY IMPORTANT) ----------
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(norm, -1, kernel)

    # --------- Step 3: Strong but stable threshold ----------
    _, thresh = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------- Step 4: Clean tiny noise ----------
    kernel_clean = np.ones((2, 2), np.uint8)  # original hardcoded size
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def apply_otsu_threshold(image: np.ndarray, mode: str = "color") -> np.ndarray:
    """
    Apply the visual enhancement used by the workbench Otsu/BW/Color controls.

    - "color" mode uses the sophisticated color-preserving document enhancement
      pipeline from enhance_digital.py, avoiding binary thresholding.
    - "bw" mode uses the original Module 1 black-and-white Otsu pipeline
      (shadow removal → background normalization → sharpen → Otsu → clean).

    Args:
        image: Input image (BGR or grayscale)
        mode:  "color" → sharpened, background-normalized color output [default]
               "bw"    → true Otsu binary output using Module 1 pipeline

    Returns:
        Processed image as 3-channel BGR for consistency
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if mode == "color":
        return enhance_bgr_array(image)

    # "bw" mode: use the original Module 1 enhance pipeline
    return _enhance_bw_pipeline(image)





def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding for better handling of
    varying lighting conditions across the document.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Adaptively thresholded image as 3-channel BGR
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply slight Gaussian blur to reduce noise
    blurred =cv2.medianBlur(gray, 5)
    
    #CLAHE (fix uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Scale-invariant block size for adaptive threshold
    # 15px is good for 800px width. Scale proportionally for larger images.
    h, w = gray.shape[:2]
    block_size = int(max(h, w) * (15.0 / 800.0))
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(15, block_size)  # At least 15
    
    # Apply adaptive Gaussian threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=5
    )
    
    # Convert back to 3-channel for consistent output
    result = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    return result


# def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
#     """
#     Advanced adaptive thresholding for noisy document images.
#     Produces clean text with minimal background noise.
#     """

#     # Convert to grayscale
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()

#     # 1. Edge-preserving denoising (better than Gaussian)
#     #denoised = cv2.bilateralFilter(gray, 9, 75, 75)
#     denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

#     # 2. Background normalization (VERY IMPORTANT STEP)
#     bg = cv2.medianBlur(denoised, 21)
#     normalized = cv2.divide(denoised, bg, scale=255)

#     # sharpen text
#     kernel_sharp = np.array([[0, -1, 0],
#                              [-1, 5, -1],
#                              [0, -1, 0]])
#     sharpened = cv2.filter2D(normalized, -1, kernel_sharp)

#     # 3. Contrast enhancement (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(sharpened)

#     # 4. Adaptive Gaussian Threshold (tuned parameters)
#     adaptive = cv2.adaptiveThreshold(
#         enhanced,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         15,   # larger block size = smoother regions
#         5     # removes background noise
#     )

#     # 5. Morphological cleaning (remove small dots)
#     kernel = np.ones((2, 2), np.uint8)
#     clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

#     # Optional: slight smoothing to remove tiny artifacts
#     #clean = cv2.medianBlur(clean, 3)
#     clean = cv2.dilate(clean, np.ones((1, 1), np.uint8), iterations=1)

#     # Convert back to 3-channel image
#     result = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

#     return result





def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to enhance contrast while preventing over-amplification of noise.
    
    Works on the L channel of LAB color space for color images,
    preserving color information while enhancing contrast.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Contrast-enhanced image as 3-channel BGR
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to the L (lightness) channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Merge channels back and convert to BGR
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # For grayscale images, apply CLAHE directly
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return result
