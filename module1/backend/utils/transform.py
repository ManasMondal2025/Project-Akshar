"""
Module 1 - Interactive Image Workbench: Transform Utilities
Handles perspective transformation (homography correction) using OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in the format: [top-left, top-right, bottom-right, bottom-left].
    This ensures consistent ordering regardless of how the user places corners.
    
    Args:
        pts: Array of 4 (x, y) coordinates
        
    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has the smallest sum, bottom-right has the largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has the smallest difference, bottom-left has the largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def compute_output_dimensions(rect: np.ndarray) -> Tuple[int, int]:
    """
    Compute the width and height of the output image based on the
    maximum distances between the ordered corner points.
    
    Args:
        rect: Ordered array of 4 corner points
        
    Returns:
        Tuple of (width, height) for the output image
    """
    (tl, tr, br, bl) = rect
    
    # Width: max of top edge and bottom edge
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))
    
    # Height: max of left edge and right edge
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))
    
    return max_width, max_height


def apply_perspective_transform(
    image: np.ndarray,
    corners: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Apply perspective transformation to extract and rectify a region
    defined by 4 corner points.
    
    Args:
        image: Input image as NumPy array (BGR)
        corners: List of 4 (x, y) tuples representing document corners
        
    Returns:
        Warped (rectified) image as NumPy array
        
    Raises:
        ValueError: If corners don't contain exactly 4 points
    """
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corner points, got {len(corners)}")
    
    # Convert to numpy array and order the points
    pts = np.array(corners, dtype="float32")
    rect = order_points(pts)
    
    # Compute output dimensions
    max_width, max_height = compute_output_dimensions(rect)
    
    # Ensure minimum dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)
    
    # Define destination points (a perfect rectangle)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped
