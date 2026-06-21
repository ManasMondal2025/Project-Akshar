"""
Module 1 - Automatic document deskew utility.

This ports the automatic deskew logic from the root-level deskew_img.py
reference into Module 1's backend utility layer. The reference file is not
imported directly; the frontend continues to call the existing /deskew and
/deskew/manual endpoints.
"""

import cv2
import numpy as np
from typing import Optional


MIN_ANGLE_TO_ROTATE = 0.15
MAX_DESKEW_ANGLE = 45.0
PROJECTION_COARSE_STEP = 0.5
PROJECTION_FINE_STEP = 0.1
PROJECTION_CONFIDENCE = 2.0


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    point_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]

    point_diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]

    return ordered


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 50:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
    )


def find_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    max_dimension = 1600
    scale = 1.0

    if max(height, width) > max_dimension:
        scale = max_dimension / float(max(height, width))
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image.copy()

    small_height, small_width = resized.shape[:2]
    image_area = small_height * small_width

    gray = _to_gray(resized)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    best_fallback = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon in (0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2).astype("float32")
                return corners / scale

        if best_fallback is None:
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle).astype("float32")
            box_area = cv2.contourArea(box)

            if box_area > image_area * 0.18:
                best_fallback = box / scale

    return best_fallback


def auto_perspective_correct(image: np.ndarray) -> np.ndarray:
    corners = find_document_corners(image)
    if corners is None:
        return image
    return four_point_transform(image, corners)


def weighted_median(values, weights) -> float:
    values = np.asarray(values)
    weights = np.asarray(weights)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    midpoint = weights.sum() / 2.0
    cumulative_weight = np.cumsum(weights)

    return float(values[np.searchsorted(cumulative_weight, midpoint)])


def build_text_mask(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)

    if max(gray.shape[:2]) > 1800:
        scale = 1800 / float(max(gray.shape[:2]))
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)


def normalize_hough_angle(angle: float) -> float:
    while angle <= -90:
        angle += 180

    while angle > 90:
        angle -= 180

    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    return angle


def detect_hough_skew_angle(thresh: np.ndarray) -> Optional[float]:
    height, width = thresh.shape[:2]

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, width // 30), max(1, height // 300)),
    )

    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    edges = cv2.Canny(connected, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(40, width // 25),
        minLineLength=max(50, width // 8),
        maxLineGap=max(10, width // 60),
    )

    if lines is None:
        return None

    angles = []
    weights = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length < max(50, width // 10):
            continue

        angle = normalize_hough_angle(float(np.degrees(np.arctan2(dy, dx))))

        if abs(angle) <= MAX_DESKEW_ANGLE:
            angles.append(angle)
            weights.append(length)

    if not angles:
        return None

    return weighted_median(angles, weights)


def detect_min_area_skew_angle(thresh: np.ndarray) -> Optional[float]:
    coordinates = np.column_stack(np.where(thresh > 0))

    if len(coordinates) < 100:
        return None

    raw_angle = cv2.minAreaRect(coordinates)[-1]

    if raw_angle < -45:
        angle = -(90 + raw_angle)
    else:
        angle = -raw_angle

    if abs(angle) > MAX_DESKEW_ANGLE:
        return None

    return float(angle)


def rotate_binary_same_size(thresh: np.ndarray, angle: float) -> np.ndarray:
    height, width = thresh.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        thresh,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def projection_score(thresh: np.ndarray) -> float:
    row_sums = np.sum(thresh > 0, axis=1).astype(np.float32)
    total_foreground = float(np.sum(row_sums))

    if total_foreground < 100:
        return 0.0

    differences = np.diff(row_sums)
    return float(np.sum(differences * differences) / (total_foreground + 1.0))


def detect_projection_skew_angle(thresh: np.ndarray) -> Optional[float]:
    height, width = thresh.shape[:2]
    foreground_ratio = np.count_nonzero(thresh) / float(height * width)

    if foreground_ratio < 0.0005 or foreground_ratio > 0.65:
        return None

    max_dimension = 900

    if max(height, width) > max_dimension:
        scale = max_dimension / float(max(height, width))
        work = cv2.resize(
            thresh,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        work = thresh

    coarse_angles = np.arange(
        -MAX_DESKEW_ANGLE,
        MAX_DESKEW_ANGLE + PROJECTION_COARSE_STEP,
        PROJECTION_COARSE_STEP,
    )

    coarse_scores = np.array(
        [projection_score(rotate_binary_same_size(work, angle)) for angle in coarse_angles],
        dtype=np.float32,
    )

    best_index = int(np.argmax(coarse_scores))
    best_angle = float(coarse_angles[best_index])
    best_score = float(coarse_scores[best_index])
    median_score = float(np.median(coarse_scores))

    if best_score <= 0:
        return None

    confidence = (best_score - median_score) / (median_score + 1e-6)

    if confidence < PROJECTION_CONFIDENCE and abs(best_angle) > 1.0:
        return None

    fine_start = max(-MAX_DESKEW_ANGLE, best_angle - PROJECTION_COARSE_STEP)
    fine_stop = min(MAX_DESKEW_ANGLE, best_angle + PROJECTION_COARSE_STEP)

    fine_angles = np.arange(
        fine_start,
        fine_stop + PROJECTION_FINE_STEP,
        PROJECTION_FINE_STEP,
    )

    best_fine_angle = best_angle
    best_fine_score = best_score

    for angle in fine_angles:
        score = projection_score(rotate_binary_same_size(work, angle))

        if score > best_fine_score:
            best_fine_angle = float(angle)
            best_fine_score = float(score)

    return best_fine_angle


def detect_skew_angle(image: np.ndarray) -> float:
    thresh = build_text_mask(image)

    projection_angle = detect_projection_skew_angle(thresh)
    if projection_angle is not None:
        return projection_angle

    hough_angle = detect_hough_skew_angle(thresh)
    if hough_angle is not None:
        return hough_angle

    min_area_angle = detect_min_area_skew_angle(thresh)
    if min_area_angle is not None:
        return min_area_angle

    return 0.0


def estimate_border_color(image: np.ndarray):
    height, width = image.shape[:2]
    patch = max(5, min(height, width) // 100)

    if len(image.shape) == 2:
        corners = np.concatenate(
            [
                image[:patch, :patch].reshape(-1),
                image[:patch, width - patch:].reshape(-1),
                image[height - patch:, :patch].reshape(-1),
                image[height - patch:, width - patch:].reshape(-1),
            ],
            axis=0,
        )
        return int(np.median(corners))

    corners = np.concatenate(
        [
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, width - patch:].reshape(-1, 3),
            image[height - patch:, :patch].reshape(-1, 3),
            image[height - patch:, width - patch:].reshape(-1, 3),
        ],
        axis=0,
    )

    color = np.median(corners, axis=0)
    return tuple(int(channel) for channel in color)


def rotate_image_without_cutting(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    white = 255 if len(image.shape) == 2 else (255, 255, 255)

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=white,
    )


def deskew_image(image: np.ndarray, perspective_correct: bool = False) -> tuple[np.ndarray, float]:
    """
    Automatic deskew using the reference algorithm.

    Module 1 has a separate Perspective Correct panel, so automatic deskew does
    not secretly apply perspective correction by default.
    """
    result = image

    if perspective_correct:
        result = auto_perspective_correct(result)

    angle = detect_skew_angle(result)

    if abs(angle) >= MIN_ANGLE_TO_ROTATE:
        result = rotate_image_without_cutting(result, angle)
    else:
        result = result.copy()

    return result, angle


def apply_deskew(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    Backward-compatible helper for existing callers.

    If angle is None, runs automatic deskew. If angle is provided, it is treated
    as a correction angle in the same direction as the reference algorithm.
    """
    if angle is None:
        result, _ = deskew_image(image)
        return result

    if abs(angle) < MIN_ANGLE_TO_ROTATE:
        return image.copy()

    return rotate_image_without_cutting(image, angle)


def apply_manual_deskew(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Apply a user-specified rotation correction.

    The frontend's manual angle convention is preserved: positive means visual
    clockwise, so OpenCV receives the opposite sign.
    """
    if abs(angle_deg) < MIN_ANGLE_TO_ROTATE:
        return image.copy()

    return rotate_image_without_cutting(image, -angle_deg)
