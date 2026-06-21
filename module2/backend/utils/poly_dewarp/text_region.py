from __future__ import annotations

import cv2
import numpy as np


def detect_text_region(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Find a conservative text block from connected components and projections."""
    height, width = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[int, int, int, int]] = []
    max_component_height = max(4, round(height * 0.08))
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if (
            area >= 6
            and component_width >= 2
            and component_height >= 2
            and component_height <= max_component_height
        ):
            components.append((x, y, x + component_width, y + component_height))

    if len(components) < 8:
        return 0, round(height * 0.05), width - 1, round(height * 0.95)

    boxes = np.asarray(components)
    x0 = max(0, int(np.percentile(boxes[:, 0], 2)))
    y0 = max(0, int(np.percentile(boxes[:, 1], 2)))
    x1 = min(width - 1, int(np.percentile(boxes[:, 2], 98)))
    y1 = min(height - 1, int(np.percentile(boxes[:, 3], 98)))

    if x1 - x0 < width * 0.25 or y1 - y0 < height * 0.25:
        return 0, round(height * 0.05), width - 1, round(height * 0.95)
    return x0, y0, x1, y1

