#!/usr/bin/env python3
"""
Standalone Python translation of ScanTailor Advanced's Select Content stage.

It reads images from ./input by default and writes annotated images plus JSON
content-box coordinates to ./output.

The implementation follows src/core/filters/select_content/ContentBoxFinder.cpp:
Wolf binarization -> shadow/garbage detection -> despeckle -> maximum
whitespace content-block construction -> text-mask estimation -> edge trimming.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from scipy import ndimage


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> "Rect":
        return cls(x, y, x + w - 1, y + h - 1)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left + 1)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top + 1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_empty(self) -> bool:
        return self.right < self.left or self.bottom < self.top

    def intersect(self, other: "Rect") -> "Rect":
        return Rect(max(self.left, other.left), max(self.top, other.top), min(self.right, other.right), min(self.bottom, other.bottom))

    def adjusted(self, dl: int, dt: int, dr: int, db: int) -> "Rect":
        return Rect(self.left + dl, self.top + dt, self.right + dr, self.bottom + db)

    def contains(self, other: "Rect") -> bool:
        return self.left <= other.left and self.top <= other.top and self.right >= other.right and self.bottom >= other.bottom

    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.top, self.bottom + 1), slice(self.left, self.right + 1)

    def to_xywh(self) -> list[int]:
        return [self.left, self.top, self.width, self.height]


@dataclass
class Region:
    known_new_obstacles: int
    bounds: Rect
    obstacles: list[Rect] = field(default_factory=list)

    def add_obstacles_from(self, other: "Region") -> None:
        for obstacle in other.obstacles:
            inter = obstacle.intersect(self.bounds)
            if not inter.is_empty():
                self.obstacles.append(inter)

    def add_new_obstacles(self, obstacles: list[Rect]) -> None:
        for obstacle in obstacles[self.known_new_obstacles :]:
            inter = obstacle.intersect(self.bounds)
            if not inter.is_empty():
                self.obstacles.append(inter)


class IntegralImage:
    def __init__(self, black: np.ndarray):
        self.integral = np.pad(black.astype(np.int64), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    def sum(self, rect: Rect) -> int:
        if rect.is_empty():
            return 0
        ii = self.integral
        l, t, r, b = rect.left, rect.top, rect.right + 1, rect.bottom + 1
        return int(ii[b, r] - ii[t, r] - ii[b, l] + ii[t, l])


class MaxWhitespaceFinder:
    AUTO_OBSTACLES = 0
    MANUAL_OBSTACLES = 1

    def __init__(self, black: np.ndarray, min_size: tuple[int, int] = (1, 1), quality: Optional[Callable[[Rect], int]] = None):
        self.integral = IntegralImage(black)
        self.min_w, self.min_h = min_size
        self.quality = quality or (lambda r: r.area)
        h, w = black.shape
        self.heap: list[tuple[int, int, Region]] = []
        self.seq = 0
        self.new_obstacles: list[Rect] = []
        self._push(Region(0, Rect.from_xywh(0, 0, w, h)))

    def _push(self, region: Region) -> None:
        heapq.heappush(self.heap, (-self.quality(region.bounds), self.seq, region))
        self.seq += 1

    def add_obstacle(self, obstacle: Rect) -> None:
        if len(self.heap) == 1:
            self.heap[0][2].obstacles.append(obstacle)
        else:
            self.new_obstacles.append(obstacle)

    def next(self, obstacle_mode: int = AUTO_OBSTACLES, max_iterations: int = 1000) -> Optional[Rect]:
        while max_iterations > 0 and self.heap:
            max_iterations -= 1
            _, _, top_region = heapq.heappop(self.heap)
            region = Region(top_region.known_new_obstacles, top_region.bounds, list(top_region.obstacles))
            region.add_new_obstacles(self.new_obstacles)

            if region.obstacles:
                self._subdivide_using_obstacles(region)
                continue

            if self.integral.sum(region.bounds) != 0:
                self._subdivide_using_raster(region)
                continue

            if obstacle_mode == self.AUTO_OBSTACLES:
                self.new_obstacles.append(region.bounds)
            return region.bounds
        return None

    def _subdivide_using_obstacles(self, region: Region) -> None:
        cx, cy = region.bounds.center()
        pivot = min(region.obstacles, key=lambda r: (cx - r.center()[0]) ** 2 + (cy - r.center()[1]) ** 2)
        self._subdivide(region, pivot)

    def _subdivide_using_raster(self, region: Region) -> None:
        pivot_pixel = self._find_black_pixel_close_to_center(region.bounds)
        pivot = self._extend_black_pixel_to_black_box(pivot_pixel, region.bounds)
        self._subdivide(region, pivot)

    def _subdivide(self, region: Region, pivot: Rect) -> None:
        bounds = region.bounds
        candidates = []
        if pivot.top - bounds.top >= self.min_h:
            candidates.append(Rect(bounds.left, bounds.top, bounds.right, pivot.top - 1))
        if bounds.bottom - pivot.bottom >= self.min_h:
            candidates.append(Rect(bounds.left, pivot.bottom + 1, bounds.right, bounds.bottom))
        if pivot.left - bounds.left >= self.min_w:
            candidates.append(Rect(bounds.left, bounds.top, pivot.left - 1, bounds.bottom))
        if bounds.right - pivot.right >= self.min_w:
            candidates.append(Rect(pivot.right + 1, bounds.top, bounds.right, bounds.bottom))

        for bounds in candidates:
            new_region = Region(len(self.new_obstacles), bounds)
            new_region.add_obstacles_from(region)
            self._push(new_region)

    def _find_black_pixel_close_to_center(self, non_white_rect: Rect) -> tuple[int, int]:
        assert self.integral.sum(non_white_rect) != 0
        cx, cy = non_white_rect.center()
        outer = non_white_rect
        inner = Rect.from_xywh(cx, cy, 1, 1)
        if self.integral.sum(inner) != 0:
            return cx, cy

        while True:
            if outer.width - inner.width <= 1 and outer.height - inner.height <= 1:
                break
            dl = inner.left - outer.left
            dr = outer.right - inner.right
            dt = inner.top - outer.top
            db = outer.bottom - inner.bottom
            middle = Rect(outer.left + ((dl + 1) >> 1), outer.top + ((dt + 1) >> 1), outer.right - (dr >> 1), outer.bottom - (db >> 1))
            if self.integral.sum(middle) == 0:
                inner = middle
            else:
                outer = middle

        if outer.left != inner.left:
            rect = Rect(outer.left, outer.top, outer.left, outer.bottom)
            if outer.height == 1:
                return (outer.left, outer.top) if self.integral.sum(rect) else (outer.right, outer.top)
            if self.integral.sum(rect):
                return self._find_black_pixel_close_to_center(rect)
        if outer.right != inner.right:
            rect = Rect(outer.right, outer.top, outer.right, outer.bottom)
            if outer.height == 1:
                return (outer.right, outer.top) if self.integral.sum(rect) else (outer.left, outer.top)
            if self.integral.sum(rect):
                return self._find_black_pixel_close_to_center(rect)
        if outer.top != inner.top:
            rect = Rect(outer.left, outer.top, outer.right, outer.top)
            if outer.width == 1:
                return (outer.left, outer.top) if self.integral.sum(rect) else (outer.left, outer.bottom)
            if self.integral.sum(rect):
                return self._find_black_pixel_close_to_center(rect)
        rect = Rect(outer.left, outer.bottom, outer.right, outer.bottom)
        if outer.width == 1:
            return outer.left, outer.bottom
        return self._find_black_pixel_close_to_center(rect)

    def _extend_black_pixel_to_black_box(self, pixel: tuple[int, int], bounds: Rect) -> Rect:
        outer = bounds
        inner = Rect.from_xywh(pixel[0], pixel[1], 1, 1)
        if self.integral.sum(outer) == outer.area:
            return outer

        while True:
            if outer.width - inner.width <= 1 and outer.height - inner.height <= 1:
                break
            dl = inner.left - outer.left
            dr = outer.right - inner.right
            dt = inner.top - outer.top
            db = outer.bottom - inner.bottom
            middle = Rect(outer.left + ((dl + 1) >> 1), outer.top + ((dt + 1) >> 1), outer.right - (dr >> 1), outer.bottom - (db >> 1))
            if self.integral.sum(middle) == middle.area:
                inner = middle
            else:
                outer = middle
        return inner


def binary_open_black(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w), max(1, h)))
    return cv2.morphologyEx(img.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0


def binary_dilate_black(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w), max(1, h)))
    return cv2.dilate(img.astype(np.uint8) * 255, kernel) > 0


def seed_fill(seed: np.ndarray, mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    allowed_seed = seed & mask
    if not allowed_seed.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n = ndimage.label(mask, structure=structure)
    seed_labels = np.unique(labels[allowed_seed])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, seed_labels)


def bounding_box(mask: np.ndarray) -> Optional[Rect]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return Rect(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def count_black(mask: np.ndarray, rect: Rect) -> int:
    if rect.is_empty():
        return 0
    return int(mask[rect.as_slice()].sum())


def wolf_binarize(gray: np.ndarray, window: tuple[int, int] = (51, 51), lower_bound: int = 1, upper_bound: int = 254, k: float = 0.3) -> np.ndarray:
    gray_f = gray.astype(np.float64)
    h, w = gray.shape
    wh, ww = window[1], window[0]
    lower_h = wh >> 1
    upper_h = wh - lower_h
    left_w = ww >> 1
    right_w = ww - left_w
    integral = np.pad(gray_f, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    integral_sq = np.pad(gray_f * gray_f, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    means = np.empty_like(gray_f)
    deviations = np.empty_like(gray_f)

    xs = np.arange(w)
    left = np.maximum(0, xs - left_w)
    right = np.minimum(w, xs + right_w)
    for y in range(h):
        top = max(0, y - lower_h)
        bottom = min(h, y + upper_h)
        area = (bottom - top) * (right - left)
        sums = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
        sqs = integral_sq[bottom, right] - integral_sq[top, right] - integral_sq[bottom, left] + integral_sq[top, left]
        mean = sums / area
        sqmean = sqs / area
        means[y] = mean
        deviations[y] = np.sqrt(np.abs(sqmean - mean * mean))

    max_dev = float(deviations.max()) or 1.0
    min_gray = float(gray.min())
    threshold = means - k * (1.0 - deviations / max_dev) * (means - min_gray)
    return (gray < lower_bound) | ((gray <= upper_bound) & (gray.astype(np.float64) < threshold))


def despeckle_normal_150dpi(src: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(src, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return src.copy()
    out = src.copy()
    objects = ndimage.find_objects(labels)
    big_threshold = 6
    for label, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        width = xs.stop - xs.start
        height = ys.stop - ys.start
        area = int((labels[slc] == label).sum())
        if width >= big_threshold or height >= big_threshold or area >= big_threshold * big_threshold:
            continue
        out[labels == label] = False
    return out


def trim_content_blocks_in_place(content: np.ndarray, content_blocks: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(content_blocks, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    out = content_blocks.copy()
    for label in range(1, n + 1):
        component = labels == label
        ys, xs = np.nonzero(component & content)
        if xs.size == 0:
            out[component] = False
            continue
        bounds = Rect(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        ys2, xs2 = np.nonzero(component)
        outside = (xs2 < bounds.left) | (xs2 > bounds.right) | (ys2 < bounds.top) | (ys2 > bounds.bottom)
        out[ys2[outside], xs2[outside]] = False
    return out


def remove_areas_touching_borders(content_blocks: np.ndarray) -> np.ndarray:
    h, w = content_blocks.shape
    max_spread = min(w, h) // 4
    visited = np.zeros_like(content_blocks, dtype=bool)
    queue: list[tuple[int, int, int]] = []
    for x in range(w):
        if content_blocks[0, x]:
            queue.append((x, 0, max_spread))
        if h > 1 and content_blocks[h - 1, x]:
            queue.append((x, h - 1, max_spread))
    for y in range(1, max(1, h - 1)):
        if content_blocks[y, 0]:
            queue.append((0, y, max_spread))
        if w > 1 and content_blocks[y, w - 1]:
            queue.append((w - 1, y, max_spread))

    head = 0
    while head < len(queue):
        x, y, dist = queue[head]
        head += 1
        if dist <= 0 or visited[y, x]:
            continue
        visited[y, x] = True
        nd = dist - 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and content_blocks[ny, nx] and not visited[ny, nx]:
                queue.append((nx, ny, nd))
    out = content_blocks.copy()
    out[visited] = False
    return out


def ultimate_eroded_points(content: np.ndarray) -> np.ndarray:
    dist = ndimage.distance_transform_edt(content)
    maxed = ndimage.maximum_filter(dist, size=3)
    return content & (dist > 0) & (dist == maxed)


def estimate_text_mask(content: np.ndarray, content_blocks: np.ndarray) -> np.ndarray:
    ueps = ultimate_eroded_points(content)
    text_mask = np.zeros_like(content, dtype=bool)
    labels, n = ndimage.label(content_blocks, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    min_text_height = 6
    objects = ndimage.find_objects(labels)

    for label, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        cc_img = labels[slc] == label
        content_img = content[slc] & cc_img
        hist = content_img.sum(axis=1).astype(int)
        block_hist = cc_img.sum(axis=1).astype(int)
        if hist.size == 0:
            continue

        ranges: list[tuple[int, int]] = []
        splittable = [(0, hist.size - 1)]
        while splittable:
            first, last = splittable.pop()
            if last - first < min_text_height - 1:
                continue
            seg = hist[first : last + 1]
            max_forward = np.maximum.accumulate(seg)
            max_back = np.maximum.accumulate(seg[::-1])[::-1]
            best_mag = -10**18
            best_split = None
            for p in range(first + 1, last):
                idx = p - first
                peak1 = int(max_forward[idx - 1])
                peak2 = int(max_back[idx + 1])
                if hist[p] * 3.5 > 0.5 * (peak1 + peak2):
                    continue
                shoulder1 = peak1 - int(hist[p])
                shoulder2 = peak2 - int(hist[p])
                if shoulder1 <= 0 or shoulder2 <= 0:
                    continue
                if min(shoulder1, shoulder2) * 20 < max(shoulder1, shoulder2):
                    continue
                magnitude = shoulder1 + shoulder2
                if magnitude > best_mag:
                    best_mag = magnitude
                    best_split = p
            if best_split is None:
                ranges.append((first, last))
            else:
                splittable.append((first, best_split - 1))
                splittable.append((best_split + 1, last))

        y_offset = slc[0].start
        x_offset = slc[1].start
        for first, last in ranges:
            if last - first < min_text_height - 1:
                continue
            weights = hist[first : last + 1]
            total = int(weights.sum())
            if total == 0:
                continue
            weighted_y = int((weights * np.arange(first, last + 1)).sum())
            center_y = int((weighted_y + total // 2) // total)
            top = center_y - min_text_height // 2
            bottom = top + min_text_height - 1
            if top < first or bottom > last:
                continue
            num_black = int(hist[top : bottom + 1].sum())
            num_total = int(block_hist[top : bottom + 1].sum())
            max_width = int(block_hist[top : bottom + 1].max(initial=0))
            if num_total == 0 or num_black < num_total * 0.22 or num_black > num_total * 0.65:
                continue
            while (top > first or bottom < last) and abs((center_y - top) - (bottom - center_y)) <= 1:
                new_top = top - 1 if top > first else top
                new_bottom = bottom + 1 if bottom < last else bottom
                new_black = num_black + int(hist[new_top]) + int(hist[new_bottom])
                new_total = num_total + int(block_hist[new_top]) + int(block_hist[new_bottom])
                if new_black < new_total * 0.22:
                    break
                num_black, num_total = new_black, new_total
                max_width = max(max_width, int(block_hist[new_top]), int(block_hist[new_bottom]))
                top, bottom = new_top, new_bottom
            if num_black > num_total * 0.65:
                continue
            if max_width < (bottom - top + 1) * 0.6:
                continue

            line_h = bottom - top + 1
            line_w = cc_img.shape[1]
            ueps_todo = int(0.4 * line_w / max(1, line_h))
            if ueps_todo:
                line_ueps = ueps[y_offset + top : y_offset + bottom + 1, x_offset : x_offset + cc_img.shape[1]] & cc_img[top : bottom + 1, :]
                _, uep_count = ndimage.label(line_ueps, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
                if uep_count < ueps_todo:
                    continue
            text_mask[y_offset + top : y_offset + bottom + 1, x_offset : x_offset + cc_img.shape[1]] |= cc_img[top : bottom + 1, :]
    return text_mask


def segment_garbage(garbage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hor = binary_open_black(garbage, (200, 1))
    vert = binary_open_black(garbage, (1, 200))
    if garbage.shape[0]:
        hor[0, :] |= garbage[0, :]
        hor[-1, :] |= garbage[-1, :]
    if garbage.shape[1]:
        vert[:, 0] |= garbage[:, 0]
        vert[:, -1] |= garbage[:, -1]

    seeds = vert | hor
    if seeds.any():
        _, labels = ndimage.distance_transform_edt(~seeds, return_indices=True)
        nearest_y, nearest_x = labels
        nearest_vert = vert[nearest_y, nearest_x]
        nearest_hor = hor[nearest_y, nearest_x]
        assigned_vert = garbage & nearest_vert
        assigned_hor = garbage & nearest_hor & ~nearest_vert
        vert |= assigned_vert
        hor |= assigned_hor
    unconnected = garbage & ~hor & ~vert
    hor |= unconnected
    vert |= unconnected
    return hor, vert


def distance_to_sources(sources: np.ndarray, borders: str = "none") -> np.ndarray:
    src = sources.copy()
    if borders == "vert":
        src[:, 0] = True
        src[:, -1] = True
    elif borders == "hor":
        src[0, :] = True
        src[-1, :] = True
    return ndimage.distance_transform_edt(~src)


@dataclass
class Garbage:
    image: np.ndarray
    borders: str
    _dist: Optional[np.ndarray] = None

    def add(self, content: np.ndarray, rect: Rect) -> None:
        self.image[rect.as_slice()] |= content[rect.as_slice()]
        self._dist = None

    def dist(self) -> np.ndarray:
        if self._dist is None:
            self._dist = distance_to_sources(self.image, self.borders)
        return self._dist


def trim(area: Rect, new_area: Rect, removed_area: Rect, content: np.ndarray, content_blocks: np.ndarray, text: np.ndarray, garbage: Garbage) -> tuple[Rect, bool]:
    if removed_area.area > 0.3 * new_area.area:
        if not (removed_area.width < 6 or removed_area.height < 6):
            return area, False

    content_pixels = count_black(content, removed_area)
    vertical_cut = new_area.top == area.top and new_area.bottom == area.bottom
    proximity_bias = 0.5 if vertical_cut else 0.65
    num_text_pixels = count_black(text, removed_area)
    if num_text_pixels == 0:
        proximity_bias = 0.4 if vertical_cut else 0.5
    else:
        total_pixels = content_pixels + count_black(garbage.image, removed_area) + 1
        upper_threshold = 5000
        text_influence = 1.0
        if num_text_pixels < upper_threshold:
            text_influence = 0.2 + (1.0 - 0.2) * math.log(max(1.0, float(num_text_pixels))) / math.log(float(upper_threshold))
        proximity_bias += (1.0 - proximity_bias) * text_influence * num_text_pixels / total_pixels
        proximity_bias = min(1.0, max(0.0, proximity_bias))

    remaining = np.zeros_like(content, dtype=bool)
    remaining[new_area.as_slice()] = content[new_area.as_slice()] & content_blocks[new_area.as_slice()]
    dist_to_others = distance_to_sources(remaining, "none")
    dist_to_garbage = garbage.dist()
    removed_mask = np.zeros_like(content, dtype=bool)
    removed_mask[removed_area.as_slice()] = content_blocks[removed_area.as_slice()]
    sum_garbage = float(dist_to_garbage[removed_mask].sum()) * proximity_bias
    sum_others = float(dist_to_others[removed_mask].sum()) * (1.0 - proximity_bias)

    if sum_garbage < sum_others:
        garbage.add(content, removed_area)
        return new_area, False
    return area, proximity_bias < 0.85


def trim_left(area: Rect, content: np.ndarray, content_blocks: np.ndarray, text: np.ndarray, garbage: Garbage) -> Rect:
    hist = content_blocks[area.as_slice()].sum(axis=0)
    start = 0
    while start < hist.size:
        first_ws = start
        while first_ws < hist.size and hist[first_ws] != 0:
            first_ws += 1
        first_non_ws = first_ws
        while first_non_ws < hist.size and hist[first_non_ws] == 0:
            first_non_ws += 1
        fw, fnw = first_ws + area.left, first_non_ws + area.left
        new_area = Rect(fnw, area.top, area.right, area.bottom)
        if new_area.is_empty():
            return area
        removed = Rect(area.left, area.top, fw - 1, area.bottom)
        if removed.is_empty():
            return new_area
        res, retry = trim(area, new_area, removed, content, content_blocks, text, garbage)
        if retry:
            start = fnw - area.left
        else:
            return res
    return area


def trim_right(area: Rect, content: np.ndarray, content_blocks: np.ndarray, text: np.ndarray, garbage: Garbage) -> Rect:
    hist = content_blocks[area.as_slice()].sum(axis=0)
    start = hist.size - 1
    while start >= 0:
        first_ws = start
        while first_ws >= 0 and hist[first_ws] != 0:
            first_ws -= 1
        first_non_ws = first_ws
        while first_non_ws >= 0 and hist[first_non_ws] == 0:
            first_non_ws -= 1
        fw, fnw = first_ws + area.left, first_non_ws + area.left
        new_area = Rect(area.left, area.top, fnw, area.bottom)
        if new_area.is_empty():
            return area
        removed = Rect(fw + 1, area.top, area.right, area.bottom)
        if removed.is_empty():
            return new_area
        res, retry = trim(area, new_area, removed, content, content_blocks, text, garbage)
        if retry:
            start = fnw - area.left
        else:
            return res
    return area


def trim_top(area: Rect, content: np.ndarray, content_blocks: np.ndarray, text: np.ndarray, garbage: Garbage) -> Rect:
    hist = content_blocks[area.as_slice()].sum(axis=1)
    start = 0
    while start < hist.size:
        first_ws = start
        while first_ws < hist.size and hist[first_ws] != 0:
            first_ws += 1
        first_non_ws = first_ws
        while first_non_ws < hist.size and hist[first_non_ws] == 0:
            first_non_ws += 1
        fw, fnw = first_ws + area.top, first_non_ws + area.top
        new_area = Rect(area.left, fnw, area.right, area.bottom)
        if new_area.is_empty():
            return area
        removed = Rect(area.left, area.top, area.right, fw - 1)
        if removed.is_empty():
            return new_area
        res, retry = trim(area, new_area, removed, content, content_blocks, text, garbage)
        if retry:
            start = fnw - area.top
        else:
            return res
    return area


def trim_bottom(area: Rect, content: np.ndarray, content_blocks: np.ndarray, text: np.ndarray, garbage: Garbage) -> Rect:
    hist = content_blocks[area.as_slice()].sum(axis=1)
    start = hist.size - 1
    while start >= 0:
        first_ws = start
        while first_ws >= 0 and hist[first_ws] != 0:
            first_ws -= 1
        first_non_ws = first_ws
        while first_non_ws >= 0 and hist[first_non_ws] == 0:
            first_non_ws -= 1
        fw, fnw = first_ws + area.top, first_non_ws + area.top
        new_area = Rect(area.left, area.top, area.right, fnw)
        if new_area.is_empty():
            return area
        removed = Rect(area.left, fw + 1, area.right, area.bottom)
        if removed.is_empty():
            return new_area
        res, retry = trim(area, new_area, removed, content, content_blocks, text, garbage)
        if retry:
            start = fnw - area.top
        else:
            return res
    return area


def filter_shadows(shadows: np.ndarray) -> np.ndarray:
    borders = np.zeros_like(shadows, dtype=bool)
    borders[0, :] = borders[-1, :] = True
    borders[:, 0] = borders[:, -1] = True
    touching = seed_fill(borders, shadows, 8)
    non_border = shadows ^ touching
    if non_border.any():
        inv = ~non_border
        mask = seed_fill(borders, inv, 8) | non_border
        text_mask = estimate_text_mask(inv, mask)
        misclassified = seed_fill(text_mask, non_border, 8)
        non_border ^= misclassified
    return non_border | touching


def find_content_box(gray: np.ndarray, dpi: tuple[float, float], page_rect: Optional[Rect] = None) -> Optional[Rect]:
    h0, w0 = gray.shape
    dpi_x, dpi_y = dpi
    sx = 150.0 / dpi_x
    sy = 150.0 / dpi_y
    w150 = max(1, int(round(w0 * sx)))
    h150 = max(1, int(round(h0 * sy)))
    gray150 = cv2.resize(gray, (w150, h150), interpolation=cv2.INTER_AREA if sx < 1.0 or sy < 1.0 else cv2.INTER_CUBIC)
    bw = wolf_binarize(gray150, (51, 51), 1, 254, 0.3)

    if page_rect is None:
        page150 = Rect.from_xywh(0, 0, w150, h150)
    else:
        page150 = Rect(
            max(0, int(round(page_rect.left * sx))),
            max(0, int(round(page_rect.top * sy))),
            min(w150 - 1, int(round(page_rect.right * sx))),
            min(h150 - 1, int(round(page_rect.bottom * sy))),
        )

    page_mask = np.zeros_like(bw, dtype=bool)
    if not page150.is_empty():
        page_mask[page150.as_slice()] = True
    bw[~page_mask] = True

    hor_shadows_seed = binary_open_black(bw, (200, 14))
    ver_shadows_seed = binary_open_black(bw, (14, 300))
    shadows_seed = hor_shadows_seed | ver_shadows_seed
    dilated = binary_dilate_black(bw, (3, 3))
    shadows_dilated = seed_fill(shadows_seed, dilated, 8)
    garbage_img = shadows_dilated & bw
    garbage_img = filter_shadows(garbage_img)

    content = bw & ~garbage_img
    despeckled = despeckle_normal_150dpi(content)
    content_blocks = np.ones_like(content, dtype=bool)
    area_threshold = min(content.shape)

    hor_ws = MaxWhitespaceFinder(despeckled, quality=lambda r: r.width * r.width * r.height)
    for _ in range(80):
        ws = hor_ws.next(MaxWhitespaceFinder.MANUAL_OBSTACLES)
        if ws is None or ws.area < area_threshold:
            break
        content_blocks[ws.as_slice()] = False
        height_fraction = ws.height // 5
        obstacle = Rect(ws.left, ws.top + height_fraction, ws.right, ws.bottom - height_fraction)
        if not obstacle.is_empty():
            hor_ws.add_obstacle(obstacle)

    vert_ws = MaxWhitespaceFinder(despeckled, quality=lambda r: r.width * r.height * r.height)
    for _ in range(40):
        ws = vert_ws.next(MaxWhitespaceFinder.MANUAL_OBSTACLES)
        if ws is None or ws.area < area_threshold:
            break
        content_blocks[ws.as_slice()] = False
        width_fraction = ws.width // 5
        obstacle = Rect(ws.left + width_fraction, ws.top, ws.right - width_fraction, ws.bottom)
        if not obstacle.is_empty():
            vert_ws.add_obstacle(obstacle)

    content_blocks = trim_content_blocks_in_place(despeckled, content_blocks)

    tmp = content | ~content_blocks
    ws_finder = MaxWhitespaceFinder(tmp, min_size=(4, 4))
    for _ in range(10):
        ws = ws_finder.next()
        if ws is None or ws.area < area_threshold:
            break
        content_blocks[ws.as_slice()] = False

    content_blocks = trim_content_blocks_in_place(despeckled, content_blocks)
    content_blocks = remove_areas_touching_borders(content_blocks)
    text_mask = estimate_text_mask(content, content_blocks) & content

    rect = bounding_box(content_blocks)
    if rect is None:
        return None

    hor_g, vert_g = segment_garbage(garbage_img)
    hor_garbage = Garbage(hor_g, "hor")
    vert_garbage = Garbage(vert_g, "vert")

    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 4, 8
    side_mask = LEFT | RIGHT | TOP | BOTTOM
    while side_mask and not rect.is_empty():
        old = rect
        if side_mask & LEFT:
            side_mask &= ~LEFT
            rect = trim_left(rect, content, content_blocks, text_mask, vert_garbage)
            if rect.is_empty():
                break
            if old != rect:
                side_mask |= LEFT | TOP | BOTTOM
        old = rect
        if side_mask & RIGHT:
            side_mask &= ~RIGHT
            rect = trim_right(rect, content, content_blocks, text_mask, vert_garbage)
            if rect.is_empty():
                break
            if old != rect:
                side_mask |= RIGHT | TOP | BOTTOM
        old = rect
        if side_mask & TOP:
            side_mask &= ~TOP
            rect = trim_top(rect, content, content_blocks, text_mask, hor_garbage)
            if rect.is_empty():
                break
            if old != rect:
                side_mask |= TOP | LEFT | RIGHT
        old = rect
        if side_mask & BOTTOM:
            side_mask &= ~BOTTOM
            rect = trim_bottom(rect, content, content_blocks, text_mask, hor_garbage)
            if rect.is_empty():
                break
            if old != rect:
                side_mask |= BOTTOM | LEFT | RIGHT

        if rect.width < 8 or rect.height < 8:
            return None
        if rect.width < 30 and rect.height > rect.width * 20:
            return None

    if rect.is_empty():
        return None
    rect = rect.adjusted(-1, -1, 1, 1).intersect(Rect.from_xywh(0, 0, w150, h150))

    out = Rect(
        max(0, int(math.floor(rect.left / sx))),
        max(0, int(math.floor(rect.top / sy))),
        min(w0 - 1, int(math.ceil((rect.right + 1) / sx)) - 1),
        min(h0 - 1, int(math.ceil((rect.bottom + 1) / sy)) - 1),
    )
    if page_rect is not None:
        out = out.intersect(page_rect)
    return None if out.is_empty() else out


def read_dpi(path: Path, fallback: float) -> tuple[float, float]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            dpi = img.info.get("dpi")
            if dpi:
                return float(dpi[0] or fallback), float(dpi[1] or fallback)
    except Exception:
        pass
    return fallback, fallback


def annotate(image: np.ndarray, rect: Optional[Rect]) -> np.ndarray:
    out = image.copy()
    if rect is None:
        return out
    overlay = out.copy()
    cv2.rectangle(overlay, (rect.left, rect.top), (rect.right, rect.bottom), (255, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    cv2.rectangle(out, (rect.left, rect.top), (rect.right, rect.bottom), (255, 0, 0), thickness=max(2, round(min(out.shape[:2]) / 350)))
    return out


def parse_page_box(value: Optional[str]) -> Optional[Rect]:
    if not value:
        return None
    parts = [int(round(float(x))) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--page-box must be x,y,width,height")
    return Rect.from_xywh(parts[0], parts[1], parts[2], parts[3])


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ScanTailor-style Select Content detection on images.")
    parser.add_argument("--input", default="input", type=Path, help="Input folder containing images.")
    parser.add_argument("--output", default="output", type=Path, help="Output folder for annotated images and JSON boxes.")
    parser.add_argument("--dpi", default=300.0, type=float, help="Fallback image DPI when metadata is missing.")
    parser.add_argument("--page-box", default=None, help="Optional page rectangle: x,y,width,height in original image pixels.")
    args = parser.parse_args()

    args.input.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    page_rect = parse_page_box(args.page_box)
    paths = list(iter_images(args.input))
    if not paths:
        print(f"No images found in {args.input}. Put images there and run again.")
        return 0

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable: {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dpi = read_dpi(path, args.dpi)
        rect = find_content_box(gray, dpi, page_rect)
        annotated = annotate(image, rect)
        out_image = args.output / f"{path.stem}_content_box.png"
        out_json = args.output / f"{path.stem}_content_box.json"
        cv2.imwrite(str(out_image), annotated)
        payload = {
            "source": str(path),
            "dpi": {"x": dpi[0], "y": dpi[1]},
            "content_box": None if rect is None else {"x": rect.left, "y": rect.top, "width": rect.width, "height": rect.height},
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{path.name}: {'no content found' if rect is None else rect.to_xywh()} -> {out_image}")
    return 0


# ── API Wrappers for Module 1 Backend ─────────────────────────────────────────

def detect_content_box_api(image: np.ndarray, dpi: int = 300) -> dict:
    """
    Detect the content box for an image and return coordinates + annotated BGR image.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    dpi_pair = (float(dpi), float(dpi))
    rect = find_content_box(gray, dpi_pair, None)

    if rect is None:
        return {
            "content_box": None,
            "content_rect": None,
            "image_width": w,
            "image_height": h,
            "annotated_image": image.copy(),
        }

    annotated = annotate(image, rect)
    return {
        "content_box": {
            "x": rect.left,
            "y": rect.top,
            "width": rect.width,
            "height": rect.height,
        },
        "content_rect": [rect.left, rect.top, rect.right, rect.bottom],
        "image_width": w,
        "image_height": h,
        "annotated_image": annotated,
    }


def apply_content_selection(image: np.ndarray, dpi: int = 300) -> np.ndarray:
    """
    Detect content box and crop the original image to that area.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    dpi_pair = (float(dpi), float(dpi))
    rect = find_content_box(gray, dpi_pair, None)

    if rect is None:
        return image.copy()

    # Clamp to image bounds
    x1 = max(0, rect.left)
    y1 = max(0, rect.top)
    x2 = min(w, rect.right + 1)
    y2 = min(h, rect.bottom + 1)

    if x2 <= x1 or y2 <= y1:
        return image.copy()

    return image[y1:y2, x1:x2].copy()


if __name__ == "__main__":
    raise SystemExit(main())
