from dataclasses import dataclass
from math import floor

import cv2
import numpy as np


@dataclass(frozen=True)
class BrickPose:
    center: tuple[float, float]
    size: tuple[float, float]
    angle_degrees: float
    polygon: np.ndarray


@dataclass(frozen=True)
class SegmentedBrick:
    contour: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class DimensionInference:
    dimensions: tuple[int, int] | None
    confidence: float
    source: str
    disagreement: bool = False


SUPPORTED_DIMENSIONS = ((2, 2), (2, 3), (2, 4))


def normalize_angle(angle_degrees: float) -> float:
    """Normalize a directionless rectangular pose to [0, 180)."""
    return float(angle_degrees % 180.0)


def calculate_brick_pose(contour: np.ndarray) -> BrickPose:
    """Return yaw of the rectangle's long axis, measured clockwise in image coordinates."""
    (cx, cy), (width, height), angle = cv2.minAreaRect(contour)
    if width < height:
        angle += 90.0
        width, height = height, width
    polygon = cv2.boxPoints(((cx, cy), (width, height), angle)).astype(np.float32)
    return BrickPose(
        center=(float(cx), float(cy)),
        size=(float(width), float(height)),
        angle_degrees=normalize_angle(angle),
        polygon=polygon,
    )


def normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (float(x / width), float(y / height))


def map_point_to_grid(
    x: float, y: float, width: int, height: int, rows: int, columns: int
) -> tuple[int, int]:
    column = min(columns - 1, max(0, floor(x / width * columns)))
    row = min(rows - 1, max(0, floor(y / height * rows)))
    return row, column


def segment_bricks(frame: np.ndarray) -> tuple[np.ndarray, list[SegmentedBrick]]:
    """Segment objects that differ from the image-border background in LAB space."""
    height, width = frame.shape[:2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(3, int(min(height, width) * 0.035))
    border_pixels = np.concatenate(
        [lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
         lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)]
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mask = np.where((distance > 24) | (saturation > 65), 255, 0).astype(np.uint8)
    kernel_size = max(3, int(round(min(height, width) / 180)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(width * height)
    instances: list[SegmentedBrick] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < max(250.0, frame_area * 0.0007) or area > frame_area * 0.75:
            continue
        rect = cv2.minAreaRect(contour)
        rect_area = max(rect[1][0] * rect[1][1], 1.0)
        if area / rect_area < 0.45 or min(rect[1]) < 10:
            continue
        instance_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(instance_mask, [contour], -1, 255, thickness=cv2.FILLED)
        instances.append(SegmentedBrick(contour=contour, mask=instance_mask))
    return mask, instances


def detect_studs(frame: np.ndarray, brick: SegmentedBrick, pose: BrickPose) -> list[tuple[float, float]]:
    x, y, w, h = cv2.boundingRect(brick.contour)
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    roi_mask = brick.mask[y:y + h, x:x + w]
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    min_radius = max(3, int(min(pose.size) / 15))
    max_radius = max(min_radius + 1, int(min(pose.size) / 4))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, min_radius * 2.2),
        param1=70, param2=16, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []
    centers: list[tuple[float, float]] = []
    for local_x, local_y, _ in circles[0]:
        ix, iy = int(round(local_x)), int(round(local_y))
        if 0 <= iy < roi_mask.shape[0] and 0 <= ix < roi_mask.shape[1] and roi_mask[iy, ix]:
            centers.append((float(local_x + x), float(local_y + y)))
    return centers


def _cluster_axis(values: np.ndarray, tolerance: float) -> list[float]:
    if len(values) == 0:
        return []
    clusters: list[list[float]] = [[float(np.min(values))]]
    for value in np.sort(values)[1:]:
        if float(value) - float(np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return [float(np.mean(cluster)) for cluster in clusters]


def infer_brick_dimensions(
    stud_centers: list[tuple[float, float]], pose: BrickPose
) -> DimensionInference:
    """Score supported sizes using aspect ratio, stud count, and lattice geometry."""
    aspect_ratio = max(pose.size) / max(min(pose.size), 1.0)
    lattice: tuple[int, int] | None = None
    lattice_quality = 0.0
    if stud_centers:
        points = np.asarray(stud_centers, dtype=np.float32) - np.asarray(pose.center, dtype=np.float32)
        radians = np.deg2rad(pose.angle_degrees)
        long_axis = np.array([np.cos(radians), np.sin(radians)], dtype=np.float32)
        short_axis = np.array([-np.sin(radians), np.cos(radians)], dtype=np.float32)
        local_x, local_y = points @ long_axis, points @ short_axis
        tolerance = max(5.0, min(pose.size) * 0.16)
        columns = len(_cluster_axis(local_x, tolerance))
        rows = len(_cluster_axis(local_y, tolerance))
        candidate = (min(rows, columns), max(rows, columns))
        if candidate in SUPPORTED_DIMENSIONS:
            lattice = candidate
            lattice_quality = min(1.0, len(stud_centers) / (candidate[0] * candidate[1]))

    scores: dict[tuple[int, int], float] = {}
    for dimensions in SUPPORTED_DIMENSIONS:
        expected_ratio = dimensions[1] / dimensions[0]
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - expected_ratio) / 0.55)
        expected_studs = dimensions[0] * dimensions[1]
        count_score = max(0.0, 1.0 - abs(len(stud_centers) - expected_studs) / expected_studs)
        if lattice is None:
            score = (0.80 * aspect_score) + (0.20 * count_score)
        else:
            lattice_score = lattice_quality if dimensions == lattice else 0.0
            score = (0.40 * aspect_score) + (0.15 * count_score) + (0.45 * lattice_score)
        scores[dimensions] = score

    best_dimensions = max(scores, key=scores.get)
    best_score = scores[best_dimensions]
    aspect_choice = min(
        SUPPORTED_DIMENSIONS,
        key=lambda dimensions: abs(aspect_ratio - dimensions[1] / dimensions[0]),
    )
    disagreement = lattice is not None and lattice != aspect_choice
    if disagreement:
        best_score = max(0.0, best_score - 0.12)
    if best_score < 0.42:
        return DimensionInference(None, best_score, "unresolved", disagreement)

    if lattice == best_dimensions and aspect_choice == best_dimensions:
        source = "stud_lattice+aspect_ratio"
    elif lattice == best_dimensions:
        source = "stud_lattice"
    else:
        source = "aspect_ratio"
    return DimensionInference(best_dimensions, min(best_score, 0.98), source, disagreement)


def classify_brick_color(frame: np.ndarray, mask: np.ndarray) -> tuple[str, list[float], list[float]]:
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    pixels = eroded > 0
    if not np.any(pixels):
        pixels = mask > 0
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = np.median(hsv_image[pixels], axis=0)
    lab = np.median(lab_image[pixels], axis=0)
    hue, saturation, value = hsv
    if value < 55:
        label = "black"
    elif saturation < 38:
        label = "white" if value > 205 else "gray"
    elif hue < 7 or hue >= 170:
        label = "red"
    elif hue < 18:
        label = "orange" if value > 120 else "brown"
    elif hue < 36:
        label = "yellow"
    elif hue < 88:
        label = "green"
    elif hue < 135:
        label = "blue"
    elif hue < 170:
        label = "red"
    else:
        label = "other"
    return label, [float(v) for v in hsv], [float(v) for v in lab]
