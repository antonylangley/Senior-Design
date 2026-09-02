from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PaperDetection:
    found: bool
    corners: np.ndarray | None
    confidence: float
    area_ratio: float
    message: str | None = None


@dataclass(frozen=True)
class _Candidate:
    corners: np.ndarray
    score: float
    area_ratio: float


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]

    return ordered


def _candidate_from_contour(contour: np.ndarray, frame_area: float) -> _Candidate | None:
    area = cv2.contourArea(contour)
    area_ratio = area / frame_area
    if area_ratio < 0.08:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        corners = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        rect_area = max(cv2.contourArea(box.astype("float32")), 1.0)
        if area / rect_area < 0.65:
            return None
        corners = box.astype("float32")

    ordered = order_corners(corners)
    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])
    width = max(width_a, width_b)
    height = max(height_a, height_b)
    if width < 80 or height < 80:
        return None

    aspect = max(width, height) / max(min(width, height), 1.0)
    if aspect < 1.05 or aspect > 1.95:
        return None

    rect_area = max(width * height, 1.0)
    fill_ratio = min(area / rect_area, 1.0)
    paper_aspect = 11.0 / 8.5
    aspect_score = max(0.0, 1.0 - abs(aspect - paper_aspect) / 0.7)
    score = (0.55 * min(area_ratio / 0.65, 1.0)) + (0.25 * fill_ratio) + (0.20 * aspect_score)
    return _Candidate(corners=ordered, score=float(score), area_ratio=float(area_ratio))


def _find_candidates(mask_or_edges: np.ndarray, frame_area: float) -> list[_Candidate]:
    contours, _ = cv2.findContours(mask_or_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[_Candidate] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        candidate = _candidate_from_contour(contour, frame_area)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def detect_paper(frame: np.ndarray) -> PaperDetection:
    if frame is None or frame.size == 0:
        raise ValueError("frame must not be empty")

    height, width = frame.shape[:2]
    longest = max(height, width)
    scale = 1.0
    working = frame
    if longest > 1200:
        scale = longest / 1200.0
        working = cv2.resize(frame, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)

    work_h, work_w = working.shape[:2]
    frame_area = float(work_h * work_w)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    light_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 110, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    candidates = _find_candidates(light_mask, frame_area)

    if not candidates:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        candidates = _find_candidates(edges, frame_area)

    if not candidates:
        return PaperDetection(
            found=False,
            corners=None,
            confidence=0.0,
            area_ratio=0.0,
            message="Paper boundary could not be confidently detected.",
        )

    best = max(candidates, key=lambda item: item.score)
    confidence = max(0.0, min(best.score, 0.99))
    if confidence < 0.32:
        return PaperDetection(
            found=False,
            corners=None,
            confidence=confidence,
            area_ratio=best.area_ratio,
            message="Paper boundary could not be confidently detected.",
        )

    corners = best.corners * scale
    return PaperDetection(
        found=True,
        corners=order_corners(corners),
        confidence=confidence,
        area_ratio=best.area_ratio,
        message=None,
    )


def warp_perspective(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ordered = order_corners(corners)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    left_height = np.linalg.norm(ordered[3] - ordered[0])

    target_width = int(max(top_width, bottom_width))
    target_height = int(max(left_height, right_height))
    if target_width < 40 or target_height < 40:
        raise ValueError("detected paper is too small to warp")

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(frame, transform, (target_width, target_height))


def enhance_scan(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must not be empty")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.medianBlur(gray, 3)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    normalized = cv2.divide(gray, illumination, scale=255)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    binary = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    blended = cv2.addWeighted(contrast, 0.72, binary, 0.28, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
