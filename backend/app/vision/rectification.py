from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RectificationConfig:
    """Planar preprocessing config; camera calibration fields can be added here later."""

    workspace_corners: np.ndarray | None = None
    output_size: tuple[int, int] | None = None
    camera_matrix: np.ndarray | None = None
    distortion_coefficients: np.ndarray | None = None
    pixels_per_millimeter: float | None = None


@dataclass(frozen=True)
class RectificationResult:
    image: np.ndarray
    active: bool
    method: str
    homography: np.ndarray | None = None


def _ordered_corners(points: np.ndarray) -> np.ndarray:
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    differences = np.diff(corners, axis=1).reshape(4)
    ordered[0] = corners[np.argmin(sums)]
    ordered[1] = corners[np.argmin(differences)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[3] = corners[np.argmax(differences)]
    return ordered


def _estimated_output_size(corners: np.ndarray) -> tuple[int, int]:
    top_left, top_right, bottom_right, bottom_left = corners
    width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    return max(1, round(float(width))), max(1, round(float(height)))


def rectify_image(frame: np.ndarray, config: RectificationConfig | None = None) -> RectificationResult:
    """Optionally undistort and warp a planar workspace; without config this is a no-op."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must not be empty")
    if config is None:
        return RectificationResult(image=frame, active=False, method="none")

    processed = frame
    methods: list[str] = []
    if config.camera_matrix is not None and config.distortion_coefficients is not None:
        processed = cv2.undistort(processed, config.camera_matrix, config.distortion_coefficients)
        methods.append("undistort")

    homography = None
    if config.workspace_corners is not None:
        source = _ordered_corners(config.workspace_corners)
        width, height = config.output_size or _estimated_output_size(source)
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        processed = cv2.warpPerspective(processed, homography, (width, height))
        methods.append("perspective")

    return RectificationResult(
        image=processed,
        active=bool(methods),
        method="+".join(methods) if methods else "none",
        homography=homography,
    )
