from dataclasses import dataclass

import numpy as np

from app.vision.image_io import (
    bytes_to_cv2,
    cv2_to_bytes,
    cv2_to_data_url,
    decode_image_data,
    resize_to_max_side,
)
from app.vision.paper import PaperDetection, detect_paper, enhance_scan, warp_perspective


@dataclass(frozen=True)
class ProcessedScan:
    paper_detected: bool
    used_full_frame: bool
    confidence: float
    corners: list[dict[str, float]] | None
    processed_image: np.ndarray
    processed_image_data: str
    ai_image_bytes: bytes
    ai_mime_type: str
    warning: str | None


def _corners_to_points(detection: PaperDetection) -> list[dict[str, float]] | None:
    if detection.corners is None:
        return None
    return [{"x": float(x), "y": float(y)} for x, y in detection.corners]


def process_scan_image(image_data: str, use_full_frame_on_failure: bool = False) -> ProcessedScan:
    image_bytes, _ = decode_image_data(image_data)
    frame = bytes_to_cv2(image_bytes)
    detection = detect_paper(frame)

    used_full_frame = False
    warning = detection.message
    if detection.found and detection.corners is not None:
        scan = warp_perspective(frame, detection.corners)
        processed = enhance_scan(scan)
    else:
        used_full_frame = use_full_frame_on_failure
        processed = resize_to_max_side(frame, 1400)
        if use_full_frame_on_failure:
            warning = "Paper boundary could not be confidently detected. Submitted the full frame instead."
        else:
            warning = "Paper boundary could not be confidently detected."

    display_image = resize_to_max_side(processed, 1400)
    ai_image = resize_to_max_side(processed, 1400)
    return ProcessedScan(
        paper_detected=detection.found,
        used_full_frame=used_full_frame,
        confidence=detection.confidence,
        corners=_corners_to_points(detection),
        processed_image=display_image,
        processed_image_data=cv2_to_data_url(display_image, "image/png"),
        ai_image_bytes=cv2_to_bytes(ai_image, "image/png"),
        ai_mime_type="image/png",
        warning=warning,
    )


def detect_paper_from_image_data(image_data: str) -> tuple[PaperDetection, int, int]:
    image_bytes, _ = decode_image_data(image_data)
    frame = bytes_to_cv2(image_bytes)
    detection = detect_paper(frame)
    height, width = frame.shape[:2]
    return detection, width, height
