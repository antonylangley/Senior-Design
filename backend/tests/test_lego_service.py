import base64

import cv2
import numpy as np

from app.services.lego_service import LegoDetectionOptions, process_lego_image


def _image_payload(image: np.ndarray) -> str:
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def test_insufficient_stud_evidence_uses_contour_fallback() -> None:
    image = np.full((400, 600, 3), 225, dtype=np.uint8)
    cv2.rectangle(image, (180, 140), (340, 240), (190, 45, 35), thickness=-1)

    result = process_lego_image(_image_payload(image), LegoDetectionOptions())

    assert len(result.bricks) == 1
    brick = result.bricks[0]
    assert brick.stud_count == 0
    assert brick.pose_source == "contour_fallback"
    assert brick.model_fit is None
    assert brick.center_px == brick.raw_pose.center_px
    assert brick.bounding_polygon == brick.raw_pose.bounding_polygon
