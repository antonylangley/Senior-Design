import cv2
import numpy as np

from app.vision.rectification import RectificationConfig, rectify_image


def test_rectification_without_config_is_noop() -> None:
    image = np.full((80, 120, 3), 127, dtype=np.uint8)

    result = rectify_image(image)

    assert result.image is image
    assert result.active is False
    assert result.method == "none"
    assert result.homography is None


def test_manual_corner_rectification_warps_to_requested_output_size() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 15), (140, 105), (255, 255, 255), thickness=-1)
    corners = np.array([[20, 15], [140, 15], [140, 105], [20, 105]], dtype=np.float32)

    result = rectify_image(
        image,
        RectificationConfig(workspace_corners=corners, output_size=(240, 180)),
    )

    assert result.active is True
    assert result.method == "perspective"
    assert result.image.shape == (180, 240, 3)
    assert result.homography is not None
    assert result.image[90, 120].tolist() == [255, 255, 255]
