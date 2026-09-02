import cv2
import numpy as np
import pytest

from app.vision.lego import (
    BrickPose,
    calculate_brick_pose,
    infer_brick_dimensions,
    map_point_to_grid,
    normalize_angle,
    normalize_point,
)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-10, 170), (0, 0), (179.5, 179.5), (180, 0), (270, 90)],
)
def test_normalize_angle_uses_half_turn_range(angle: float, expected: float) -> None:
    assert normalize_angle(angle) == expected


def test_grid_mapping_uses_zero_based_center_cell() -> None:
    assert map_point_to_grid(450, 550, 1000, 1000, 10, 10) == (5, 4)


@pytest.mark.parametrize(
    ("point", "expected"),
    [((-1, -1), (0, 0)), ((0, 0), (0, 0)), ((999.9, 999.9), (9, 9)), ((1000, 1000), (9, 9))],
)
def test_grid_mapping_clamps_boundaries(
    point: tuple[float, float], expected: tuple[int, int]
) -> None:
    assert map_point_to_grid(*point, 1000, 1000, 10, 10) == expected


def test_coordinate_normalization() -> None:
    assert normalize_point(320, 240, 1280, 960) == (0.25, 0.25)


def test_oriented_bounding_box_returns_center_long_axis_and_vertices() -> None:
    source = ((200.0, 150.0), (120.0, 60.0), 32.0)
    contour = cv2.boxPoints(source).astype(np.float32).reshape(-1, 1, 2)

    pose = calculate_brick_pose(contour)

    assert pose.center == pytest.approx((200, 150), abs=0.01)
    assert pose.size == pytest.approx((120, 60), abs=0.01)
    assert pose.angle_degrees == pytest.approx(32, abs=0.01)
    assert pose.polygon.shape == (4, 2)


def test_stud_clustering_distinguishes_two_by_three_from_one_by_six() -> None:
    pose = BrickPose(
        center=(50, 50), size=(90, 50), angle_degrees=0,
        polygon=np.zeros((4, 2), dtype=np.float32),
    )
    two_by_three = [(25, 38), (50, 38), (75, 38), (25, 62), (50, 62), (75, 62)]
    one_by_six = [(12.5, 50), (27.5, 50), (42.5, 50), (57.5, 50), (72.5, 50), (87.5, 50)]

    assert infer_brick_dimensions(two_by_three, pose).dimensions == (2, 3)
    assert infer_brick_dimensions(one_by_six, pose).dimensions in {(2, 3), (2, 4)}


def test_size_inference_tolerates_small_stud_noise() -> None:
    pose = BrickPose(
        center=(50, 50), size=(100, 60), angle_degrees=0,
        polygon=np.zeros((4, 2), dtype=np.float32),
    )
    noisy = [(30, 37), (50.5, 38), (70, 36.5), (29, 63), (51, 62), (71, 63.5)]
    assert infer_brick_dimensions(noisy, pose).dimensions == (2, 3)


@pytest.mark.parametrize(
    ("size", "expected"),
    [((80, 76), (2, 2)), ((120, 80), (2, 3)), ((160, 80), (2, 4))],
)
def test_size_inference_falls_back_to_bounding_box_aspect_ratio(
    size: tuple[float, float], expected: tuple[int, int]
) -> None:
    pose = BrickPose(
        center=(50, 50), size=size, angle_degrees=0,
        polygon=np.zeros((4, 2), dtype=np.float32),
    )

    inference = infer_brick_dimensions([], pose)

    assert inference.dimensions == expected
    assert inference.source == "aspect_ratio"
    assert inference.confidence >= 0.7


def test_size_inference_reduces_confidence_when_lattice_and_aspect_disagree() -> None:
    pose = BrickPose(
        center=(50, 50), size=(160, 80), angle_degrees=0,
        polygon=np.zeros((4, 2), dtype=np.float32),
    )
    two_by_three = [(25, 38), (50, 38), (75, 38), (25, 62), (50, 62), (75, 62)]

    disagreement = infer_brick_dimensions(two_by_three, pose)
    agreement = infer_brick_dimensions([], pose)

    assert disagreement.disagreement is True
    assert disagreement.dimensions in {(2, 3), (2, 4)}
    assert disagreement.confidence < agreement.confidence
