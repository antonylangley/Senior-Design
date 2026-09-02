import numpy as np
import pytest

from app.vision.lego import BrickPose
from app.vision.lego_model import (
    canonical_brick_model,
    estimate_scene_scale,
    fit_canonical_model,
    ideal_body_polygon,
    local_to_image,
)


def _rough_pose(
    center: tuple[float, float], angle: float, scale: float, dimensions: tuple[int, int]
) -> BrickPose:
    model = canonical_brick_model(dimensions)
    polygon = ideal_body_polygon(model, center, angle, scale)
    return BrickPose(
        center=center,
        size=(model.body_size_mm[0] * scale, model.body_size_mm[1] * scale),
        angle_degrees=angle % 180,
        polygon=polygon,
    )


@pytest.mark.parametrize(
    ("dimensions", "expected"),
    [
        ((2, 2), [[-4, -4], [4, -4], [-4, 4], [4, 4]]),
        ((2, 3), [[-8, -4], [0, -4], [8, -4], [-8, 4], [0, 4], [8, 4]]),
        ((2, 4), [[-12, -4], [-4, -4], [4, -4], [12, -4],
                  [-12, 4], [-4, 4], [4, 4], [12, 4]]),
    ],
)
def test_canonical_stud_geometry(
    dimensions: tuple[int, int], expected: list[list[int]]
) -> None:
    model = canonical_brick_model(dimensions)

    assert model.stud_centers_mm.tolist() == expected
    assert model.stud_centers_mm.mean(axis=0).tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("center", "angle", "scale"),
    [((220.0, 180.0), 0.0, 6.0), ((420.5, 315.25), 0.0, 6.0),
     ((220.0, 180.0), 37.0, 6.0), ((220.0, 180.0), 37.0, 4.25)],
)
def test_perfect_two_by_four_fit_recovers_similarity_transform(
    center: tuple[float, float], angle: float, scale: float
) -> None:
    model = canonical_brick_model((2, 4))
    detected = local_to_image(model.stud_centers_mm, center, angle, scale).tolist()

    fit = fit_canonical_model((2, 4), detected, _rough_pose(center, angle, scale, (2, 4)))

    assert fit is not None
    assert fit.center == pytest.approx(center, abs=1e-5)
    assert fit.scale_px_per_mm == pytest.approx(scale, abs=1e-5)
    assert fit.angle_degrees == pytest.approx(angle % 180, abs=1e-5)
    assert fit.matched_studs == 8
    assert fit.reprojection_error_px == pytest.approx(0, abs=1e-5)


def test_two_by_four_with_one_missing_stud_still_fits() -> None:
    model = canonical_brick_model((2, 4))
    detected = local_to_image(model.stud_centers_mm, (300, 240), 24, 5.5).tolist()[1:]

    fit = fit_canonical_model((2, 4), detected, _rough_pose((300, 240), 24, 5.5, (2, 4)))

    assert fit is not None
    assert fit.matched_studs == 7
    assert fit.center == pytest.approx((300, 240), abs=1e-4)


def test_two_by_four_rejects_false_positive_stud() -> None:
    model = canonical_brick_model((2, 4))
    detected = local_to_image(model.stud_centers_mm, (300, 240), 24, 5.5).tolist()
    detected.append([480.0, 90.0])

    fit = fit_canonical_model((2, 4), detected, _rough_pose((300, 240), 24, 5.5, (2, 4)))

    assert fit is not None
    assert fit.matched_studs == 8
    assert fit.rejected_detected_indices == (8,)


def test_noisy_studs_recover_pose_within_tolerance() -> None:
    model = canonical_brick_model((2, 4))
    exact = local_to_image(model.stud_centers_mm, (315, 205), 63, 5.8)
    noise = np.array([[0.5, -0.3], [-0.4, 0.7], [0.2, 0.1], [-0.6, -0.2],
                      [0.3, 0.4], [-0.2, -0.5], [0.6, 0.2], [-0.3, 0.1]])

    fit = fit_canonical_model(
        (2, 4), (exact + noise).tolist(), _rough_pose((315, 205), 63, 5.8, (2, 4))
    )

    assert fit is not None
    assert fit.center == pytest.approx((315, 205), abs=0.35)
    assert fit.angle_degrees == pytest.approx(63, abs=0.3)
    assert fit.scale_px_per_mm == pytest.approx(5.8, abs=0.03)
    assert fit.reprojection_error_px < 0.8


def test_insufficient_evidence_returns_no_fit_for_contour_fallback() -> None:
    assert fit_canonical_model((2, 4), [(10, 10), (20, 20)]) is None


def test_ideal_body_polygon_uses_nominal_physical_dimensions() -> None:
    model = canonical_brick_model((2, 4))
    polygon = ideal_body_polygon(model, (100, 100), 0, 5)

    assert np.ptp(polygon[:, 0]) == pytest.approx(31.8 * 5)
    assert np.ptp(polygon[:, 1]) == pytest.approx(15.8 * 5)
    assert polygon.mean(axis=0).tolist() == pytest.approx([100, 100])


def test_scene_scale_median_rejects_obvious_outlier() -> None:
    model = canonical_brick_model((2, 4))
    fits = []
    for scale in (6.11, 6.15, 6.09, 7.72, 6.12):
        detected = local_to_image(model.stud_centers_mm, (200, 200), 20, scale).tolist()
        fit = fit_canonical_model((2, 4), detected, _rough_pose((200, 200), 20, scale, (2, 4)))
        assert fit is not None
        fits.append(fit)

    estimate = estimate_scene_scale(fits)

    assert estimate.scale_px_per_mm == pytest.approx(6.115, abs=0.01)
    assert estimate.sample_count == 4
    assert estimate.candidate_count == 5
    assert estimate.outlier_count == 1
    assert estimate.trustworthy is True


def test_two_by_two_reports_ninety_degree_symmetry() -> None:
    model = canonical_brick_model((2, 2))
    detected = local_to_image(model.stud_centers_mm, (150, 120), 127, 5).tolist()

    fit = fit_canonical_model((2, 2), detected, _rough_pose((150, 120), 127, 5, (2, 2)))

    assert fit is not None
    assert fit.rotational_symmetry_degrees == 90
    assert 0 <= fit.angle_degrees < 90
