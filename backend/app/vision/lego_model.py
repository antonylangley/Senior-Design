from dataclasses import dataclass
from itertools import permutations
from math import ceil

import numpy as np

from app.vision.lego import BrickPose, normalize_angle


STUD_PITCH_MM = 8.0
STUD_DIAMETER_MM = 4.8
BRICK_HEIGHT_MM = 9.6


@dataclass(frozen=True)
class CanonicalBrickModel:
    dimensions: tuple[int, int]
    stud_centers_mm: np.ndarray
    body_size_mm: tuple[float, float]
    rotational_symmetry_degrees: int


@dataclass(frozen=True)
class ModelFit:
    center: tuple[float, float]
    angle_degrees: float
    scale_px_per_mm: float
    polygon: np.ndarray
    predicted_studs: np.ndarray
    matched_detected_indices: tuple[int, ...]
    matched_model_indices: tuple[int, ...]
    rejected_detected_indices: tuple[int, ...]
    reprojection_error_px: float
    confidence: float
    expected_studs: int
    rotational_symmetry_degrees: int

    @property
    def matched_studs(self) -> int:
        return len(self.matched_detected_indices)


@dataclass(frozen=True)
class SceneScaleEstimate:
    scale_px_per_mm: float | None
    confidence: float
    relative_variation: float | None
    sample_count: int
    candidate_count: int
    outlier_count: int
    trustworthy: bool


_BODY_SIZES = {
    (2, 2): (15.8, 15.8),
    (2, 3): (23.8, 15.8),
    (2, 4): (31.8, 15.8),
}


def canonical_brick_model(dimensions: tuple[int, int]) -> CanonicalBrickModel:
    if dimensions not in _BODY_SIZES:
        raise ValueError(f"unsupported LEGO footprint: {dimensions}")
    rows, columns = dimensions
    x_values = (np.arange(columns, dtype=np.float64) - (columns - 1) / 2) * STUD_PITCH_MM
    y_values = (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * STUD_PITCH_MM
    studs = np.array([(x, y) for y in y_values for x in x_values], dtype=np.float64)
    return CanonicalBrickModel(
        dimensions=dimensions,
        stud_centers_mm=studs,
        body_size_mm=_BODY_SIZES[dimensions],
        rotational_symmetry_degrees=90 if dimensions == (2, 2) else 180,
    )


def local_to_image(
    points: np.ndarray,
    center: tuple[float, float],
    angle_degrees: float,
    scale_px_per_mm: float,
) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=np.float64,
    )
    return scale_px_per_mm * np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(center)


def ideal_body_polygon(
    model: CanonicalBrickModel,
    center: tuple[float, float],
    angle_degrees: float,
    scale_px_per_mm: float,
) -> np.ndarray:
    width, height = model.body_size_mm
    local_corners = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2],
         [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float64,
    )
    return local_to_image(local_corners, center, angle_degrees, scale_px_per_mm).astype(np.float32)


def _transform_from_pairs(
    model_a: np.ndarray,
    model_b: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    model_vector = model_b - model_a
    image_vector = image_b - image_a
    model_distance = np.linalg.norm(model_vector)
    if model_distance < 1e-6:
        return None
    scale = float(np.linalg.norm(image_vector) / model_distance)
    if scale <= 0.1:
        return None
    angle = float(
        np.degrees(np.arctan2(image_vector[1], image_vector[0]))
        - np.degrees(np.arctan2(model_vector[1], model_vector[0]))
    )
    projected_a = local_to_image(model_a[None, :], (0, 0), angle, scale)[0]
    center = image_a - projected_a
    return scale, angle, center


def _associate(
    predicted: np.ndarray, detected: np.ndarray, threshold: float
) -> list[tuple[int, int, float]]:
    distances = np.linalg.norm(predicted[:, None, :] - detected[None, :, :], axis=2)
    candidates = sorted(
        (float(distances[model_index, detected_index]), model_index, detected_index)
        for model_index in range(len(predicted))
        for detected_index in range(len(detected))
        if distances[model_index, detected_index] <= threshold
    )
    used_models: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, model_index, detected_index in candidates:
        if model_index in used_models or detected_index in used_detections:
            continue
        used_models.add(model_index)
        used_detections.add(detected_index)
        matches.append((model_index, detected_index, distance))
    return matches


def _fit_similarity(
    model_points: np.ndarray,
    image_points: np.ndarray,
    fixed_scale: float | None = None,
) -> tuple[float, float, np.ndarray]:
    model_center = model_points.mean(axis=0)
    image_center = image_points.mean(axis=0)
    model_centered = model_points - model_center
    image_centered = image_points - image_center
    model_complex = model_centered[:, 0] + 1j * model_centered[:, 1]
    image_complex = image_centered[:, 0] + 1j * image_centered[:, 1]
    coefficient = np.vdot(model_complex, image_complex) / np.vdot(model_complex, model_complex)
    angle = float(np.degrees(np.angle(coefficient)))
    scale = fixed_scale if fixed_scale is not None else float(abs(coefficient))
    transformed_model_center = local_to_image(
        model_center[None, :], (0, 0), angle, scale
    )[0]
    center = image_center - transformed_model_center
    return scale, angle, center


def _angle_distance(angle_a: float, angle_b: float, symmetry: int) -> float:
    difference = abs((angle_a - angle_b) % symmetry)
    return min(difference, symmetry - difference)


def fit_canonical_model(
    dimensions: tuple[int, int],
    detected_studs: list[tuple[float, float]],
    rough_pose: BrickPose | None = None,
    fixed_scale: float | None = None,
) -> ModelFit | None:
    model = canonical_brick_model(dimensions)
    detected = np.asarray(detected_studs, dtype=np.float64).reshape(-1, 2)
    minimum_matches = max(3, ceil(len(model.stud_centers_mm) * 0.60))
    if len(detected) < minimum_matches:
        return None

    rough_scale = None
    if rough_pose is not None:
        rough_scale = float(np.median([
            rough_pose.size[0] / model.body_size_mm[0],
            rough_pose.size[1] / model.body_size_mm[1],
        ]))

    best: tuple[tuple[float, ...], float, float, np.ndarray, list[tuple[int, int, float]]] | None = None
    model_pairs = list(permutations(range(len(model.stud_centers_mm)), 2))
    detected_pairs = list(permutations(range(len(detected)), 2))
    for model_a_index, model_b_index in model_pairs:
        for image_a_index, image_b_index in detected_pairs:
            hypothesis = _transform_from_pairs(
                model.stud_centers_mm[model_a_index], model.stud_centers_mm[model_b_index],
                detected[image_a_index], detected[image_b_index],
            )
            if hypothesis is None:
                continue
            scale, angle, center = hypothesis
            if fixed_scale is not None and abs(scale - fixed_scale) / fixed_scale > 0.25:
                continue
            if rough_scale is not None and abs(scale - rough_scale) / max(rough_scale, 1e-6) > 0.55:
                continue
            working_scale = fixed_scale or scale
            threshold = max(2.5, STUD_PITCH_MM * working_scale * 0.24)
            predicted = local_to_image(model.stud_centers_mm, tuple(center), angle, working_scale)
            matches = _associate(predicted, detected, threshold)
            if len(matches) < minimum_matches:
                continue
            rms = float(np.sqrt(np.mean([match[2] ** 2 for match in matches])))
            rough_angle_error = (
                _angle_distance(angle, rough_pose.angle_degrees, model.rotational_symmetry_degrees)
                if rough_pose is not None else 0.0
            )
            rough_center_error = (
                float(np.linalg.norm(center - np.asarray(rough_pose.center)))
                if rough_pose is not None else 0.0
            )
            score = (float(len(matches)), -rms, -rough_center_error, -rough_angle_error)
            if best is None or score > best[0]:
                best = (score, working_scale, angle, center, matches)

    if best is None:
        return None

    _, scale, angle, center, matches = best
    for _ in range(3):
        model_indices = [match[0] for match in matches]
        detected_indices = [match[1] for match in matches]
        scale, angle, center = _fit_similarity(
            model.stud_centers_mm[model_indices], detected[detected_indices], fixed_scale,
        )
        predicted = local_to_image(model.stud_centers_mm, tuple(center), angle, scale)
        threshold = max(2.5, STUD_PITCH_MM * scale * 0.24)
        refined_matches = _associate(predicted, detected, threshold)
        if len(refined_matches) < minimum_matches:
            break
        matches = refined_matches

    predicted = local_to_image(model.stud_centers_mm, tuple(center), angle, scale)
    residuals = [np.linalg.norm(predicted[m] - detected[d]) for m, d, _ in matches]
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    matched_model_indices = tuple(match[0] for match in matches)
    matched_detected_indices = tuple(match[1] for match in matches)
    rejected = tuple(sorted(set(range(len(detected))) - set(matched_detected_indices)))
    match_ratio = len(matches) / len(model.stud_centers_mm)
    precision = len(matches) / len(detected)
    residual_score = max(0.0, 1.0 - rms / max(2.5, STUD_PITCH_MM * scale * 0.18))
    geometry_score = 1.0
    if rough_pose is not None:
        angle_error = _angle_distance(angle, rough_pose.angle_degrees, model.rotational_symmetry_degrees)
        geometry_score = max(0.0, 1.0 - angle_error / 35.0)
    confidence = min(
        0.99,
        0.50 * match_ratio + 0.20 * precision + 0.20 * residual_score + 0.10 * geometry_score,
    )
    normalized_angle = normalize_angle(angle)
    if model.rotational_symmetry_degrees == 90:
        normalized_angle %= 90.0
        predicted = local_to_image(model.stud_centers_mm, tuple(center), normalized_angle, scale)

    return ModelFit(
        center=(float(center[0]), float(center[1])),
        angle_degrees=float(normalized_angle),
        scale_px_per_mm=float(scale),
        polygon=ideal_body_polygon(model, tuple(center), normalized_angle, scale),
        predicted_studs=predicted.astype(np.float32),
        matched_detected_indices=matched_detected_indices,
        matched_model_indices=matched_model_indices,
        rejected_detected_indices=rejected,
        reprojection_error_px=rms,
        confidence=confidence,
        expected_studs=len(model.stud_centers_mm),
        rotational_symmetry_degrees=model.rotational_symmetry_degrees,
    )


def estimate_scene_scale(
    fits: list[ModelFit], rectified: bool = False
) -> SceneScaleEstimate:
    strong_scales = np.asarray(
        [fit.scale_px_per_mm for fit in fits if fit.confidence >= 0.70], dtype=np.float64
    )
    minimum_samples = 2 if rectified else 3
    if len(strong_scales) == 0:
        return SceneScaleEstimate(None, 0.0, None, 0, 0, 0, False)
    median = float(np.median(strong_scales))
    deviations = np.abs(strong_scales - median)
    mad = float(np.median(deviations))
    robust_sigma = 1.4826 * mad
    tolerance = max(median * 0.08, 3.0 * robust_sigma)
    inliers = strong_scales[deviations <= tolerance]
    robust_scale = float(np.median(inliers))
    relative_variation = float(1.4826 * np.median(np.abs(inliers - robust_scale)) / robust_scale)
    variation_limit = 0.12 if rectified else 0.06
    trustworthy = len(inliers) >= minimum_samples and relative_variation <= variation_limit
    sample_score = min(1.0, len(inliers) / 5.0)
    consistency_score = max(0.0, 1.0 - relative_variation / max(variation_limit, 1e-6))
    confidence = sample_score * (0.4 + 0.6 * consistency_score)
    return SceneScaleEstimate(
        scale_px_per_mm=robust_scale,
        confidence=float(confidence),
        relative_variation=relative_variation,
        sample_count=int(len(inliers)),
        candidate_count=int(len(strong_scales)),
        outlier_count=int(len(strong_scales) - len(inliers)),
        trustworthy=trustworthy,
    )
