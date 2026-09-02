from dataclasses import dataclass

import cv2
import numpy as np

from app.models.api import Point
from app.models.lego import (
    BrickDimensions, GridDefinition, GridPosition, ImageSize, LegoBrick,
    LegoDebugImages, LegoDetectResponse, ModelFitDiagnostics, NumericColor,
    RawPose, RectificationStatus, SceneScaleDiagnostics,
)
from app.vision.image_io import bytes_to_cv2, cv2_to_data_url, decode_image_data, resize_to_max_side
from app.vision.lego import (
    BrickPose, DimensionInference, SegmentedBrick, calculate_brick_pose,
    classify_brick_color, detect_studs, infer_brick_dimensions,
    map_point_to_grid, normalize_point, segment_bricks,
)
from app.vision.lego_model import (
    ModelFit, SceneScaleEstimate, canonical_brick_model, estimate_scene_scale,
    fit_canonical_model,
)
from app.vision.rectification import RectificationConfig, RectificationResult, rectify_image


@dataclass(frozen=True)
class LegoDetectionOptions:
    grid_rows: int = 10
    grid_columns: int = 10
    debug: bool = False
    rectification: RectificationConfig | None = None


@dataclass(frozen=True)
class _PreliminaryBrick:
    instance: SegmentedBrick
    raw_pose: BrickPose
    stud_centers: list[tuple[float, float]]
    dimension_inference: DimensionInference
    color: str
    hsv: list[float]
    lab: list[float]
    fit: ModelFit | None


def _fit_is_strong(fit: ModelFit | None, raw_pose: BrickPose) -> bool:
    if fit is None or fit.confidence < 0.65:
        return False
    center_shift = np.linalg.norm(np.asarray(fit.center) - np.asarray(raw_pose.center))
    if center_shift > max(8.0, min(raw_pose.size) * 0.30):
        return False
    return fit.reprojection_error_px <= max(3.0, fit.scale_px_per_mm * 1.6)


def _points(points: np.ndarray | list[tuple[float, float]]) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def _model_diagnostics(
    fit: ModelFit | None, stud_centers: list[tuple[float, float]]
) -> ModelFitDiagnostics | None:
    if fit is None:
        return None
    detected = np.asarray(stud_centers, dtype=np.float64)
    matched = detected[list(fit.matched_detected_indices)]
    rejected = detected[list(fit.rejected_detected_indices)] if fit.rejected_detected_indices else []
    return ModelFitDiagnostics(
        matched_studs=fit.matched_studs,
        expected_studs=fit.expected_studs,
        detected_candidates=len(stud_centers),
        rejected_studs=len(fit.rejected_detected_indices),
        reprojection_error_px=round(fit.reprojection_error_px, 3),
        scale_px_per_mm=round(fit.scale_px_per_mm, 5),
        predicted_stud_centers_px=_points(fit.predicted_studs),
        matched_stud_centers_px=_points(matched),
        rejected_stud_centers_px=_points(rejected),
    )


def _debug_images(
    frame: np.ndarray, mask: np.ndarray, bricks: list[LegoBrick],
    rectification: RectificationResult,
) -> LegoDebugImages:
    components = frame.copy()
    studs = frame.copy()
    refinement = frame.copy()
    for brick in bricks:
        raw_polygon = np.asarray(brick.raw_pose.bounding_polygon, dtype=np.int32)
        final_polygon = np.asarray(brick.bounding_polygon, dtype=np.int32)
        cv2.polylines(components, [raw_polygon], True, (255, 200, 0), 2)
        cv2.polylines(components, [final_polygon], True, (0, 255, 0), 2)
        cv2.polylines(refinement, [raw_polygon], True, (255, 200, 0), 2)
        cv2.polylines(refinement, [final_polygon], True, (0, 255, 0), 2)
        raw_center = (round(brick.raw_pose.center_px.x), round(brick.raw_pose.center_px.y))
        final_center = (round(brick.center_px.x), round(brick.center_px.y))
        cv2.circle(refinement, raw_center, 5, (255, 200, 0), -1)
        cv2.circle(refinement, final_center, 5, (0, 0, 255), -1)
        for x, y in brick.stud_centers_px:
            cv2.circle(studs, (round(x), round(y)), 5, (255, 0, 255), 2)
        if brick.model_fit is not None:
            for x, y in brick.model_fit.predicted_stud_centers_px:
                cv2.circle(refinement, (round(x), round(y)), 5, (0, 255, 255), 1)
            for x, y in brick.model_fit.matched_stud_centers_px:
                cv2.circle(refinement, (round(x), round(y)), 4, (0, 255, 0), -1)
            for x, y in brick.model_fit.rejected_stud_centers_px:
                cv2.drawMarker(
                    refinement, (round(x), round(y)), (0, 0, 255),
                    cv2.MARKER_TILTED_CROSS, 12, 2,
                )
    return LegoDebugImages(
        rectified_view=cv2_to_data_url(frame, "image/png") if rectification.active else None,
        segmentation_mask=cv2_to_data_url(mask, "image/png"),
        components=cv2_to_data_url(components, "image/png"),
        studs=cv2_to_data_url(studs, "image/png"),
        pose_refinement=cv2_to_data_url(refinement, "image/png"),
    )


def _preliminary_bricks(frame: np.ndarray, instances: list[SegmentedBrick]) -> list[_PreliminaryBrick]:
    preliminary: list[_PreliminaryBrick] = []
    for instance in instances:
        raw_pose = calculate_brick_pose(instance.contour)
        stud_centers = detect_studs(frame, instance, raw_pose)
        dimensions = infer_brick_dimensions(stud_centers, raw_pose)
        color, hsv, lab = classify_brick_color(frame, instance.mask)
        fit = (
            fit_canonical_model(dimensions.dimensions, stud_centers, raw_pose)
            if dimensions.dimensions is not None else None
        )
        preliminary.append(_PreliminaryBrick(
            instance=instance, raw_pose=raw_pose, stud_centers=stud_centers,
            dimension_inference=dimensions, color=color, hsv=hsv, lab=lab, fit=fit,
        ))
    return preliminary


def _refine_with_scene_scale(
    bricks: list[_PreliminaryBrick], scene_scale: SceneScaleEstimate
) -> list[ModelFit | None]:
    refined: list[ModelFit | None] = []
    for brick in bricks:
        fit = brick.fit
        if scene_scale.trustworthy and brick.dimension_inference.dimensions is not None:
            constrained = fit_canonical_model(
                brick.dimension_inference.dimensions, brick.stud_centers,
                brick.raw_pose, fixed_scale=scene_scale.scale_px_per_mm,
            )
            if constrained is not None and (fit is None or constrained.confidence >= fit.confidence - 0.08):
                fit = constrained
        refined.append(fit)
    return refined


def process_lego_image(image_data: str, options: LegoDetectionOptions) -> LegoDetectResponse:
    image_bytes, _ = decode_image_data(image_data)
    raw_frame = resize_to_max_side(bytes_to_cv2(image_bytes), 1800)
    rectification = rectify_image(raw_frame, options.rectification)
    frame = rectification.image
    height, width = frame.shape[:2]
    mask, instances = segment_bricks(frame)
    preliminary = _preliminary_bricks(frame, instances)
    scene_scale = estimate_scene_scale(
        [brick.fit for brick in preliminary if brick.fit is not None],
        rectified=rectification.active,
    )
    refined_fits = _refine_with_scene_scale(preliminary, scene_scale)

    bricks: list[LegoBrick] = []
    for index, (brick, fit) in enumerate(zip(preliminary, refined_fits), start=1):
        strong_fit = _fit_is_strong(fit, brick.raw_pose)
        center = fit.center if strong_fit and fit is not None else brick.raw_pose.center
        angle = fit.angle_degrees if strong_fit and fit is not None else brick.raw_pose.angle_degrees
        polygon = fit.polygon if strong_fit and fit is not None else brick.raw_pose.polygon
        normalized = normalize_point(*center, width, height)
        row, column = map_point_to_grid(*center, width, height, options.grid_rows, options.grid_columns)
        fill = cv2.contourArea(brick.instance.contour) / max(
            brick.raw_pose.size[0] * brick.raw_pose.size[1], 1
        )
        pose_confidence = (
            fit.confidence if strong_fit and fit is not None else min(0.60, 0.35 + 0.25 * fill)
        )
        confidence = min(
            0.96, 0.32 + 0.23 * fill + 0.20 * brick.dimension_inference.confidence
            + 0.25 * pose_confidence,
        )
        dimensions = brick.dimension_inference.dimensions
        symmetry = canonical_brick_model(dimensions).rotational_symmetry_degrees if dimensions else 180
        bricks.append(LegoBrick(
            id=index,
            color=brick.color,
            dimensions=BrickDimensions(studs_x=dimensions[0], studs_y=dimensions[1]) if dimensions else None,
            stud_count=len(brick.stud_centers),
            stud_centers_px=_points(brick.stud_centers),
            center_px=Point(x=round(center[0], 2), y=round(center[1], 2)),
            center_normalized=Point(x=round(normalized[0], 6), y=round(normalized[1], 6)),
            angle_degrees=round(angle, 2),
            rotational_symmetry_degrees=symmetry,
            pose_source="lego_model_fit" if strong_fit else "contour_fallback",
            pose_confidence=round(pose_confidence, 3),
            model_fit=_model_diagnostics(fit, brick.stud_centers),
            raw_pose=RawPose(
                center_px=Point(
                    x=round(brick.raw_pose.center[0], 2), y=round(brick.raw_pose.center[1], 2)
                ),
                angle_degrees=round(brick.raw_pose.angle_degrees, 2),
                bounding_polygon=_points(brick.raw_pose.polygon),
            ),
            grid_position=GridPosition(row=row, column=column),
            bounding_polygon=_points(polygon),
            confidence=round(confidence, 3),
            dimension_confidence=round(brick.dimension_inference.confidence, 3),
            dimension_source=brick.dimension_inference.source,
            representative_color=NumericColor(hsv=brick.hsv, lab=brick.lab),
        ))

    warning = None if bricks else "No separated brick regions were detected. Try a simpler, contrasting background."
    return LegoDetectResponse(
        image=ImageSize(width=width, height=height),
        processed_image_data=cv2_to_data_url(frame, "image/png"),
        grid=GridDefinition(rows=options.grid_rows, columns=options.grid_columns),
        bricks=bricks,
        rectification=RectificationStatus(active=rectification.active, method=rectification.method),
        scene_scale=SceneScaleDiagnostics(
            scale_px_per_mm=round(scene_scale.scale_px_per_mm, 5) if scene_scale.scale_px_per_mm else None,
            confidence=round(scene_scale.confidence, 3),
            relative_variation=(
                round(scene_scale.relative_variation, 5)
                if scene_scale.relative_variation is not None else None
            ),
            sample_count=scene_scale.sample_count,
            candidate_count=scene_scale.candidate_count,
            outlier_count=scene_scale.outlier_count,
            trustworthy=scene_scale.trustworthy,
        ),
        debug=_debug_images(frame, mask, bricks, rectification) if options.debug else None,
        warning=warning,
    )
