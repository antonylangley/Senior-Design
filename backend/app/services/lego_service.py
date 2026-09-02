from dataclasses import dataclass

import cv2
import numpy as np

from app.models.lego import (
    BrickDimensions, GridDefinition, GridPosition, ImageSize, LegoBrick,
    LegoDebugImages, LegoDetectResponse, NumericColor, RectificationStatus,
)
from app.models.api import Point
from app.vision.image_io import bytes_to_cv2, cv2_to_data_url, decode_image_data, resize_to_max_side
from app.vision.lego import (
    calculate_brick_pose, classify_brick_color, detect_studs, infer_brick_dimensions,
    map_point_to_grid, normalize_point, segment_bricks,
)
from app.vision.rectification import RectificationConfig, RectificationResult, rectify_image


@dataclass(frozen=True)
class LegoDetectionOptions:
    grid_rows: int = 10
    grid_columns: int = 10
    debug: bool = False
    rectification: RectificationConfig | None = None


def _debug_images(
    frame: np.ndarray, mask: np.ndarray, bricks: list[LegoBrick], rectification: RectificationResult
) -> LegoDebugImages:
    components = frame.copy()
    studs = frame.copy()
    for brick in bricks:
        polygon = np.asarray(brick.bounding_polygon, dtype=np.int32)
        cv2.polylines(components, [polygon], True, (0, 255, 0), 2)
        cv2.circle(components, (round(brick.center_px.x), round(brick.center_px.y)), 5, (0, 0, 255), -1)
        for x, y in brick.stud_centers_px:
            cv2.circle(studs, (round(x), round(y)), 5, (255, 0, 255), 2)
    return LegoDebugImages(
        rectified_view=cv2_to_data_url(frame, "image/png") if rectification.active else None,
        segmentation_mask=cv2_to_data_url(mask, "image/png"),
        components=cv2_to_data_url(components, "image/png"),
        studs=cv2_to_data_url(studs, "image/png"),
    )


def process_lego_image(image_data: str, options: LegoDetectionOptions) -> LegoDetectResponse:
    image_bytes, _ = decode_image_data(image_data)
    raw_frame = resize_to_max_side(bytes_to_cv2(image_bytes), 1800)
    rectification = rectify_image(raw_frame, options.rectification)
    frame = rectification.image
    height, width = frame.shape[:2]
    mask, instances = segment_bricks(frame)
    bricks: list[LegoBrick] = []
    for index, instance in enumerate(instances, start=1):
        pose = calculate_brick_pose(instance.contour)
        stud_centers = detect_studs(frame, instance, pose)
        dimension_inference = infer_brick_dimensions(stud_centers, pose)
        color, hsv, lab = classify_brick_color(frame, instance.mask)
        normalized = normalize_point(*pose.center, width, height)
        row, column = map_point_to_grid(
            *pose.center, width, height, options.grid_rows, options.grid_columns
        )
        fill = cv2.contourArea(instance.contour) / max(pose.size[0] * pose.size[1], 1)
        confidence = min(
            0.95,
            0.40 + 0.30 * fill + 0.25 * dimension_inference.confidence,
        )
        bricks.append(LegoBrick(
            id=index, color=color,
            dimensions=BrickDimensions(
                studs_x=dimension_inference.dimensions[0],
                studs_y=dimension_inference.dimensions[1],
            ) if dimension_inference.dimensions else None,
            stud_count=len(stud_centers),
            stud_centers_px=[[round(x, 2), round(y, 2)] for x, y in stud_centers],
            center_px=Point(x=round(pose.center[0], 2), y=round(pose.center[1], 2)),
            center_normalized=Point(x=round(normalized[0], 6), y=round(normalized[1], 6)),
            angle_degrees=round(pose.angle_degrees, 2),
            grid_position=GridPosition(row=row, column=column),
            bounding_polygon=[[round(float(x), 2), round(float(y), 2)] for x, y in pose.polygon],
            confidence=round(confidence, 3),
            dimension_confidence=round(dimension_inference.confidence, 3),
            dimension_source=dimension_inference.source,
            representative_color=NumericColor(hsv=hsv, lab=lab),
        ))
    warning = None if bricks else "No separated brick regions were detected. Try a simpler, contrasting background."
    return LegoDetectResponse(
        image=ImageSize(width=width, height=height),
        processed_image_data=cv2_to_data_url(frame, "image/png"),
        grid=GridDefinition(rows=options.grid_rows, columns=options.grid_columns),
        bricks=bricks,
        rectification=RectificationStatus(active=rectification.active, method=rectification.method),
        debug=_debug_images(frame, mask, bricks, rectification) if options.debug else None,
        warning=warning,
    )
