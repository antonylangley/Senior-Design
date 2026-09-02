from pydantic import BaseModel, Field

from app.models.api import ImagePayload, Point


class ManualRectificationConfig(BaseModel):
    workspace_corners: list[Point] = Field(min_length=4, max_length=4)
    output_width: int | None = Field(default=None, ge=1, le=8000)
    output_height: int | None = Field(default=None, ge=1, le=8000)


class LegoDetectRequest(ImagePayload):
    grid_rows: int = Field(default=10, ge=1, le=50)
    grid_columns: int = Field(default=10, ge=1, le=50)
    debug: bool = False
    rectification: ManualRectificationConfig | None = None


class ImageSize(BaseModel):
    width: int
    height: int


class GridDefinition(BaseModel):
    rows: int
    columns: int


class GridPosition(BaseModel):
    row: int
    column: int


class BrickDimensions(BaseModel):
    studs_x: int
    studs_y: int


class NumericColor(BaseModel):
    hsv: list[float]
    lab: list[float]


class RawPose(BaseModel):
    center_px: Point
    angle_degrees: float
    bounding_polygon: list[list[float]]


class ModelFitDiagnostics(BaseModel):
    matched_studs: int
    expected_studs: int
    detected_candidates: int
    rejected_studs: int
    reprojection_error_px: float
    scale_px_per_mm: float
    predicted_stud_centers_px: list[list[float]]
    matched_stud_centers_px: list[list[float]]
    rejected_stud_centers_px: list[list[float]]


class LegoBrick(BaseModel):
    id: int
    color: str
    dimensions: BrickDimensions | None = None
    stud_count: int
    stud_centers_px: list[list[float]]
    center_px: Point
    center_normalized: Point
    angle_degrees: float
    rotational_symmetry_degrees: int
    pose_source: str
    pose_confidence: float = Field(ge=0, le=1)
    model_fit: ModelFitDiagnostics | None = None
    raw_pose: RawPose
    grid_position: GridPosition
    bounding_polygon: list[list[float]]
    confidence: float = Field(ge=0, le=1)
    dimension_confidence: float = Field(ge=0, le=1)
    dimension_source: str
    representative_color: NumericColor


class LegoDebugImages(BaseModel):
    rectified_view: str | None = None
    segmentation_mask: str
    components: str
    studs: str
    pose_refinement: str


class RectificationStatus(BaseModel):
    active: bool
    method: str


class SceneScaleDiagnostics(BaseModel):
    scale_px_per_mm: float | None = None
    confidence: float = Field(ge=0, le=1)
    relative_variation: float | None = None
    sample_count: int
    candidate_count: int
    outlier_count: int
    trustworthy: bool


class LegoDetectResponse(BaseModel):
    image: ImageSize
    processed_image_data: str
    grid: GridDefinition
    bricks: list[LegoBrick]
    rectification: RectificationStatus
    scene_scale: SceneScaleDiagnostics
    debug: LegoDebugImages | None = None
    warning: str | None = None
