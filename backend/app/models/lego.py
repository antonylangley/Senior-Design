from pydantic import BaseModel, Field

from app.models.api import ImagePayload, Point


class LegoDetectRequest(ImagePayload):
    grid_rows: int = Field(default=10, ge=1, le=50)
    grid_columns: int = Field(default=10, ge=1, le=50)
    debug: bool = False


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


class LegoBrick(BaseModel):
    id: int
    color: str
    dimensions: BrickDimensions | None = None
    stud_count: int
    stud_centers_px: list[list[float]]
    center_px: Point
    center_normalized: Point
    angle_degrees: float
    grid_position: GridPosition
    bounding_polygon: list[list[float]]
    confidence: float = Field(ge=0, le=1)
    representative_color: NumericColor


class LegoDebugImages(BaseModel):
    segmentation_mask: str
    components: str
    studs: str


class LegoDetectResponse(BaseModel):
    image: ImageSize
    grid: GridDefinition
    bricks: list[LegoBrick]
    debug: LegoDebugImages | None = None
    warning: str | None = None
