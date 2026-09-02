from pydantic import BaseModel, Field

from app.models.intent import RobotIntent, SketchRecognitionResult


class ImagePayload(BaseModel):
    image_data: str = Field(min_length=1)


class DetectRequest(ImagePayload):
    pass


class ScanRequest(ImagePayload):
    use_full_frame_on_failure: bool = False


class RecognizeRequest(ImagePayload):
    mime_type: str | None = None


class TextIntentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=240)


class Point(BaseModel):
    x: float
    y: float


class DetectResponse(BaseModel):
    paper_detected: bool
    confidence: float
    corners: list[Point] | None = None
    image_width: int
    image_height: int
    warning: str | None = None


class ScanResponse(BaseModel):
    paper_detected: bool
    used_full_frame: bool
    confidence: float
    corners: list[Point] | None = None
    processed_image_data: str
    recognition: SketchRecognitionResult | None = None
    recognition_error: str | None = None
    warning: str | None = None


class TextIntentResponse(BaseModel):
    intent: RobotIntent
