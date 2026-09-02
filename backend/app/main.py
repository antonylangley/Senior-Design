import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ai import AIConfigurationError, AIRecognitionError, OpenAISketchRecognizer
from app.config import get_settings
from app.models.api import (
    DetectRequest,
    DetectResponse,
    RecognizeRequest,
    ScanRequest,
    ScanResponse,
    TextIntentRequest,
    TextIntentResponse,
)
from app.models.lego import LegoDetectRequest, LegoDetectResponse
from app.services.lego_service import LegoDetectionOptions, process_lego_image
from app.services.scan_service import detect_paper_from_image_data, process_scan_image
from app.services.text_intent import normalize_text_intent
from app.vision.image_io import bytes_to_cv2, cv2_to_bytes, decode_image_data, resize_to_max_side
from app.vision.rectification import RectificationConfig


settings = get_settings()

app = FastAPI(title="Robot Intent Interface", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _recognizer() -> OpenAISketchRecognizer:
    return OpenAISketchRecognizer(settings=get_settings())


@app.get("/health")
def health() -> dict[str, bool | str]:
    configured = bool(get_settings().openai_api_key)
    return {"status": "ok", "openai_configured": configured}


@app.post("/api/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    try:
        detection, width, height = detect_paper_from_image_data(request.image_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    corners = None
    if detection.corners is not None:
        corners = [{"x": float(x), "y": float(y)} for x, y in detection.corners]

    return DetectResponse(
        paper_detected=detection.found,
        confidence=detection.confidence,
        corners=corners,
        image_width=width,
        image_height=height,
        warning=detection.message,
    )


@app.post("/api/scan", response_model=ScanResponse)
def scan(request: ScanRequest) -> ScanResponse:
    try:
        processed = process_scan_image(
            request.image_data,
            use_full_frame_on_failure=request.use_full_frame_on_failure,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recognition = None
    recognition_error = None
    should_recognize = processed.paper_detected or processed.used_full_frame
    if should_recognize:
        try:
            recognition = _recognizer().recognize(processed.ai_image_bytes, processed.ai_mime_type)
        except AIConfigurationError as exc:
            recognition_error = str(exc)
        except AIRecognitionError as exc:
            recognition_error = str(exc)

    return ScanResponse(
        paper_detected=processed.paper_detected,
        used_full_frame=processed.used_full_frame,
        confidence=processed.confidence,
        corners=processed.corners,
        processed_image_data=processed.processed_image_data,
        recognition=recognition,
        recognition_error=recognition_error,
        warning=processed.warning,
    )


@app.post("/api/recognize", response_model=ScanResponse)
def recognize(request: RecognizeRequest) -> ScanResponse:
    try:
        image_bytes, mime_type = decode_image_data(request.image_data, request.mime_type)
        frame = resize_to_max_side(bytes_to_cv2(image_bytes), 1400)
        ai_image_bytes = cv2_to_bytes(frame, "image/png")
        recognition = _recognizer().recognize(ai_image_bytes, "image/png")
    except AIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIRecognitionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScanResponse(
        paper_detected=False,
        used_full_frame=True,
        confidence=0.0,
        processed_image_data=request.image_data,
        recognition=recognition,
        warning=None,
    )


@app.post("/api/text-intent", response_model=TextIntentResponse)
def text_intent(request: TextIntentRequest) -> TextIntentResponse:
    try:
        intent = normalize_text_intent(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TextIntentResponse(intent=intent)


@app.post("/api/lego/detect", response_model=LegoDetectResponse)
def detect_lego_bricks(request: LegoDetectRequest) -> LegoDetectResponse:
    try:
        rectification = None
        if request.rectification is not None:
            if bool(request.rectification.output_width) != bool(request.rectification.output_height):
                raise ValueError("rectification output_width and output_height must be supplied together")
            output_size = None
            if request.rectification.output_width and request.rectification.output_height:
                output_size = (
                    request.rectification.output_width,
                    request.rectification.output_height,
                )
            rectification = RectificationConfig(
                workspace_corners=np.asarray(
                    [[point.x, point.y] for point in request.rectification.workspace_corners],
                    dtype=np.float32,
                ),
                output_size=output_size,
            )
        return process_lego_image(
            request.image_data,
            LegoDetectionOptions(
                grid_rows=request.grid_rows,
                grid_columns=request.grid_columns,
                debug=request.debug,
                rectification=rectification,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
