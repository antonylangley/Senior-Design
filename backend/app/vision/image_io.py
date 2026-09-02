import base64
import re

import cv2
import numpy as np


DATA_URL_RE = re.compile(r"^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class ImageDecodeError(ValueError):
    pass


def decode_image_data(image_data: str, fallback_mime_type: str | None = None) -> tuple[bytes, str]:
    raw = image_data.strip()
    match = DATA_URL_RE.match(raw)
    if match:
        mime_type = match.group(1).lower()
        encoded = match.group(2)
    else:
        mime_type = (fallback_mime_type or "image/jpeg").lower()
        encoded = raw

    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ImageDecodeError(f"unsupported image MIME type: {mime_type}")

    try:
        return base64.b64decode(encoded, validate=True), mime_type
    except Exception as exc:
        raise ImageDecodeError("image data is not valid base64") from exc


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ImageDecodeError("image data could not be decoded")
    return frame


def resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def cv2_to_bytes(image: np.ndarray, mime_type: str = "image/png") -> bytes:
    if mime_type == "image/jpeg":
        encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        ext = ".jpg"
    elif mime_type == "image/webp":
        encode_args = [int(cv2.IMWRITE_WEBP_QUALITY), 92]
        ext = ".webp"
    else:
        encode_args = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        ext = ".png"
        mime_type = "image/png"

    ok, encoded = cv2.imencode(ext, image, encode_args)
    if not ok:
        raise ImageDecodeError("processed image could not be encoded")
    return encoded.tobytes()


def cv2_to_data_url(image: np.ndarray, mime_type: str = "image/png") -> str:
    image_bytes = cv2_to_bytes(image, mime_type)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
