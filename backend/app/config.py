from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env", override=False)


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_vision_model: str = "gpt-5.6"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return Settings().cors_origins
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.6"),
        cors_origins=_parse_origins(os.getenv("BACKEND_CORS_ORIGINS")),
    )
