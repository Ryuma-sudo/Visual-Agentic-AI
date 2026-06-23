from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:
    load_env_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    yolo_model: str = os.getenv("YOLO_MODEL", "models/yolo11n.pt")
    wikipedia_user_agent: str = os.getenv(
        "WIKIPEDIA_USER_AGENT",
        "AIO-VisualAgent/1.0 (contact: local-project@example.com)",
    )


settings = Settings()
