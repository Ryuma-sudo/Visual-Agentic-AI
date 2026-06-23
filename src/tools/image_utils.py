from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.config import PROJECT_ROOT


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text.strip().strip('"').strip("'")).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def extract_image_path_or_url(text: str) -> str:
    """Extract an image URL or local path without spending an LLM request."""
    text = text.strip()

    url_match = re.search(r"https?://[^\s\"')>]+", text)
    if url_match:
        return url_match.group(0).rstrip(".,;")

    labeled_match = re.search(
        r"(?:image|img|path|file|image_path)\s*[:=]\s*([\"']?)(.+?)\1(?:\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if labeled_match:
        candidate = labeled_match.group(2).strip().rstrip(".,;")
        if Path(candidate).suffix.lower() in IMAGE_EXTENSIONS:
            return candidate

    for token in re.findall(r"[^\s\"']+", text):
        cleaned = token.strip().rstrip(".,;)")
        if Path(cleaned).suffix.lower() in IMAGE_EXTENSIONS:
            return cleaned

    return text


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def encode_image(image_path_or_url: str, get_mime_type: bool = False):
    def fail():
        return (None, None) if get_mime_type else None

    image_path_or_url = extract_image_path_or_url(image_path_or_url)

    if is_url(image_path_or_url):
        try:
            response = requests.get(image_path_or_url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            return fail()

        mime_type = response.headers.get("content-type", "").split(";")[0]
        if not mime_type.startswith("image/"):
            guessed, _ = mimetypes.guess_type(image_path_or_url)
            mime_type = guessed or mime_type
        if not mime_type or not mime_type.startswith("image/"):
            return fail()

        b64 = base64.b64encode(response.content).decode("utf-8")
        return (b64, mime_type) if get_mime_type else b64

    image_path = resolve_local_path(image_path_or_url)
    if not image_path.exists() or not image_path.is_file():
        return fail()

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        return fail()

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return (b64, mime_type) if get_mime_type else b64
