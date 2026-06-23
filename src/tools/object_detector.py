from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, settings
from src.tools.image_utils import extract_image_path_or_url, is_url, resolve_local_path


_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        model_path = Path(settings.yolo_model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        _yolo_model = YOLO(str(model_path))
    return _yolo_model


def normalize_image_source(text: str) -> str:
    raw = text.strip().strip('"').strip("'")

    # Keep complete Windows/local paths intact, especially paths containing spaces.
    raw_path = Path(raw)
    if raw_path.is_absolute() or raw_path.exists():
        return str(raw_path)

    image_path_or_url = extract_image_path_or_url(raw)
    if is_url(image_path_or_url):
        return image_path_or_url
    return str(resolve_local_path(image_path_or_url))


class ObjectDetectingAndCountingInput(BaseModel):
    text: str = Field(description="Path or URL to an image.")


@tool(
    "detect_and_count_objects",
    description=(
        "Detect and count objects within an image. Returns a dictionary with a "
        "counting dictionary and a list of detections with class, confidence, "
        "and bbox in (x1, y1, x2, y2) format."
    ),
    args_schema=ObjectDetectingAndCountingInput,
)
def detect_and_count_objects(text: Annotated[str, "Path or URL to an image"]):
    source = normalize_image_source(text)

    try:
        results = get_yolo_model().predict(source, verbose=False)
    except Exception as exc:
        return f"YOLO inference failed: {exc}"

    detections: list[dict] = []
    counting: dict[str, int] = {}
    for result in results:
        boxes = result.boxes
        class_names = result.names
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = round(float(box.conf[0]), 4)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )
            counting[class_name] = counting.get(class_name, 0) + 1

    return {"counting": counting, "detections": detections}
