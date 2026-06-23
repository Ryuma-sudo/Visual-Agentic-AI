from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT, settings
from src.llm import build_llm
from src.tools.object_detector import detect_and_count_objects


TEST_IMAGE = PROJECT_ROOT / "assets" / "model_smoke_test.png"


def mask_key(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"


def ensure_test_image() -> Path:
    TEST_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    if TEST_IMAGE.exists():
        return TEST_IMAGE

    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 50, 240, 170), outline="black", width=5)
    draw.text((92, 92), "TEST", fill="black")
    image.save(TEST_IMAGE)
    return TEST_IMAGE


def print_config() -> None:
    print("[config]")
    print(f"  project_root: {CONFIG_PROJECT_ROOT}")
    print(f"  gemini_model: {settings.gemini_model}")
    print(f"  yolo_model: {settings.yolo_model}")
    print(f"  google_api_key: {mask_key(settings.google_api_key)}")
    print(f"  langsmith_tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")
    print(f"  langsmith_project: {os.getenv('LANGSMITH_PROJECT', 'Visual-Agent')}")
    print(f"  langsmith_api_key: {mask_key(os.getenv('LANGSMITH_API_KEY'))}")


def setup_langsmith_env() -> tuple[str, str] | None:
    api_key = os.getenv("LANGSMITH_API_KEY") or ""
    project = os.getenv("LANGSMITH_PROJECT") or "Visual-Agent"
    if not api_key:
        return None

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    return api_key, project


def test_langsmith() -> bool:
    print("[langsmith]")
    env = setup_langsmith_env()
    if env is None:
        print("  skipped: LANGSMITH_API_KEY is missing")
        return False

    api_key, project = env
    from langsmith import Client, traceable

    client = Client(api_key=api_key)
    projects = list(client.list_projects(limit=1))
    print("  api_connection: ok")
    print(f"  visible_projects_sample: {len(projects)}")
    print(f"  target_project: {project}")

    @traceable(name="visual-agent-langsmith-smoke-test")
    def smoke_trace() -> str:
        return "langsmith-ok"

    result = smoke_trace()
    if result != "langsmith-ok":
        raise RuntimeError("Unexpected LangSmith smoke trace result")
    print(f"  trace_result: {result}")
    print("  status: ok")
    return True


def test_gemini() -> None:
    print("[gemini]")
    llm = build_llm()
    response = llm.invoke("Reply with exactly: gemini-ok")
    content = str(response.content).strip()
    print(f"  response: {content}")
    if "gemini-ok" not in content.lower():
        raise RuntimeError("Gemini responded, but not with the expected smoke-test text.")
    print("  status: ok")


def test_yolo() -> None:
    print("[yolo]")
    image_path = ensure_test_image()
    result = detect_and_count_objects.invoke({"text": str(image_path)})
    print(f"  image: {image_path}")
    print(f"  result: {result}")
    if isinstance(result, str) and result.startswith("YOLO inference failed"):
        raise RuntimeError(result)
    print("  status: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Visual Agentic AI project backends.")
    parser.add_argument("--skip-gemini", action="store_true", help="Do not call Gemini API.")
    parser.add_argument("--skip-yolo", action="store_true", help="Do not run YOLO inference.")
    parser.add_argument("--skip-langsmith", action="store_true", help="Do not call LangSmith API or create a trace.")
    args = parser.parse_args()

    print_config()

    langsmith_enabled = False
    if args.skip_langsmith:
        print("[langsmith]")
        print("  skipped")
    else:
        langsmith_enabled = test_langsmith()

    if args.skip_gemini:
        print("[gemini]")
        print("  skipped")
    else:
        test_gemini()
        if langsmith_enabled:
            print("  note: Gemini call should appear in LangSmith because tracing is enabled for this test process")

    if args.skip_yolo:
        print("[yolo]")
        print("  skipped")
    else:
        test_yolo()

    print("[done]")
    print("  all selected checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
