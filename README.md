# Visual Agentic AI

A multi-agent AI application for research and image understanding. The system uses a supervisor agent to route user requests to specialist agents for research, image description, and object detection.

## Features

- Supervisor-based multi-agent workflow with LangGraph.
- Research agent using Arxiv and Wikipedia.
- Vision agent for image description and object detection.
- Gemini 2.5 Flash-Lite as the LLM backend.
- YOLO-based local object detection and counting.
- Supports local image paths and image URLs.
- Basic local web UI.

## Project Structure

```text
code/
  assets/                 # Put local images here
  examples/               # Example scripts
  models/                 # Local model weights
  scripts/                # Setup and helper scripts
  src/                    # Application source code
    agents/               # Research and vision agents
    tools/                # Arxiv, Wikipedia, image, and YOLO tools
    utils/                # Message and printing helpers
    cli.py                # CLI entry point
    config.py             # Environment/config loading
    llm.py                # Gemini model setup
    supervisor.py         # Supervisor graph
    web_app.py            # Local web UI
  .env                    # Local secrets/config, ignored by git
  .env.example            # Example environment variables
  .gitignore
  README.md
  requirements.txt
```

## Requirements

- Python 3.10 or newer
- PowerShell on Windows
- Gemini API key from Google AI Studio

Create a Gemini API key here:

https://aistudio.google.com/app/apikey

## Installation

Open PowerShell in the project folder:

```powershell
cd "D:\AIO\Project\Visual Agentic AI\code"
```

Run the setup script:

```powershell
.\scripts\setup_venv.ps1
```

If PowerShell blocks the script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_venv.ps1
```

## Configuration

Create or edit `.env` in the project root:

```env
GOOGLE_API_KEY=your_google_ai_studio_key_here
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=Visual-Agent
GEMINI_MODEL=gemini-2.5-flash-lite
YOLO_MODEL=models/yolo11n.pt
```

YOLO model options:

- `models/yolo11n.pt`: lightweight and recommended for local testing.
- `models/yolo11x.pt`: larger and more accurate, but slower.

## Web UI

Start the local web interface:

```powershell
.\.venv\Scripts\python.exe -m src.web_app
```

Open this URL in your browser:

```text
http://127.0.0.1:7860
```

The page lets you enter a prompt and an optional image path or URL. Gemini is only called after you press Submit.

## CLI Usage

### Ask a research question

```powershell
.\.venv\Scripts\python.exe -m src.cli "What is Rotary Positional Encoding? Find related papers."
```

### Describe an image from a URL

```powershell
.\.venv\Scripts\python.exe -m src.cli "Describe this image: https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"
```

### Count objects in a local image

Put your image in `assets/`, then run:

```powershell
.\.venv\Scripts\python.exe -m src.cli "How many dogs are in image: ./assets/multi-dogs.jpg"
```

### Ask a mixed vision and research question

```powershell
.\.venv\Scripts\python.exe -m src.cli "What concept is shown in image: ./assets/rope.png? Find research papers about it."
```

## Examples

```powershell
.\.venv\Scripts\python.exe examples\run_research.py
.\.venv\Scripts\python.exe examples\run_vision.py
.\.venv\Scripts\python.exe examples\run_supervisor.py
```


## Project Smoke Test

All project backend checks are in one script:

```powershell
.\.venv\Scripts\python.exe scripts\test_project.py
```

This tests:

- Gemini API
- YOLO local inference
- LangSmith connection and trace, if `LANGSMITH_API_KEY` is set

Run without spending a Gemini request:

```powershell
.\.venv\Scripts\python.exe scripts\test_project.py --skip-gemini
```

Run only Gemini + config, without YOLO or LangSmith:

```powershell
.\.venv\Scripts\python.exe scripts\test_project.py --skip-yolo --skip-langsmith
```

Run YOLO only, without Gemini or LangSmith:

```powershell
.\.venv\Scripts\python.exe scripts\test_project.py --skip-gemini --skip-langsmith
```

Set LangSmith values in `.env` if you want LangSmith traces:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=Visual-Agent
```

If LangSmith is enabled and Gemini is not skipped, the Gemini smoke call should appear in your LangSmith project.
## Development Checks

These checks do not call the Gemini API:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\cli.py src\web_app.py
.\.venv\Scripts\python.exe -c "from src.supervisor import build_supervisor; build_supervisor(); print('graph ok')"
```

## Notes

- Gemini API usage may be rate-limited depending on your plan.
- Arxiv and Wikipedia tools do not require API keys.
- Local image paths are resolved relative to the project root.
- YOLO model files are ignored by git.

## Troubleshooting

### `GOOGLE_API_KEY is missing`

Check that `.env` exists and contains:

```env
GOOGLE_API_KEY=your_real_key_here
```

### PowerShell cannot run `setup_venv.ps1`

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_venv.ps1
```

### Image file not found

Use a path relative to the project root:

```text
./assets/image_name.jpg
```

### YOLO is slow

Use the lighter model:

```env
YOLO_MODEL=models/yolo11n.pt
```



