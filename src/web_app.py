from __future__ import annotations

import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.supervisor import build_supervisor
from src.tools.object_detector import detect_and_count_objects
from src.utils.messages import get_final_text

HOST = "127.0.0.1"
PORT = int(os.getenv("VISUAL_AGENT_PORT", "7860"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SUPERVISOR = None




def is_object_count_request(prompt: str) -> bool:
    lowered = prompt.lower()
    count_words = ("how many", "count", "detect", "number of")
    object_words = ("dog", "dogs", "cat", "cats", "person", "people", "object", "objects")
    return any(word in lowered for word in count_words) and any(word in lowered for word in object_words)


def format_detection_answer(result) -> str:
    if isinstance(result, str):
        return result
    counting = result.get("counting", {}) if isinstance(result, dict) else {}
    detections = result.get("detections", []) if isinstance(result, dict) else []
    if not counting:
        return "No supported objects were detected in the image."

    lines = ["Detected objects:"]
    for class_name, count in sorted(counting.items()):
        lines.append(f"- {class_name}: {count}")
    if detections:
        lines.append("")
        lines.append("Detections:")
        for index, item in enumerate(detections, start=1):
            label = item.get("class", "object")
            confidence = item.get("confidence", 0)
            bbox = item.get("bbox", [])
            lines.append(f"{index}. {label} confidence={confidence} bbox={bbox}")
    return "\n".join(lines)

def get_supervisor():
    global _SUPERVISOR
    if _SUPERVISOR is None:
        _SUPERVISOR = build_supervisor()
    return _SUPERVISOR


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Visual Agentic AI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202f;
      --muted: #647084;
      --line: #d9dee8;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --danger: #b42318;
      --success: #067647;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell {
      width: min(980px, calc(100vw - 32px));
      margin: 32px auto;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .subtle {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    textarea {
      min-height: 170px;
      resize: vertical;
      line-height: 1.45;
    }
    textarea:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37, 99, 235, .12); }
    .field { margin-bottom: 14px; }
    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      padding: 11px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .6; cursor: wait; }
    .secondary {
      background: #eef2ff;
      color: #233876;
    }
    .secondary:hover { background: #e0e7ff; }
    .output {
      min-height: 250px;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .hint-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .hint-list button {
      width: 100%;
      text-align: left;
      background: #f8fafc;
      color: var(--text);
      border: 1px solid var(--line);
      font-weight: 600;
      padding: 10px;
    }
    .hint-list button:hover { background: #eef4ff; }
    .note {
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .error { color: var(--danger); }
    .ok { color: var(--success); }
    @media (max-width: 820px) {
      .grid { grid-template-columns: 1fr; }
      header { align-items: start; flex-direction: column; }
      .status { white-space: normal; }
    }
  </style>
</head>
<body>
  <main class=\"shell\">
    <header>
      <div>
        <h1>Visual Agentic AI</h1>
        <p class=\"subtle\">Ask research questions, describe images, or count objects with a supervisor agent.</p>
      </div>
      <div id=\"status\" class=\"status\">Idle</div>
    </header>

    <section class=\"grid\">
      <form id=\"askForm\" class=\"panel\">
        <div class=\"field\">
          <label for=\"prompt\">Prompt</label>
          <textarea id=\"prompt\" name=\"prompt\" placeholder=\"What is Rotary Positional Encoding? Find related papers.\"></textarea>
        </div>
        <div class=\"field\">
          <label for=\"image\">Optional image path or URL</label>
          <input id=\"image\" name=\"image\" placeholder=\"./assets/image.jpg or https://example.com/image.png\" />
        </div>
        <div class=\"actions\">
          <button id=\"submitBtn\" type=\"submit\">Submit</button>
          <button class=\"secondary\" id=\"clearBtn\" type=\"button\">Clear</button>
        </div>
        <p class=\"note\">Object counting uses local YOLO directly. Other agent tasks call Gemini after you press Submit.</p>
      </form>

      <aside class=\"panel\">
        <label>Examples</label>
        <ul class=\"hint-list\">
          <li><button type=\"button\" data-prompt=\"What is Rotary Positional Encoding? Find related papers.\">Research question</button></li>
          <li><button type=\"button\" data-prompt=\"Describe this image.\" data-image=\"https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png\">Describe image URL</button></li>
          <li><button type=\"button\" data-prompt=\"How many dogs are in this image?\" data-image=\"./assets/multi-dogs.jpg\">Count objects</button></li>
          <li><button type=\"button\" data-prompt=\"What concept is shown in this image? Find research papers about it.\" data-image=\"./assets/rope.png\">Vision + research</button></li>
        </ul>
        <p class=\"note\">For local images, put files in <code>assets/</code> and reference them as <code>./assets/name.jpg</code>.</p>
      </aside>
    </section>

    <section class=\"panel\" style=\"margin-top:16px\">
      <label>Answer</label>
      <div id=\"output\" class=\"output\">No answer yet.</div>
    </section>
  </main>

  <script>
    const form = document.getElementById('askForm');
    const promptEl = document.getElementById('prompt');
    const imageEl = document.getElementById('image');
    const output = document.getElementById('output');
    const statusEl = document.getElementById('status');
    const submitBtn = document.getElementById('submitBtn');

    function setStatus(text, className = '') {
      statusEl.textContent = text;
      statusEl.className = 'status ' + className;
    }

    document.querySelectorAll('[data-prompt]').forEach((button) => {
      button.addEventListener('click', () => {
        promptEl.value = button.dataset.prompt || '';
        imageEl.value = button.dataset.image || '';
      });
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
      promptEl.value = '';
      imageEl.value = '';
      output.textContent = 'No answer yet.';
      setStatus('Idle');
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptEl.value.trim();
      const image = imageEl.value.trim();
      if (!prompt && !image) {
        output.textContent = 'Please enter a prompt or image path/URL.';
        setStatus('Missing input', 'error');
        return;
      }

      submitBtn.disabled = true;
      output.textContent = 'Thinking...';
      setStatus('Running');

      try {
        const response = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, image })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        output.textContent = data.answer || '(No answer returned)';
        setStatus('Done', 'ok');
      } catch (error) {
        output.textContent = error.message;
        setStatus('Error', 'error');
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class VisualAgentHandler(BaseHTTPRequestHandler):
    def _send_text(self, status: int, content: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send_text(status, json.dumps(payload), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(200, HTML_PAGE)
            return
        if parsed.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_text(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/ask":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw) if raw else {}
            prompt = str(payload.get("prompt", "")).strip()
            image = str(payload.get("image", "")).strip()
            user_input = prompt
            if image:
                user_input = f"{prompt}\nImage: {image}" if prompt else f"Image: {image}"
            if not user_input:
                self._send_json(400, {"error": "Prompt or image path is required."})
                return

            if image and is_object_count_request(prompt):
                result = detect_and_count_objects.invoke({"text": image})
                self._send_json(200, {"answer": format_detection_answer(result), "mode": "yolo"})
                return

            final_chunk = None
            for chunk in get_supervisor().stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"recursion_limit": 50},
            ):
                final_chunk = chunk

            answer = get_final_text(final_chunk)
            self._send_json(200, {"answer": answer, "mode": "agent"})
        except Exception as exc:
            self._send_json(500, {"error": html.escape(str(exc))})

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), VisualAgentHandler)
    print(f"Visual Agentic AI UI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




