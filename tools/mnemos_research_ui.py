"""Local web UI for MNEMOS research artifact intake.

This UI replaces the long PowerShell command for research intake. It is a
local-only Flask app: upload files, select or type an Ollama model, test
MNEMOS/Ollama connectivity, and run ``tools.mnemos_research_intake``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import sys
from pathlib import Path
from typing import Any, Callable

import requests
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnemos_ollama_chat import DEFAULT_OLLAMA_BASE_URL, normalize_base_url
from tools.mnemos_research_intake import run_intake

DEFAULT_MNEMOS_BASE_URL = "http://127.0.0.1:8700"


def default_ollama_base_url() -> str:
    """Resolve the local Ollama URL from env, including custom OLLAMA_HOST."""
    explicit = os.getenv("OLLAMA_BASE_URL", "").strip()
    if explicit:
        return normalize_base_url(explicit)
    host = os.getenv("OLLAMA_HOST", "").strip()
    if host:
        return normalize_base_url(host)
    return normalize_base_url(DEFAULT_OLLAMA_BASE_URL)


def _normalize_mnemos_url(value: str) -> str:
    text = (value or "").strip() or DEFAULT_MNEMOS_BASE_URL
    if "://" not in text:
        text = "http://" + text
    return text.rstrip("/").replace("http://0.0.0.0", "http://127.0.0.1", 1)


def default_ollama_models(base_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{normalize_base_url(base_url)}/api/tags", timeout=8)
    response.raise_for_status()
    data = response.json()
    models = data.get("models", []) if isinstance(data, dict) else []
    return [model for model in models if isinstance(model, dict) and model.get("name")]


def default_mnemos_health(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{_normalize_mnemos_url(base_url)}/health", timeout=8)
    response.raise_for_status()
    data = response.json()
    status = data.get("status") if isinstance(data, dict) else None
    return {"ok": status in {"ok", "healthy"}, "status": status, "raw": data}


def _split_tags(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _positive_int(raw: str, default: int) -> int:
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _save_uploads(files: list[Any], upload_dir: Path) -> list[Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in files:
        if not item or not getattr(item, "filename", ""):
            continue
        filename = secure_filename(item.filename)
        if not filename:
            continue
        target = upload_dir / filename
        counter = 1
        while target.exists():
            target = upload_dir / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
            counter += 1
        item.save(target)
        saved.append(target)
    return saved


def create_app(
    *,
    upload_dir: Path | None = None,
    intake_runner: Callable[..., dict[str, Any]] = run_intake,
    ollama_models_fn: Callable[[str], list[dict[str, Any]]] = default_ollama_models,
    mnemos_health_fn: Callable[[str], dict[str, Any]] = default_mnemos_health,
) -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_DIR"] = Path(upload_dir or Path(tempfile.gettempdir()) / "mnemos_research_ui_uploads")

    @app.get("/")
    def index():
        return INDEX_HTML.replace("__DEFAULT_OLLAMA_BASE_URL__", default_ollama_base_url())

    @app.get("/api/ollama-models")
    def ollama_models():
        base_url = request.args.get("ollama_base_url", default_ollama_base_url())
        try:
            models = ollama_models_fn(base_url)
            return jsonify({"ok": True, "models": models})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "models": []}), 200

    @app.post("/api/test-connection")
    def test_connection():
        body = request.get_json(silent=True) or {}
        mnemos_base_url = body.get("mnemos_base_url", DEFAULT_MNEMOS_BASE_URL)
        ollama_base_url = body.get("ollama_base_url", default_ollama_base_url())
        mnemos = {"ok": False, "error": "not_checked"}
        ollama = {"ok": False, "error": "not_checked", "model_count": 0}
        try:
            mnemos = mnemos_health_fn(mnemos_base_url)
        except Exception as exc:
            mnemos = {"ok": False, "error": str(exc)}
        try:
            models = ollama_models_fn(ollama_base_url)
            ollama = {"ok": True, "model_count": len(models)}
        except Exception as exc:
            ollama = {"ok": False, "error": str(exc), "model_count": 0}
        return jsonify({"ok": bool(mnemos.get("ok") and ollama.get("ok")), "mnemos": mnemos, "ollama": ollama})

    @app.post("/api/intake")
    def intake():
        uploads = _save_uploads(request.files.getlist("files"), app.config["UPLOAD_DIR"])
        if not uploads:
            return jsonify({"ok": False, "error": "No files selected."}), 400

        mnemos_base_url = _normalize_mnemos_url(request.form.get("mnemos_base_url", DEFAULT_MNEMOS_BASE_URL))
        ollama_base_url = normalize_base_url(request.form.get("ollama_base_url", default_ollama_base_url()))
        project = request.form.get("project", "").strip()
        capability = request.form.get("capability", "").strip()
        if not project or not capability:
            return jsonify({"ok": False, "error": "Project and capability are required."}), 400

        # The existing SDK/adapter read these env vars. Set them process-local
        # for this local UI request.
        import os

        os.environ["MNEMOS_BASE_URL"] = mnemos_base_url
        os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        os.environ["MNEMOS_TIMEOUT_S"] = str(_positive_int(request.form.get("mnemos_timeout_s", "180"), 180))

        output_raw = request.form.get("output", "").strip()
        output_path = Path(output_raw) if output_raw else None
        result = intake_runner(
            files=uploads,
            project=project,
            capability=capability,
            status=request.form.get("status", "new").strip() or "new",
            tags=_split_tags(request.form.get("tags", "")),
            summarize=request.form.get("summarize", "").lower() in {"1", "true", "on", "yes"},
            output_path=output_path,
            ollama_model=request.form.get("ollama_model", "").strip() or "llama3.1",
            batch_size=_positive_int(request.form.get("batch_size", "25"), 25),
        )
        return jsonify({"ok": result.get("status") == "ok", "result": result, "uploaded_files": [str(p) for p in uploads]})

    return app


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNEMOS Research Intake</title>
  <style>
    :root { color-scheme: light; --ink:#172026; --muted:#52616b; --line:#d9e1e8; --fill:#f6f8fa; --accent:#176b87; --ok:#166534; --bad:#991b1b; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, sans-serif; color:var(--ink); background:#ffffff; }
    main { max-width: 1080px; margin: 0 auto; padding: 28px 20px 44px; }
    h1 { margin: 0 0 6px; font-size: 30px; font-weight: 700; }
    p { color: var(--muted); line-height: 1.45; }
    form { display:grid; gap:18px; }
    fieldset { border:1px solid var(--line); border-radius:8px; padding:18px; }
    legend { font-weight:700; padding:0 8px; }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }
    label { display:grid; gap:6px; font-size: 14px; font-weight: 700; }
    input, select { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px 11px; font-size:14px; background:#fff; }
    input[type=file] { padding: 9px; }
    .inline { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .inline label { display:flex; flex-direction:row; align-items:center; font-weight:600; }
    button { border:0; border-radius:6px; padding:10px 14px; font-weight:700; cursor:pointer; }
    button.primary { background:var(--accent); color:#fff; }
    button.secondary { background:#e8eef2; color:#172026; }
    button:disabled { opacity:.58; cursor:not-allowed; }
    .status { min-height: 44px; border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--fill); white-space:pre-wrap; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .hint { font-size:13px; color:var(--muted); font-weight:400; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>MNEMOS Research Intake</h1>
  <p>Upload PDFs, Word files, Markdown, code, or data files into MNEMOS, optionally summarize with a local Ollama model, then use Ollama separately for evidence-grounded prompting.</p>

  <form id="intakeForm">
    <fieldset>
      <legend>Connections</legend>
      <div class="grid">
        <label>MNEMOS base URL
          <input name="mnemos_base_url" id="mnemosBaseUrl" value="http://127.0.0.1:8700">
        </label>
        <label>MNEMOS timeout seconds
          <input name="mnemos_timeout_s" id="mnemosTimeoutS" type="number" min="1" value="180">
          <span class="hint">PDF embedding/indexing can exceed the SDK default 5 seconds.</span>
        </label>
        <label>Ollama base URL
          <input name="ollama_base_url" id="ollamaBaseUrl" value="__DEFAULT_OLLAMA_BASE_URL__">
          <span class="hint">Detected from OLLAMA_BASE_URL or OLLAMA_HOST when available. You can type a different local URL.</span>
        </label>
      </div>
      <div class="inline" style="margin-top:12px;">
        <button class="secondary" type="button" id="testConnection">Test Connection</button>
        <button class="secondary" type="button" id="refreshModels">Refresh Models</button>
      </div>
    </fieldset>

    <fieldset>
      <legend>Artifact Metadata</legend>
      <div class="grid">
        <label>Project
          <input name="project" value="MNEMOS" required>
        </label>
        <label>Capability
          <input name="capability" placeholder="local research memory" required>
        </label>
        <label>Status
          <select name="status">
            <option value="new">new</option>
            <option value="reviewed">reviewed</option>
            <option value="promising">promising</option>
            <option value="rejected">rejected</option>
            <option value="integrated">integrated</option>
          </select>
        </label>
        <label>Tags
          <input name="tags" placeholder="workflow, pdf, github">
          <span class="hint">Comma-separated.</span>
        </label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Files And Summary</legend>
      <label>Files
        <input name="files" type="file" multiple required>
      </label>
      <div class="grid" style="margin-top:14px;">
        <label>Ollama model
          <select id="ollamaModelSelect"></select>
          <input name="ollama_model" id="ollamaModel" placeholder="type model name if not listed">
          <span class="hint">Only needed when "Summarize with Ollama" is checked; you can type a model manually.</span>
        </label>
        <label>Index batch size
          <input name="batch_size" type="number" min="1" value="25">
          <span class="hint">Smaller batches are slower but less likely to time out on large PDFs.</span>
        </label>
        <label>Output packet path
          <input name="output" placeholder="docs/research/my_packet.md">
        </label>
      </div>
      <div class="inline" style="margin-top:12px;">
        <label><input name="summarize" type="checkbox" value="true" checked> Summarize with Ollama</label>
      </div>
    </fieldset>

    <div class="inline">
      <button class="primary" type="submit" id="runIntake">Run Intake</button>
    </div>
  </form>

  <h2>Result</h2>
  <div class="status" id="statusBox">Ready.</div>
</main>
<script>
const statusBox = document.getElementById('statusBox');
const modelSelect = document.getElementById('ollamaModelSelect');
const modelInput = document.getElementById('ollamaModel');

function setStatus(text, ok=null) {
  statusBox.className = 'status' + (ok === true ? ' ok' : ok === false ? ' bad' : '');
  statusBox.textContent = text;
}

async function refreshModels() {
  const base = document.getElementById('ollamaBaseUrl').value;
  setStatus('Loading Ollama models...');
  const res = await fetch('/api/ollama-models?ollama_base_url=' + encodeURIComponent(base));
  const data = await res.json();
  modelSelect.innerHTML = '';
  if (!data.ok) {
    setStatus('Ollama model lookup failed: ' + data.error, false);
    return;
  }
  data.models.forEach(model => {
    const opt = document.createElement('option');
    opt.value = model.name;
    opt.textContent = model.name;
    modelSelect.appendChild(opt);
  });
  if (data.models.length) {
    modelInput.value = data.models[0].name;
  }
  setStatus('Loaded ' + data.models.length + ' Ollama model(s).', true);
}

modelSelect.addEventListener('change', () => { modelInput.value = modelSelect.value; });
document.getElementById('refreshModels').addEventListener('click', refreshModels);

document.getElementById('testConnection').addEventListener('click', async () => {
  setStatus('Testing connections...');
  const res = await fetch('/api/test-connection', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      mnemos_base_url: document.getElementById('mnemosBaseUrl').value,
      ollama_base_url: document.getElementById('ollamaBaseUrl').value
    })
  });
  const data = await res.json();
  setStatus(JSON.stringify(data, null, 2), data.ok);
});

document.getElementById('intakeForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  setStatus('Running intake...');
  const form = new FormData(event.target);
  form.set('ollama_model', modelInput.value || modelSelect.value || 'llama3.1');
  const res = await fetch('/api/intake', { method: 'POST', body: form });
  const data = await res.json();
  setStatus(JSON.stringify(data, null, 2), data.ok);
});

refreshModels();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--upload-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(upload_dir=args.upload_dir)
    print(f"MNEMOS Research Intake UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
