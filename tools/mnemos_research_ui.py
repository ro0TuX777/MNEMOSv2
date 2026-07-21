"""Local web UI for MNEMOS research artifact intake.

This UI replaces the long PowerShell command for research intake. It is a
local-only Flask app: upload files, select or type an Ollama model, test
MNEMOS/Ollama connectivity, and run ``tools.mnemos_research_intake``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnemos_ollama_chat import DEFAULT_OLLAMA_BASE_URL, normalize_base_url
from tools.mnemos_research_intake import run_intake

DEFAULT_MNEMOS_BASE_URL = os.getenv("MNEMOS_BASE_URL", "http://127.0.0.1:8700")
DEFAULT_RECEIPT_DIR = ROOT / "logs" / "evidence_receipts"


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


def safe_receipt_id(receipt_id: str) -> str | None:
    safe = "".join(ch for ch in str(receipt_id or "") if ch.isalnum() or ch in {"-", "_"})
    return safe or None


def _receipt_path(receipt_dir: Path, receipt_id: str) -> Path | None:
    safe = safe_receipt_id(receipt_id)
    if not safe:
        return None
    return receipt_dir / f"{safe}.json"


def load_evidence_receipt(receipt_dir: Path, receipt_id: str) -> dict[str, Any] | None:
    path = _receipt_path(receipt_dir, receipt_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _query_preview(query: str, max_chars: int = 96) -> str:
    clean = " ".join(str(query or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _receipt_retrieval_mode(receipt: dict[str, Any]) -> str:
    metadata = receipt.get("retrieval_metadata") or {}
    if not isinstance(metadata, dict):
        return "unknown"
    return str(
        metadata.get("retrieval_mode")
        or metadata.get("effective_retrieval_mode")
        or metadata.get("requested_retrieval_mode")
        or "unknown"
    )


def list_evidence_receipts(receipt_dir: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    if not receipt_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        if len(rows) >= limit:
            break
        receipt = load_evidence_receipt(receipt_dir, path.stem)
        if not receipt:
            continue
        citations = receipt.get("citations") or []
        citation_check = receipt.get("citation_check") or {}
        generation = receipt.get("generation") or {}
        rows.append(
            {
                "receipt_id": str(receipt.get("receipt_id") or path.stem),
                "created": receipt.get("created"),
                "created_label": _format_created(receipt.get("created")),
                "query_preview": _query_preview(str(receipt.get("query") or "")),
                "model": str(receipt.get("requested_model") or receipt.get("actual_model") or ""),
                "status": str(receipt.get("status") or "unknown"),
                "source_count": len(citations) if isinstance(citations, list) else 0,
                "retrieval_mode": _receipt_retrieval_mode(receipt),
                "verdict": citation_check.get("verdict") if isinstance(citation_check, dict) else None,
                "coverage": citation_check.get("coverage") if isinstance(citation_check, dict) else None,
                "evidence_count": citation_check.get("evidence_count") if isinstance(citation_check, dict) else None,
                "truncated": bool(generation.get("truncated")) if isinstance(generation, dict) else False,
            }
        )
    return rows


def _format_created(created: Any) -> str:
    try:
        return datetime.fromtimestamp(float(created)).strftime("%b %d, %Y %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _pill(kind: str, label: str) -> str:
    css = f"pill {kind}".strip()
    return f'<span class="{css}"><span class="dot"></span>{html.escape(label)}</span>'


_VERDICT_PILLS = {
    "all_evidence_cited": ("ok", "all cited"),
    "partial_evidence_cited": ("accent", "partial coverage"),
    "cites_missing_evidence": ("warn", "cites missing"),
    "no_citations_in_answer": ("warn", "no citations"),
    "no_evidence_available": ("", "no evidence"),
}


def _verdict_pill(verdict: Any) -> str:
    kind, label = _VERDICT_PILLS.get(str(verdict or ""), ("", "not recorded"))
    return _pill(kind, label)


def _generation_pill(status: str, truncated: bool) -> str:
    if status == "ok":
        return _pill("warn", "truncated") if truncated else _pill("ok", "complete")
    if status == "no_evidence":
        return _pill("warn", "withheld")
    if status == "mnemos_error":
        return _pill("bad", "MNEMOS error")
    if status == "ollama_error":
        return _pill("bad", "Ollama error")
    return _pill("", status or "unknown")


def _status_pill(status: str) -> str:
    kind = "ok" if status == "ok" else "bad" if status.endswith("error") else "warn"
    return _pill(kind, f"status · {status or 'unknown'}")


def _json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


BASE_CSS = """
  :root {
    color-scheme: light dark;
    --paper: #F6F8F7; --panel: #FFFFFF; --panel-2: #EEF3F1;
    --ink: #1B2723; --muted: #5B6B65;
    --line: #D7E0DC; --line-soft: #E6ECE9;
    --accent: #0B7F66; --accent-ink: #FFFFFF; --accent-soft: #0B7F6614;
    --copper: #A85E28;
    --ok: #1E7A46; --warn: #9A6700; --bad: #A3312C;
    --ok-soft: #1E7A4614; --warn-soft: #9A670014; --bad-soft: #A3312C12;
    --shadow: 0 1px 2px rgba(27,39,35,.05), 0 4px 14px rgba(27,39,35,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #0E1412; --panel: #151D1A; --panel-2: #1B2521;
      --ink: #E2EAE6; --muted: #93A39C;
      --line: #2A3833; --line-soft: #223029;
      --accent: #0FA07E; --accent-ink: #07130F; --accent-soft: #0FA07E1F;
      --copper: #D08B4F;
      --ok: #44A05F; --warn: #B58322; --bad: #E0685E;
      --ok-soft: #44A05F1F; --warn-soft: #B583221F; --bad-soft: #E0685E1C;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(0,0,0,.25);
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 14px/1.5 system-ui, "Segoe UI", sans-serif; }
  .mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }
  button { font: inherit; cursor: pointer; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
  .muted { color: var(--muted); }

  header.masthead { border-bottom: 1px solid var(--line); background: var(--panel); }
  .masthead-inner { max-width: 1160px; margin: 0 auto; padding: 14px 24px; display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
  .wordmark { display: flex; flex-direction: column; gap: 1px; }
  .wordmark .name { font-family: "Iowan Old Style", Georgia, "Times New Roman", serif; font-size: 21px; font-weight: 700; letter-spacing: .14em; }
  .wordmark .sub { font-size: 10.5px; letter-spacing: .22em; text-transform: uppercase; color: var(--copper); font-weight: 600; }
  nav.primary { display: flex; gap: 4px; margin-left: 8px; }
  nav.primary a { padding: 7px 14px; border-radius: 7px; text-decoration: none; color: var(--muted); font-weight: 600; }
  nav.primary a:hover { color: var(--ink); background: var(--panel-2); }
  nav.primary a.active { color: var(--accent); background: var(--accent-soft); }
  .service-pills { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }

  .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--line); color: var(--muted); background: var(--panel); white-space: nowrap; }
  .pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex: none; }
  .pill.ok { color: var(--ok); border-color: transparent; background: var(--ok-soft); }
  .pill.ok .dot { background: var(--ok); }
  .pill.warn { color: var(--warn); border-color: transparent; background: var(--warn-soft); }
  .pill.warn .dot { background: var(--warn); }
  .pill.bad { color: var(--bad); border-color: transparent; background: var(--bad-soft); }
  .pill.bad .dot { background: var(--bad); }
  .pill.accent { color: var(--accent); border-color: transparent; background: var(--accent-soft); }
  .pill.accent .dot { background: var(--accent); }

  main { max-width: 1160px; margin: 0 auto; padding: 26px 24px 56px; }
  h1 { font-size: 21px; margin: 0 0 4px; }
  .lede { color: var(--muted); margin: 0 0 22px; max-width: 62ch; }

  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
  .card h2 { font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 0; padding: 12px 18px; border-bottom: 1px solid var(--line-soft); font-weight: 700; }
  .card .body { padding: 16px 18px; }

  .field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  @media (max-width: 640px) { .field-grid { grid-template-columns: 1fr; } }
  label.field { display: grid; gap: 5px; font-size: 12.5px; font-weight: 600; color: var(--muted); }
  label.field input, label.field select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--line); border-radius: 7px; padding: 9px 11px; width: 100%; }
  label.field input:hover, label.field select:hover { border-color: var(--muted); }
  .hint { font-weight: 400; font-size: 12px; color: var(--muted); }
  .inline { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .inline label.check { display: flex; flex-direction: row; align-items: center; gap: 7px; font-weight: 600; font-size: 13px; color: var(--ink); }

  .btn-primary { background: var(--accent); color: var(--accent-ink); border: 0; border-radius: 8px; padding: 10px 18px; font-weight: 700; }
  .btn-primary:hover { filter: brightness(1.08); }
  .btn-primary:disabled { opacity: .55; cursor: not-allowed; }
  .btn-quiet { background: none; border: 1px solid var(--line); color: var(--ink); border-radius: 8px; padding: 9px 14px; font-weight: 600; }
  .btn-quiet:hover { border-color: var(--muted); }

  .intake-layout { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 18px; align-items: start; }
  @media (max-width: 900px) { .intake-layout { grid-template-columns: 1fr; } }
  .intake-form { display: grid; gap: 16px; }

  .dropzone { border: 1.5px dashed var(--line); border-radius: 9px; padding: 26px 18px; display: grid; justify-items: center; gap: 6px; text-align: center; color: var(--muted); background: var(--paper); transition: border-color .15s; cursor: pointer; }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); color: var(--ink); }
  .dropzone .big { font-size: 15px; font-weight: 600; color: var(--ink); }
  .filechip { display: flex; align-items: center; gap: 10px; margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; background: var(--panel); }
  .filechip .fname { font-weight: 600; overflow-wrap: anywhere; }
  .filechip .fmeta { color: var(--muted); font-size: 12px; margin-left: auto; white-space: nowrap; }
  .filechip .x { border: 0; background: none; color: var(--muted); font-size: 15px; padding: 2px 6px; }
  .filechip .x:hover { color: var(--bad); }

  .runlog { position: sticky; top: 18px; }
  .runlog .statusline { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
  .runlog .msg { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 13px; }
  .runlog .kv { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); display: grid; gap: 6px; font-size: 12.5px; }
  .runlog .kv div { display: flex; justify-content: space-between; gap: 12px; }
  .runlog .kv dt { color: var(--muted); }
  .runlog .kv dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; text-align: right; }

  .toolbar { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; align-items: center; }
  .toolbar input[type="search"] { flex: 1; min-width: 220px; font: inherit; color: var(--ink); border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 9px 12px; }
  .chip { border: 1px solid var(--line); background: var(--panel); color: var(--muted); border-radius: 999px; padding: 5px 13px; font-size: 12.5px; font-weight: 600; }
  .chip:hover { color: var(--ink); }
  .chip.on { background: var(--accent-soft); border-color: transparent; color: var(--accent); }

  .tablewrap { overflow-x: auto; }
  table.receipts { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.receipts th { text-align: left; font-size: 10.5px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); font-weight: 700; padding: 10px 14px; border-bottom: 1px solid var(--line); }
  table.receipts td { padding: 11px 14px; border-bottom: 1px solid var(--line-soft); vertical-align: middle; }
  table.receipts tr:hover td { background: var(--accent-soft); cursor: pointer; }
  td.rid { font-size: 12px; color: var(--muted); white-space: nowrap; }
  td.query { max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  td.time { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }

  .meter { display: flex; align-items: center; gap: 8px; min-width: 110px; }
  .meter .track { flex: 1; height: 6px; border-radius: 4px; background: var(--line-soft); overflow: hidden; }
  .meter .fill { height: 100%; border-radius: 4px; background: var(--accent); }
  .meter .val { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }

  .receipt-doc { max-width: 860px; }
  .receipt-head { padding: 20px 22px; border-bottom: 1px solid var(--line); display: grid; gap: 10px; }
  .receipt-head .row1 { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .receipt-head .rid { font-size: 13px; overflow-wrap: anywhere; }
  .copybtn { border: 1px solid var(--line); background: none; color: var(--muted); border-radius: 6px; padding: 2px 9px; font-size: 11.5px; }
  .copybtn:hover { color: var(--accent); border-color: var(--accent); }
  .receipt-head .when { color: var(--muted); font-size: 12.5px; margin-left: auto; white-space: nowrap; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; }

  .seal { display: flex; gap: 10px; align-items: center; padding: 12px 22px; border-bottom: 1px solid var(--line); background: var(--panel-2); font-size: 12.5px; }
  .seal .mark { color: var(--copper); font-size: 16px; }
  .seal .label { font-weight: 700; color: var(--copper); letter-spacing: .1em; text-transform: uppercase; font-size: 10.5px; white-space: nowrap; }
  .seal .hash { color: var(--muted); word-break: break-all; }

  .verif { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-bottom: 1px solid var(--line); }
  @media (max-width: 720px) { .verif { grid-template-columns: 1fr; } }
  .verif > div { padding: 16px 22px; }
  .verif > div + div { border-left: 1px solid var(--line-soft); }
  @media (max-width: 720px) { .verif > div + div { border-left: 0; border-top: 1px solid var(--line-soft); } }
  .verif h3 { margin: 0 0 8px; font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
  .verif .headline { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
  .verif .note { color: var(--muted); font-size: 12.5px; }

  .coverage-track { height: 8px; border-radius: 4px; background: var(--line-soft); overflow: hidden; margin: 8px 0 6px; }
  .coverage-fill { height: 100%; background: var(--accent); border-radius: 4px; }

  .spread { display: flex; align-items: flex-end; gap: 5px; height: 44px; margin: 8px 0 4px; }
  .spread .bar { width: 14px; background: var(--accent); border-radius: 3px 3px 0 0; }
  .spread .bar.dim { opacity: .38; }

  .rsection { padding: 16px 22px; border-bottom: 1px solid var(--line-soft); }
  .rsection:last-child { border-bottom: 0; }
  .rsection h3 { margin: 0 0 8px; font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
  .rsection p.text { margin: 0; max-width: 68ch; overflow-wrap: anywhere; }
  .rsection .qtext { font-size: 15px; font-weight: 600; }
  .rsection .subnote { color: var(--muted); font-size: 12.5px; margin-top: 6px; }

  table.cites { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.cites th { text-align: left; color: var(--muted); font-weight: 600; padding: 6px 10px; border-bottom: 1px solid var(--line-soft); }
  table.cites td { padding: 7px 10px; border-bottom: 1px solid var(--line-soft); overflow-wrap: anywhere; }
  table.cites td.score { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cited-mark { color: var(--ok); font-weight: 700; }
  .uncited-mark { color: var(--muted); }

  details.fold summary { cursor: pointer; font-weight: 600; color: var(--accent); font-size: 13px; list-style: none; display: inline-flex; align-items: center; gap: 6px; }
  details.fold summary::before { content: "\\25B8"; transition: transform .12s; }
  details.fold[open] summary::before { transform: rotate(90deg); }
  @media (prefers-reduced-motion: reduce) { details.fold summary::before, .dropzone { transition: none; } }
  details.fold pre { margin: 10px 0 0; }
  pre { padding: 12px; background: var(--paper); border: 1px solid var(--line-soft); border-radius: 8px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; max-height: 320px; overflow-y: auto; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }
  .boundary { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
"""


def _render_page(
    title: str,
    body: str,
    *,
    active: str = "evidence",
    script: str = "",
    pills: str = "",
) -> str:
    nav_intake = ' class="active"' if active == "intake" else ""
    nav_evidence = ' class="active"' if active == "evidence" else ""
    script_tag = f"<script>{script}</script>" if script else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{BASE_CSS}</style>
</head>
<body>
<header class="masthead">
  <div class="masthead-inner">
    <div class="wordmark"><span class="name">MNEMOS</span><span class="sub">Evidence Desk</span></div>
    <nav class="primary" aria-label="Sections">
      <a href="/"{nav_intake}>Intake</a>
      <a href="/evidence"{nav_evidence}>Evidence Receipts</a>
    </nav>
    <div class="service-pills">{pills}</div>
  </div>
</header>
<main>
{body}
</main>
{script_tag}
</body>
</html>"""


def _coverage_meter(coverage: Any, evidence_count: Any) -> str:
    try:
        ratio = max(0.0, min(1.0, float(coverage)))
        total = int(evidence_count)
    except (TypeError, ValueError):
        return (
            '<div class="meter"><div class="track"><div class="fill" style="width:0%"></div></div>'
            '<span class="val">&mdash;</span></div>'
        )
    cited = round(ratio * total)
    return (
        f'<div class="meter"><div class="track"><div class="fill" style="width:{ratio * 100:.0f}%"></div></div>'
        f'<span class="val">{cited}/{total}</span></div>'
    )


def _receipt_filter_bucket(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "unknown")
    if status == "ok":
        return "truncated" if row.get("truncated") else "ok"
    if status == "no_evidence":
        return "no_evidence"
    return "error"


def render_evidence_list_page(receipts: list[dict[str, Any]]) -> str:
    row_parts: list[str] = []
    bucket_counts = {"ok": 0, "truncated": 0, "no_evidence": 0, "error": 0}
    for row in receipts:
        bucket = _receipt_filter_bucket(row)
        bucket_counts[bucket] += 1
        rid = str(row["receipt_id"])
        short_id = "&hellip;" + html.escape(rid[-12:]) if len(rid) > 12 else html.escape(rid)
        status = str(row.get("status") or "unknown")
        row_parts.append(
            f'<tr tabindex="0" data-bucket="{bucket}" data-mode="{html.escape(str(row.get("retrieval_mode") or ""))}"'
            f' data-href="/evidence/{html.escape(rid)}">'
            f'<td class="query" title="{html.escape(row["query_preview"])}">{html.escape(row["query_preview"])}</td>'
            f"<td>{_verdict_pill(row.get('verdict'))}</td>"
            f"<td>{_coverage_meter(row.get('coverage'), row.get('evidence_count'))}</td>"
            f"<td>{_generation_pill(status, bool(row.get('truncated')))}</td>"
            f'<td class="rid mono">{short_id}</td>'
            f'<td class="time">{html.escape(row.get("created_label") or "")}</td>'
            "</tr>"
        )
    rows = "\n".join(row_parts) or '<tr><td colspan="6" class="muted">No evidence receipts found.</td></tr>'
    total = len(receipts)
    chips = "\n".join(
        f'<button class="chip{" on" if key == "all" else ""}" type="button" data-filter="{key}">{label} &middot; {count}</button>'
        for key, label, count in [
            ("all", "All", total),
            ("ok", "ok", bucket_counts["ok"]),
            ("truncated", "truncated", bucket_counts["truncated"]),
            ("no_evidence", "no evidence", bucket_counts["no_evidence"]),
            ("error", "errors", bucket_counts["error"]),
        ]
    )
    body = f"""
<h1>MNEMOS Evidence Receipts</h1>
<p class="lede">Every MNEMOS-grounded answer leaves a receipt: the evidence sent, what the answer cited, and whether generation completed honestly.</p>
<div class="toolbar">
  <input type="search" id="receiptSearch" placeholder="Search query text or receipt id&hellip;" aria-label="Search receipts">
  {chips}
</div>
<div class="card">
  <div class="tablewrap">
    <table class="receipts">
      <thead><tr><th>Query</th><th>Verdict</th><th>Coverage</th><th>Generation</th><th>Receipt</th><th>When</th></tr></thead>
      <tbody id="receiptRows">{rows}</tbody>
    </table>
  </div>
</div>"""
    script = """
const search = document.getElementById('receiptSearch');
const chips = document.querySelectorAll('.toolbar .chip');
let bucket = 'all';
function applyFilters() {
  const term = (search.value || '').toLowerCase();
  document.querySelectorAll('#receiptRows tr[data-bucket]').forEach((row) => {
    const bucketHit = bucket === 'all' || row.dataset.bucket === bucket;
    const termHit = !term || row.textContent.toLowerCase().includes(term);
    row.style.display = bucketHit && termHit ? '' : 'none';
  });
}
chips.forEach((chip) => chip.addEventListener('click', () => {
  bucket = chip.dataset.filter;
  chips.forEach((c) => c.classList.toggle('on', c === chip));
  applyFilters();
}));
search.addEventListener('input', applyFilters);
document.querySelectorAll('#receiptRows tr[data-href]').forEach((row) => {
  row.addEventListener('click', () => { window.location = row.dataset.href; });
  row.addEventListener('keydown', (e) => { if (e.key === 'Enter') window.location = row.dataset.href; });
});
"""
    return _render_page("MNEMOS Evidence Receipts", body, active="evidence", script=script)


def _coverage_panel(citation_check: dict[str, Any]) -> str:
    coverage = citation_check.get("coverage")
    evidence_count = citation_check.get("evidence_count")
    if not citation_check or coverage is None or not evidence_count:
        if citation_check and citation_check.get("verdict") == "no_evidence_available":
            return (
                '<div class="headline">No evidence available</div>'
                '<div class="note">No chunks were admitted, so there was nothing to cite.</div>'
            )
        return (
            '<div class="headline">Not recorded</div>'
            '<div class="note">This receipt predates citation verification.</div>'
        )
    ratio = max(0.0, min(1.0, float(coverage)))
    total = int(evidence_count)
    cited = [i for i in (citation_check.get("cited_indices") or [])]
    invalid = [i for i in (citation_check.get("invalid_indices") or [])]
    uncited = [i for i in (citation_check.get("uncited_evidence_indices") or [])]
    parts = []
    if cited:
        parts.append("Answer cited " + " ".join(f"[{i}]" for i in cited))
    else:
        parts.append("Answer contains no bracket citations")
    if uncited:
        parts.append("never cited " + " ".join(f"[{i}]" for i in uncited))
    if invalid:
        parts.append("cites missing " + " ".join(f"[{i}]" for i in invalid))
    else:
        parts.append("no invalid citations")
    return (
        f'<div class="headline">{round(ratio * total)} of {total} chunks cited</div>'
        f'<div class="coverage-track"><div class="coverage-fill" style="width:{ratio * 100:.0f}%"></div></div>'
        f'<div class="note">{html.escape(" · ".join(parts))}</div>'
    )


def _generation_panel(generation: dict[str, Any], status: str) -> str:
    done_reason = generation.get("done_reason")
    if not generation or (done_reason is None and generation.get("eval_count") is None):
        if status != "ok":
            return (
                '<div class="headline">Not generated</div>'
                f'<div class="note">Status {html.escape(status)} — Ollama was not called or did not complete.</div>'
            )
        return (
            '<div class="headline">Not recorded</div>'
            '<div class="note">This receipt predates generation annotations.</div>'
        )
    truncated = bool(generation.get("truncated"))
    headline = "Stopped at token limit" if truncated else "Completed"
    bits = []
    if done_reason:
        bits.append(f"done_reason = {done_reason}")
    prompt_tokens = generation.get("prompt_eval_count")
    eval_tokens = generation.get("eval_count")
    if isinstance(prompt_tokens, int) and isinstance(eval_tokens, int):
        bits.append(f"{prompt_tokens:,} prompt + {eval_tokens:,} completion tokens")
    if truncated:
        bits.append("citations may be incomplete — the served footer carries the same warning")
    return (
        f'<div class="headline">{headline}</div>'
        f'<div class="note">{html.escape(" · ".join(bits))}</div>'
    )


def _spread_panel(
    citations: list[dict[str, Any]],
    score_stats: dict[str, Any],
    cited_indices: set[Any],
    has_check: bool,
) -> str:
    scored: list[tuple[Any, float]] = []
    for item in citations:
        score = item.get("score")
        if score is None:
            continue
        try:
            scored.append((item.get("index"), float(score)))
        except (TypeError, ValueError):
            continue
    if not scored:
        return (
            '<div class="headline">No scores</div>'
            '<div class="note">No retrieval scores were recorded on this receipt.</div>'
        )
    top = max(score for _, score in scored) or 1.0
    bars = "".join(
        f'<div class="bar{" dim" if has_check and index not in cited_indices else ""}"'
        f' style="height:{max(6.0, score / top * 100):.0f}%" title="[{html.escape(str(index))}] {score:.4f}"></div>'
        for index, score in scored[:10]
    )
    values = [score for _, score in scored]
    stats = {
        "max": score_stats.get("max", round(max(values), 4)),
        "mean": score_stats.get("mean", round(sum(values) / len(values), 4)),
        "min": score_stats.get("min", round(min(values), 4)),
    }
    label = ", ".join(f"{score:.2f}" for _, score in scored[:10])
    return (
        f'<div class="spread" role="img" aria-label="Retrieval scores: {label}">{bars}</div>'
        f'<div class="note">max {stats["max"]} · mean {stats["mean"]} · min {stats["min"]}'
        " — the weak tail is shown, not hidden.</div>"
    )


def render_evidence_detail_page(receipt: dict[str, Any]) -> str:
    citations = [item for item in (receipt.get("citations") or []) if isinstance(item, dict)]
    citation_check = receipt.get("citation_check")
    citation_check = citation_check if isinstance(citation_check, dict) else {}
    generation = receipt.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    score_stats = receipt.get("score_stats")
    score_stats = score_stats if isinstance(score_stats, dict) else {}
    metadata = receipt.get("retrieval_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    status = str(receipt.get("status") or "unknown")
    truncated = bool(generation.get("truncated"))
    cited_indices = set(citation_check.get("cited_indices") or [])

    badges = [_status_pill(status)]
    if citation_check:
        badges.append(_verdict_pill(citation_check.get("verdict")))
    if truncated:
        badges.append(_pill("warn", "truncated at token limit"))
    if metadata.get("query_condensed"):
        badges.append(_pill("", "query condensed"))

    requested_model = str(receipt.get("requested_model") or "")
    actual_model = str(receipt.get("actual_model") or "")
    model_label = requested_model or actual_model
    if actual_model and requested_model and actual_model != requested_model:
        model_label = f"{requested_model} → {actual_model}"
    when_label = " · ".join(part for part in [_format_created(receipt.get("created")), model_label] if part)

    content_hash = str(receipt.get("content_hash") or "")
    hash_html = (
        f'<span class="hash mono">{html.escape(content_hash)}</span>'
        if content_hash
        else '<span class="hash muted">not recorded (pre-verification receipt)</span>'
    )

    def _mark(index: Any) -> str:
        if not citation_check:
            return "<td></td>"
        if index in cited_indices:
            return '<td class="cited-mark">✓</td>'
        return '<td class="uncited-mark">·</td>'

    citation_rows = "\n".join(
        "<tr>"
        + _mark(item.get("index"))
        + f"<td>[{html.escape(str(item.get('index', '')))}]</td>"
        f"<td>{html.escape(str(item.get('source', 'unknown')))}</td>"
        f'<td class="score">{html.escape(str(item.get("score", "")))}</td>'
        f'<td class="mono">{html.escape(str(item.get("engram_id", "")))}</td>'
        "</tr>"
        for item in citations
    ) or '<tr><td colspan="5" class="muted">No citations recorded.</td></tr>'

    condensed_note = ""
    if metadata.get("query_condensed"):
        retrieval_query = str(metadata.get("retrieval_query") or "")
        turns = metadata.get("history_turns")
        turns_text = f" · {turns} history turns forwarded" if turns else ""
        condensed_note = (
            f'<p class="subnote">Condensed for retrieval → <em>{html.escape(retrieval_query)}</em>{html.escape(turns_text)}</p>'
        )

    answer = str(receipt.get("answer") or "")
    answer_html = (
        f'<p class="text">{html.escape(answer)}</p>'
        if answer
        else '<p class="text muted">No answer generated.</p>'
    )
    warning = str(receipt.get("warning") or "")
    warning_html = (
        f'<div class="rsection"><h3>Warning</h3><p class="text">{html.escape(warning)}</p></div>'
        if warning
        else ""
    )

    evidence_block = str(receipt.get("evidence_block") or "")
    body = f"""
<h1>Evidence receipt</h1>
<p class="lede">The full trail behind one answer — what was asked, what evidence was admitted, what the answer actually used.</p>

<div class="card receipt-doc">
  <div class="receipt-head">
    <div class="row1">
      <span class="rid mono" id="receiptId">{html.escape(str(receipt.get('receipt_id', '')))}</span>
      <button class="copybtn" type="button" id="copyId">copy id</button>
      <span class="when">{html.escape(when_label)}</span>
    </div>
    <div class="badges">{''.join(badges)}</div>
  </div>

  <div class="seal">
    <span class="mark">◈</span>
    <span class="label">Integrity</span>
    {hash_html}
  </div>

  <div class="verif">
    <div><h3>Citation coverage</h3>{_coverage_panel(citation_check)}</div>
    <div><h3>Generation</h3>{_generation_panel(generation, status)}</div>
    <div><h3>Score spread</h3>{_spread_panel(citations, score_stats, cited_indices, bool(citation_check))}</div>
  </div>

  <div class="rsection">
    <h3>Query</h3>
    <p class="text qtext">{html.escape(str(receipt.get('query') or ''))}</p>
    {condensed_note}
  </div>

  <div class="rsection">
    <h3>Citations</h3>
    <table class="cites">
      <thead><tr><th></th><th>#</th><th>Source</th><th>Score</th><th>Engram ID</th></tr></thead>
      <tbody>{citation_rows}</tbody>
    </table>
  </div>

  <div class="rsection">
    <h3>Answer</h3>
    {answer_html}
  </div>
  {warning_html}

  <div class="rsection">
    <details class="fold">
      <summary>Evidence block sent to Ollama · {len(citations)} chunks · {len(evidence_block):,} chars</summary>
      <pre>{html.escape(evidence_block)}</pre>
    </details>
  </div>

  <div class="rsection">
    <details class="fold">
      <summary>Retrieval metadata · mode {html.escape(_receipt_retrieval_mode(receipt))}</summary>
      <pre>{html.escape(_json_pretty(receipt.get('retrieval_metadata') or {}))}</pre>
    </details>
  </div>

  <div class="rsection">
    <h3>Boundary</h3>
    <p class="boundary mono">{html.escape(str(receipt.get('claim_boundary') or ''))}</p>
  </div>
</div>"""
    script = """
const copyBtn = document.getElementById('copyId');
if (copyBtn && navigator.clipboard) {
  copyBtn.addEventListener('click', async () => {
    await navigator.clipboard.writeText(document.getElementById('receiptId').textContent.trim());
    copyBtn.textContent = 'copied';
    setTimeout(() => { copyBtn.textContent = 'copy id'; }, 1400);
  });
}
"""
    return _render_page("MNEMOS Evidence Receipt", body, active="evidence", script=script)


def create_app(
    *,
    upload_dir: Path | None = None,
    receipt_dir: Path | None = None,
    intake_runner: Callable[..., dict[str, Any]] = run_intake,
    ollama_models_fn: Callable[[str], list[dict[str, Any]]] = default_ollama_models,
    mnemos_health_fn: Callable[[str], dict[str, Any]] = default_mnemos_health,
) -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_DIR"] = Path(upload_dir or Path(tempfile.gettempdir()) / "mnemos_research_ui_uploads")
    app.config["RECEIPT_DIR"] = Path(receipt_dir or os.getenv("MNEMOS_EVIDENCE_RECEIPT_DIR", str(DEFAULT_RECEIPT_DIR)))

    @app.get("/")
    def index():
        pills = (
            '<span class="pill" id="svcMnemos"><span class="dot"></span>MNEMOS · checking…</span>'
            '<span class="pill" id="svcOllama"><span class="dot"></span>Ollama · checking…</span>'
        )
        page = _render_page(
            "MNEMOS Research Intake",
            INDEX_BODY,
            active="intake",
            script=INDEX_SCRIPT,
            pills=pills,
        )
        return page.replace("__DEFAULT_OLLAMA_BASE_URL__", default_ollama_base_url()).replace(
            "__DEFAULT_MNEMOS_BASE_URL__", DEFAULT_MNEMOS_BASE_URL
        )

    @app.get("/evidence")
    def evidence_list():
        receipts = list_evidence_receipts(app.config["RECEIPT_DIR"])
        return Response(render_evidence_list_page(receipts), mimetype="text/html")

    @app.get("/evidence/<receipt_id>")
    def evidence_detail(receipt_id: str):
        receipt = load_evidence_receipt(app.config["RECEIPT_DIR"], receipt_id)
        if receipt is None:
            return Response("MNEMOS evidence receipt not found.", status=404, mimetype="text/plain")
        return Response(render_evidence_detail_page(receipt), mimetype="text/html")

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
        try:
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
        except Exception as exc:
            app.logger.exception("Research intake failed")
            return jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "uploaded_files": [str(p) for p in uploads],
                }
            ), 500
        return jsonify({"ok": result.get("status") == "ok", "result": result, "uploaded_files": [str(p) for p in uploads]})

    return app


INDEX_BODY = """
  <h1>MNEMOS Research Intake</h1>
  <p class="lede">Upload PDFs, documents, or code. Files are extracted, chunked, and indexed into MNEMOS with source lineage — ready for evidence-grounded chat.</p>

  <div class="intake-layout">
    <form id="intakeForm" class="intake-form">
      <div class="card">
        <h2>Connections</h2>
        <div class="body">
          <div class="field-grid">
            <label class="field">MNEMOS base URL
              <input name="mnemos_base_url" id="mnemosBaseUrl" value="__DEFAULT_MNEMOS_BASE_URL__">
            </label>
            <label class="field">MNEMOS timeout seconds
              <input name="mnemos_timeout_s" id="mnemosTimeoutS" type="number" min="1" value="180">
              <span class="hint">PDF embedding/indexing can exceed the SDK default 5 seconds.</span>
            </label>
            <label class="field">Ollama base URL
              <input name="ollama_base_url" id="ollamaBaseUrl" value="__DEFAULT_OLLAMA_BASE_URL__">
              <span class="hint">Detected from OLLAMA_BASE_URL or OLLAMA_HOST when available.</span>
            </label>
          </div>
          <div class="inline" style="margin-top:12px;">
            <button class="btn-quiet" type="button" id="testConnection">Test Connection</button>
            <button class="btn-quiet" type="button" id="refreshModels">Refresh Models</button>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Files</h2>
        <div class="body">
          <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Add files">
            <span class="big">Drop files here</span>
            <span>or click to browse · PDF, DOCX, MD, code, data</span>
          </div>
          <input name="files" id="fileInput" type="file" multiple style="display:none">
          <div id="fileChips"></div>
        </div>
      </div>

      <div class="card">
        <h2>Metadata &amp; options</h2>
        <div class="body">
          <div class="field-grid">
            <label class="field">Project
              <input name="project" value="MNEMOS" required>
            </label>
            <label class="field">Capability
              <input name="capability" placeholder="local research memory" required>
            </label>
            <label class="field">Status
              <select name="status">
                <option value="new">new</option>
                <option value="reviewed">reviewed</option>
                <option value="promising">promising</option>
                <option value="rejected">rejected</option>
                <option value="integrated">integrated</option>
              </select>
            </label>
            <label class="field">Tags
              <input name="tags" placeholder="workflow, pdf, github">
              <span class="hint">Comma-separated.</span>
            </label>
            <label class="field">Ollama model
              <select id="ollamaModelSelect" aria-label="Ollama model list"></select>
              <input name="ollama_model" id="ollamaModel" placeholder="type model name if not listed">
              <span class="hint">Only used when summarizing; you can type a model manually.</span>
            </label>
            <label class="field">Index batch size
              <input name="batch_size" type="number" min="1" value="25">
              <span class="hint">Smaller batches are slower but less likely to time out on large PDFs.</span>
            </label>
            <label class="field">Output packet path
              <input name="output" placeholder="docs/research/my_packet.md">
            </label>
          </div>
          <div class="inline" style="margin-top:14px;">
            <label class="check"><input name="summarize" type="checkbox" value="true" checked> Summarize with Ollama</label>
          </div>
        </div>
      </div>

      <div class="inline">
        <button class="btn-primary" type="submit" id="runIntake">Run Intake</button>
      </div>
    </form>

    <div class="card runlog">
      <h2>Run log</h2>
      <div class="body">
        <div class="statusline"><span class="pill" id="runPill"><span class="dot"></span>idle</span></div>
        <div class="msg" id="statusBox">Ready.</div>
        <dl class="kv" id="resultKv" hidden></dl>
        <details class="fold" id="rawWrap" hidden>
          <summary>Raw response</summary>
          <pre id="rawJson"></pre>
        </details>
      </div>
    </div>
  </div>
"""

INDEX_SCRIPT = """
const statusBox = document.getElementById('statusBox');
const runPill = document.getElementById('runPill');
const resultKv = document.getElementById('resultKv');
const rawWrap = document.getElementById('rawWrap');
const rawJson = document.getElementById('rawJson');
const modelSelect = document.getElementById('ollamaModelSelect');
const modelInput = document.getElementById('ollamaModel');
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const fileChips = document.getElementById('fileChips');
const runButton = document.getElementById('runIntake');

function setPill(el, kind, text) {
  el.className = 'pill' + (kind ? ' ' + kind : '');
  el.innerHTML = '<span class="dot"></span>' + text;
}

function setStatus(text, ok = null) {
  statusBox.textContent = text;
  if (ok === true) setPill(runPill, 'ok', 'done');
  else if (ok === false) setPill(runPill, 'bad', 'failed');
}

function renderChips() {
  fileChips.innerHTML = '';
  [...fileInput.files].forEach((file, index) => {
    const chip = document.createElement('div');
    chip.className = 'filechip';
    const name = document.createElement('span');
    name.className = 'fname';
    name.textContent = file.name;
    const meta = document.createElement('span');
    meta.className = 'fmeta';
    meta.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'x';
    remove.setAttribute('aria-label', 'Remove ' + file.name);
    remove.textContent = '\\u00d7';
    remove.addEventListener('click', () => {
      const dt = new DataTransfer();
      [...fileInput.files].forEach((f, i) => { if (i !== index) dt.items.add(f); });
      fileInput.files = dt.files;
      renderChips();
    });
    chip.append(name, meta, remove);
    fileChips.appendChild(chip);
  });
}

function addFiles(list) {
  const dt = new DataTransfer();
  [...fileInput.files].forEach((f) => dt.items.add(f));
  [...list].forEach((f) => dt.items.add(f));
  fileInput.files = dt.files;
  renderChips();
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener('change', renderChips);
['dragover', 'dragenter'].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach((evt) => dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

async function refreshModels() {
  const base = document.getElementById('ollamaBaseUrl').value;
  const res = await fetch('/api/ollama-models?ollama_base_url=' + encodeURIComponent(base));
  const data = await res.json();
  modelSelect.innerHTML = '';
  if (!data.ok) {
    setPill(document.getElementById('svcOllama'), 'bad', 'Ollama · unreachable');
    return;
  }
  data.models.forEach((model) => {
    const opt = document.createElement('option');
    opt.value = model.name;
    opt.textContent = model.name;
    modelSelect.appendChild(opt);
  });
  if (data.models.length && !modelInput.value) modelInput.value = data.models[0].name;
  setPill(document.getElementById('svcOllama'), 'ok', 'Ollama · ' + data.models.length + ' models');
}

async function testConnections(announce) {
  if (announce) setStatus('Testing connections...');
  const res = await fetch('/api/test-connection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mnemos_base_url: document.getElementById('mnemosBaseUrl').value,
      ollama_base_url: document.getElementById('ollamaBaseUrl').value,
    }),
  });
  const data = await res.json();
  const m = data.mnemos || {};
  const o = data.ollama || {};
  setPill(document.getElementById('svcMnemos'), m.ok ? 'ok' : 'bad', 'MNEMOS · ' + (m.ok ? (m.status || 'healthy') : 'unreachable'));
  setPill(document.getElementById('svcOllama'), o.ok ? 'ok' : 'bad', 'Ollama · ' + (o.ok ? o.model_count + ' models' : 'unreachable'));
  if (announce) {
    const problems = [];
    if (!m.ok) problems.push('MNEMOS: ' + (m.error || m.status || 'unreachable'));
    if (!o.ok) problems.push('Ollama: ' + (o.error || 'unreachable'));
    setStatus(data.ok ? 'Both connections healthy.' : problems.join('\\n'), data.ok);
  }
  return data;
}

function renderResult(data) {
  resultKv.innerHTML = '';
  const entries = [];
  const result = data.result || {};
  for (const [key, value] of Object.entries(result)) {
    if (['string', 'number', 'boolean'].includes(typeof value) && String(value).length <= 120) {
      entries.push([key, String(value)]);
    }
  }
  if (Array.isArray(data.uploaded_files)) entries.unshift(['uploaded files', String(data.uploaded_files.length)]);
  entries.forEach(([key, value]) => {
    const row = document.createElement('div');
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    row.append(dt, dd);
    resultKv.appendChild(row);
  });
  resultKv.hidden = entries.length === 0;
  rawJson.textContent = JSON.stringify(data, null, 2);
  rawWrap.hidden = false;
}

modelSelect.addEventListener('change', () => { modelInput.value = modelSelect.value; });
document.getElementById('refreshModels').addEventListener('click', refreshModels);
document.getElementById('testConnection').addEventListener('click', () => testConnections(true));

document.getElementById('intakeForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setStatus('Add at least one file before running intake.', false);
    return;
  }
  setPill(runPill, 'accent', 'running');
  setStatus('Running intake\\u2026 extract \\u2192 chunk \\u2192 index' + (event.target.summarize.checked ? ' \\u2192 summarize' : '') + '. Large PDFs can take a few minutes.');
  resultKv.hidden = true;
  rawWrap.hidden = true;
  runButton.disabled = true;
  try {
    const form = new FormData(event.target);
    form.set('ollama_model', modelInput.value || modelSelect.value || 'llama3.1');
    const res = await fetch('/api/intake', { method: 'POST', body: form });
    const data = await res.json();
    setStatus(data.ok ? 'Intake complete.' : 'Intake failed: ' + (data.error || (data.result && data.result.status) || 'see raw response'), data.ok);
    renderResult(data);
  } catch (err) {
    setStatus('Intake request failed: ' + err, false);
  } finally {
    runButton.disabled = false;
  }
});

refreshModels();
testConnections(false);
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
