"""Open WebUI/Ollama-compatible MNEMOS evidence proxy.

This local-only bridge lets chat front ends point at a familiar Ollama or
OpenAI-compatible endpoint while keeping MNEMOS as the evidence source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import requests
from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnemos_ollama_chat import (  # noqa: E402
    CLAIM_BOUNDARY,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaChatClient,
    normalize_base_url,
    run_query,
    run_query_stream,
)


class OllamaTagsClient(Protocol):
    def tags(self) -> dict[str, Any]:
        ...


class RequestsOllamaTagsClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, *, timeout_s: float = 10.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout_s = timeout_s

    def tags(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=max(0.1, self.timeout_s))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Ollama tags response must be a JSON object")
        return data


QueryRunner = Callable[..., dict[str, Any]]
StreamQueryRunner = Callable[..., Iterator[dict[str, Any]]]
DEFAULT_RECEIPT_DIR = ROOT / "logs" / "evidence_receipts"
# Statuses that produce an evidence receipt. "ollama_error" covers generation
# failures mid-stream, where evidence was already retrieved and partially used.
RECEIPT_STATUSES = {"ok", "no_evidence", "mnemos_error", "ollama_error"}


def _env_flag(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


LOGGER = logging.getLogger("mnemos.openwebui_proxy")

CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")


def verify_citations(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic, passive check of bracket citations against sent evidence.

    Shadow-observability annotation only: it records what the answer cited, it
    never blocks or alters the answer (R0 posture, not R1 enforcement).
    """
    available: set[int] = set()
    for citation in citations or []:
        index_value = citation.get("index")
        if index_value is None:
            continue
        try:
            available.add(int(index_value))
        except (TypeError, ValueError):
            continue
    cited: list[int] = []
    for match in CITATION_PATTERN.finditer(str(answer or "")):
        index = int(match.group(1))
        if index not in cited:
            cited.append(index)
    valid = [index for index in cited if index in available]
    invalid = [index for index in cited if index not in available]
    uncited = sorted(available - set(valid))
    if not available:
        verdict = "no_evidence_available"
    elif invalid:
        verdict = "cites_missing_evidence"
    elif not valid:
        verdict = "no_citations_in_answer"
    elif not uncited:
        verdict = "all_evidence_cited"
    else:
        verdict = "partial_evidence_cited"
    return {
        "cited_indices": cited,
        "invalid_indices": invalid,
        "uncited_evidence_indices": uncited,
        "evidence_count": len(available),
        "coverage": round(len(valid) / len(available), 4) if available else None,
        "verdict": verdict,
    }


def generation_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract generation-honesty fields from the final Ollama response."""
    response = result.get("ollama_response")
    if not isinstance(response, dict):
        response = {}
    done_reason = response.get("done_reason")
    return {
        "done_reason": done_reason,
        "truncated": done_reason == "length",
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def _usage_from_result(result: dict[str, Any]) -> dict[str, int]:
    info = generation_info(result)
    prompt_tokens = int(info.get("prompt_eval_count") or 0)
    completion_tokens = int(info.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _score_stats(citations: list[dict[str, Any]]) -> dict[str, Any] | None:
    scores: list[float] = []
    for citation in citations or []:
        score = citation.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return None
    return {
        "count": len(scores),
        "max": round(max(scores), 4),
        "min": round(min(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
    }


def receipt_content_hash(receipt: dict[str, Any]) -> str:
    """Tamper-evidence hash over the receipt's factual core."""
    core = {
        "receipt_id": receipt.get("receipt_id"),
        "created": receipt.get("created"),
        "query": receipt.get("query"),
        "answer": receipt.get("answer"),
        "evidence_block": receipt.get("evidence_block"),
        "citations": receipt.get("citations"),
    }
    payload = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part).strip()
    return str(content or "").strip()


def extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _content_to_text(message.get("content"))
            if text:
                return text
    return ""


FOOTER_MARKER = "---\n\nMNEMOS Evidence Used"


def strip_evidence_footer(text: str) -> str:
    """Remove the deterministic evidence footer from a prior assistant turn."""
    value = str(text or "")
    index = value.rfind(FOOTER_MARKER)
    if index == -1:
        return value.strip()
    return value[:index].strip()


def build_history_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Sanitize prior turns for forwarding to Ollama.

    Evidence footers are stripped from assistant turns so the model never sees
    (or imitates) its own receipt boilerplate, and the tail is capped so long
    chats cannot crowd out the evidence block.
    """
    if not _env_flag("MNEMOS_PROXY_HISTORY"):
        return []
    history: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _content_to_text(message.get("content"))
        if role == "assistant":
            text = strip_evidence_footer(text)
        if text:
            history.append({"role": str(role), "content": text})
    max_turns = int(os.getenv("MNEMOS_PROXY_HISTORY_MAX_TURNS", "8"))
    if max_turns <= 0:
        return []
    return history[-max_turns:]


def split_query_and_history(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Return the latest user query and the sanitized turns that precede it."""
    for position in range(len(messages) - 1, -1, -1):
        message = messages[position]
        if message.get("role") == "user":
            text = _content_to_text(message.get("content"))
            if text:
                return text, build_history_messages(messages[:position])
    return "", []


def _model_ids_from_tags(tags: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for model in tags.get("models") or []:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if name:
                ids.append(str(name))
    return ids


def normalize_openwebui_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    prefixes = [
        item.strip()
        for item in os.getenv("MNEMOS_PROXY_MODEL_PREFIXES", "mnemos").split(",")
        if item.strip()
    ]
    for prefix in prefixes:
        for separator in (".", "/"):
            marker = f"{prefix}{separator}"
            if text.startswith(marker):
                return text[len(marker) :]
    return text


def should_append_footer(payload: dict[str, Any]) -> bool:
    if os.getenv("MNEMOS_PROXY_FOOTER", "on").strip().lower() in {"0", "false", "no", "off"}:
        return False
    query = extract_latest_user_text(payload.get("messages") or [])
    # Open WebUI sends hidden title/tag/task prompts through the same provider.
    # Keep those responses clean. If multi-turn forwarding is added later,
    # strip this deterministic footer from prior assistant messages first.
    if query.lstrip().startswith("### Task:"):
        return False
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("task"):
        return False
    return True


def build_evidence_footer(
    result: dict[str, Any],
    *,
    receipt_url: str | None,
    answer_text: str = "",
) -> str:
    """Build the footer block on its own, without the answer body."""
    citations = result.get("citations") or []
    lines = ["---", "", "MNEMOS Evidence Used", ""]
    if citations:
        lines.append("```text")
        for citation in citations:
            index = citation.get("index", "?")
            source = citation.get("source", "unknown")
            score = citation.get("score", 0.0)
            engram_id = citation.get("engram_id") or "unknown"
            try:
                score_text = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_text = str(score)
            lines.append(f"[{index}] source={source}")
            lines.append(f"    score={score_text}")
            lines.append(f"    engram_id={engram_id}")
        lines.append("```")
    else:
        if result.get("status") == "mnemos_error":
            lines.append("MNEMOS retrieval failed - answer withheld by the MNEMOS proxy.")
        else:
            lines.append("No MNEMOS evidence retrieved - answer withheld by the MNEMOS proxy.")
        warning = str(result.get("warning") or "").strip()
        if warning and warning != answer_text:
            lines.append(f"Warning: {warning}")

    if generation_info(result)["truncated"]:
        lines.extend(
            [
                "",
                "Warning: answer stopped at the token limit (done_reason=length); "
                "citations may be incomplete.",
            ]
        )

    if receipt_url:
        lines.extend(["", f"MNEMOS Evidence Receipt: {receipt_url}"])
    lines.extend(["", f"Boundary: {result.get('claim_boundary') or CLAIM_BOUNDARY}"])
    return "\n".join(lines)


def append_evidence_footer(answer: str, result: dict[str, Any], *, receipt_url: str | None) -> str:
    # Markdown discipline: a "---" line directly under paragraph text renders
    # as a setext heading, so the footer must be separated by a blank line, and
    # the citation lines live in a fenced block to keep their line structure.
    base = str(answer or result.get("warning") or "").strip()
    footer = build_evidence_footer(result, receipt_url=receipt_url, answer_text=base)
    return f"{base}\n\n{footer}" if base else footer


def _safe_receipt_id(receipt_id: str) -> str:
    return "".join(ch for ch in receipt_id if ch.isalnum() or ch in {"-", "_"})


def _receipt_path(receipt_dir: Path, receipt_id: str) -> Path:
    safe_id = _safe_receipt_id(receipt_id)
    if not safe_id:
        raise ValueError("empty receipt id")
    return receipt_dir / f"{safe_id}.json"


def write_evidence_receipt(
    receipt_dir: Path,
    *,
    receipt_id: str,
    created: int,
    query: str,
    requested_model: str,
    actual_model: str,
    answer: str,
    result: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    citations = result.get("citations") or []
    receipt = {
        "receipt_id": receipt_id,
        "created": created,
        "query": query,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "answer": answer,
        "status": result.get("status"),
        "citations": citations,
        "evidence_block": result.get("evidence_block") or "",
        "retrieval_metadata": result.get("retrieval_metadata") or {},
        "claim_boundary": result.get("claim_boundary") or CLAIM_BOUNDARY,
        "warning": result.get("warning"),
        "citation_check": verify_citations(answer, citations),
        "generation": generation_info(result),
        "score_stats": _score_stats(citations),
    }
    receipt["content_hash"] = receipt_content_hash(receipt)
    path = _receipt_path(receipt_dir, receipt_id)
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    # Receipts are proof artifacts: overflow is archived, never silently deleted.
    max_receipts = int(os.getenv("MNEMOS_EVIDENCE_RECEIPT_MAX_FILES", "500"))
    receipt_files = sorted(receipt_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
    overflow = receipt_files[: max(0, len(receipt_files) - max_receipts)]
    if overflow:
        archive_dir = receipt_dir / "archive"
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            for stale in overflow:
                stale.rename(archive_dir / stale.name)
            LOGGER.info(
                "archived %d evidence receipt(s) past MNEMOS_EVIDENCE_RECEIPT_MAX_FILES=%d to %s",
                len(overflow),
                max_receipts,
                archive_dir,
            )
        except OSError as exc:
            LOGGER.warning("failed to archive stale evidence receipts: %s", exc)
    return path


# The receipt page is fully self-contained (inline CSS/JS, no CDN) so it keeps
# working on air-gapped/local-only deployments. All dynamic content is drawn
# client-side from the embedded receipt JSON; the server only escapes and
# injects that JSON, which keeps the template free of per-field escaping bugs.
_RECEIPT_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNEMOS Evidence Receipt</title>
  <style>
    :root {
      --bg: #f6f7f9; --card: #ffffff; --ink: #1f2937; --muted: #6b7280;
      --line: #e5e7eb; --green: #16a34a; --green-soft: #dcfce7;
      --amber: #d97706; --amber-soft: #fef3c7; --red: #dc2626; --red-soft: #fee2e2;
      --gray-soft: #f3f4f6; --blue: #2563eb; --blue-soft: #dbeafe;
    }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--ink); }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 32px 0 12px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px 24px; font-size: 13px; }
    .meta-grid dt { color: var(--muted); }
    .meta-grid dd { margin: 2px 0 8px; overflow-wrap: anywhere; font-family: ui-monospace, monospace; font-size: 12px; }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .pill.ok, .pill.all_evidence_cited { background: var(--green-soft); color: var(--green); }
    .pill.partial_evidence_cited { background: var(--amber-soft); color: var(--amber); }
    .pill.no_citations_in_answer, .pill.cites_missing_evidence,
    .pill.mnemos_error, .pill.ollama_error { background: var(--red-soft); color: var(--red); }
    .pill.no_evidence, .pill.no_evidence_available { background: var(--gray-soft); color: var(--muted); }
    .verdict-row { display: flex; flex-wrap: wrap; align-items: center; gap: 18px; }
    .coverage-bar { flex: 1 1 220px; height: 10px; border-radius: 999px; background: var(--gray-soft); overflow: hidden; min-width: 160px; }
    .coverage-bar > div { height: 100%; background: var(--green); }
    .counts { display: flex; gap: 14px; font-size: 13px; color: var(--muted); }
    .counts b { color: var(--ink); }
    .legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--muted); margin-top: 10px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 22px; height: 0; border-top: 3px solid; border-radius: 2px; }
    svg text { font-family: system-ui, sans-serif; }
    .node rect { cursor: pointer; }
    .node.dim, .edge.dim { opacity: 0.18; }
    .answer-body { font-size: 15px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
    .cite-chip { display: inline-block; padding: 0 7px; border-radius: 6px; font-size: 12px; font-weight: 700;
                 font-family: ui-monospace, monospace; cursor: pointer; vertical-align: 1px; }
    .cite-chip.valid { background: var(--green-soft); color: var(--green); }
    .cite-chip.invalid { background: var(--red-soft); color: var(--red); }
    .ev-card { border: 1px solid var(--line); border-left-width: 4px; border-radius: 10px; background: var(--card);
               padding: 12px 16px; margin-bottom: 12px; scroll-margin-top: 16px; }
    .ev-card.cited { border-left-color: var(--green); }
    .ev-card.uncited { border-left-color: #cbd5e1; }
    .ev-card.flash { box-shadow: 0 0 0 3px var(--blue-soft); }
    .ev-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; font-size: 13px; }
    .ev-index { font-family: ui-monospace, monospace; font-weight: 700; }
    .ev-source { font-weight: 600; overflow-wrap: anywhere; }
    .ev-id { color: var(--muted); font-family: ui-monospace, monospace; font-size: 11px; }
    .score-wrap { display: inline-flex; align-items: center; gap: 6px; margin-left: auto; }
    .score-bar { width: 90px; height: 6px; border-radius: 999px; background: var(--gray-soft); overflow: hidden; }
    .score-bar > div { height: 100%; background: var(--blue); }
    .score-num { font-family: ui-monospace, monospace; font-size: 12px; }
    details.ev-text { margin-top: 8px; }
    details.ev-text summary { cursor: pointer; font-size: 12px; color: var(--muted); }
    details.ev-text pre, pre.plain { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--gray-soft);
                     padding: 12px; border-radius: 6px; font-size: 12.5px; margin: 8px 0 0; }
    pre.plain { margin: 0; }
    a { color: var(--blue); }
    .topbar { display: flex; align-items: center; gap: 16px; }
    .topbar h1 { flex: 1; }
    .export-btn { border: 1px solid var(--line); background: var(--card); color: var(--ink);
                  padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; }
    .export-btn:hover { border-color: var(--blue); color: var(--blue); }
    .coverage-note { font-size: 12px; color: var(--muted); margin: 10px 0 0; line-height: 1.5; }
    .print-only { display: none; }
    @media print {
      .no-print { display: none !important; }
      .print-only { display: block; font-size: 11px; color: #555; margin-top: 6px; }
      body { background: #fff; }
      .wrap { max-width: none; padding: 0; }
      .card, .ev-card { break-inside: avoid; box-shadow: none; }
      a { color: inherit; text-decoration: none; }
      details.ev-text summary { display: none; }
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <h1>MNEMOS Evidence Receipt</h1>
    <button type="button" id="export-pdf" class="export-btn no-print" title="Export this receipt as a PDF via the browser print dialog">Export PDF</button>
  </div>
  <div class="print-only" id="print-stamp"></div>
  <div id="page"></div>
</div>
<script type="application/json" id="receipt-data">__RECEIPT_JSON__</script>
<script>
(function () {
  var receipt = JSON.parse(document.getElementById("receipt-data").textContent);
  var citations = receipt.citations || [];
  var check = receipt.citation_check || {};
  var citedSet = {};
  (check.cited_indices || []).forEach(function (i) { citedSet[i] = true; });
  var invalidSet = {};
  (check.invalid_indices || []).forEach(function (i) { invalidSet[i] = true; });

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function baseName(p) {
    var parts = String(p || "unknown").split(/[\\/]/);
    return parts[parts.length - 1] || String(p);
  }
  function trunc(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "\\u2026" : s; }

  // ---- evidence excerpts, parsed from the block actually sent to the model ----
  var excerpts = {};
  var block = String(receipt.evidence_block || "");
  var re = /\\[(\\d{1,3})\\] source=[^\\n]*\\n\\n?([\\s\\S]*?)(?=\\n\\n\\[\\d{1,3}\\] source=|$)/g;
  var m;
  while ((m = re.exec(block)) !== null) { excerpts[parseInt(m[1], 10)] = m[2].trim(); }

  // ---- verdict banner ----
  var verdict = check.verdict || "unknown";
  var coverage = typeof check.coverage === "number" ? check.coverage : null;
  var validCount = (check.cited_indices || []).filter(function (i) { return !invalidSet[i]; }).length;
  var html = "";
  html += '<div class="card" style="margin-top:14px">';
  html += '<div class="verdict-row">';
  html += '<span class="pill ' + esc(verdict) + '">' + esc(verdict.replace(/_/g, " ")) + "</span>";
  if (coverage !== null) {
    html += '<div class="coverage-bar" title="citation coverage"><div style="width:' + Math.round(coverage * 100) + '%"></div></div>';
    html += "<b>" + Math.round(coverage * 100) + "% coverage</b>";
  }
  html += '<div class="counts">';
  html += "<span><b>" + citations.length + "</b> retrieved</span>";
  html += "<span><b>" + validCount + "</b> cited</span>";
  html += "<span><b>" + (check.uncited_evidence_indices || []).length + "</b> uncited</span>";
  if ((check.invalid_indices || []).length) {
    html += '<span style="color:var(--red)"><b>' + check.invalid_indices.length + "</b> invalid</span>";
  }
  html += "</div></div>";
  if (coverage !== null) {
    html += '<p class="coverage-note">Coverage = evidence chunks the answer validly cited \\u00f7 chunks sent to the model (' +
      validCount + " of " + (check.evidence_count || citations.length) + " here). It measures how much of the retrieved " +
      "evidence the answer actually drew on \\u2014 it is <b>not</b> a correctness or confidence score. Low coverage is " +
      "normal when retrieved chunks overlap or duplicate each other; the verdict above and any red edges in the graph " +
      "are the integrity signals.</p>";
  }

  // ---- provenance graph ----
  var invalidCited = check.invalid_indices || [];
  var rows = citations.map(function (c) { return { kind: "evidence", c: c }; })
    .concat(invalidCited.map(function (i) { return { kind: "missing", index: i }; }));
  if (rows.length) {
    var W = 1120, rowH = 74, topPad = 46, nodeW = 300, nodeH = 56;
    var H = Math.max(230, topPad + rows.length * rowH + 20);
    var evX = (W - nodeW) / 2, qX = 20, aX = W - 220 - 20;
    var midY = topPad + ((rows.length - 1) * rowH) / 2 + nodeH / 2;
    var scores = citations.map(function (c) { return typeof c.score === "number" ? c.score : 0; });
    var maxS = Math.max.apply(null, scores.concat([0.0001]));
    var minS = Math.min.apply(null, scores.concat([maxS]));
    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" style="width:100%;height:auto;margin-top:18px" role="img" aria-label="provenance graph">';
    function edge(x1, y1, x2, y2, stroke, width, dash, cls, title) {
      var mx = (x1 + x2) / 2;
      return '<path class="edge ' + cls + '" d="M' + x1 + " " + y1 + " C" + mx + " " + y1 + ", " + mx + " " + y2 + ", " + x2 + " " + y2 +
        '" fill="none" stroke="' + stroke + '" stroke-width="' + width + '"' +
        (dash ? ' stroke-dasharray="' + dash + '"' : "") + "><title>" + esc(title) + "</title></path>";
    }
    rows.forEach(function (row, r) {
      var y = topPad + r * rowH + nodeH / 2;
      if (row.kind === "evidence") {
        var s = typeof row.c.score === "number" ? row.c.score : 0;
        var norm = maxS === minS ? 1 : (s - minS) / (maxS - minS);
        svg += edge(qX + 200, midY, evX, y, "#94a3b8", (1.5 + 3 * norm).toFixed(1), null,
          "e-ret e-" + row.c.index, "retrieved, score " + s);
        if (citedSet[row.c.index]) {
          svg += edge(evX + nodeW, y, aX, midY, "var(--green)", 2.5, null,
            "e-cite e-" + row.c.index, "cited as [" + row.c.index + "] in the answer");
        } else {
          svg += edge(evX + nodeW, y, aX, midY, "#cbd5e1", 1.5, "5 5",
            "e-uncite e-" + row.c.index, "retrieved but not cited");
        }
      } else {
        svg += edge(evX + nodeW, y, aX, midY, "var(--red)", 2, "3 4",
          "e-invalid e-" + row.index, "answer cites [" + row.index + "] but no such evidence was sent");
      }
    });
    // column headers
    svg += '<text x="' + (qX + 100) + '" y="20" text-anchor="middle" font-size="11" fill="var(--muted)" letter-spacing="1">QUERY</text>';
    svg += '<text x="' + (evX + nodeW / 2) + '" y="20" text-anchor="middle" font-size="11" fill="var(--muted)" letter-spacing="1">EVIDENCE SENT TO MODEL</text>';
    svg += '<text x="' + (aX + 110) + '" y="20" text-anchor="middle" font-size="11" fill="var(--muted)" letter-spacing="1">ANSWER</text>';
    function nodeRect(x, y, w, h, fill, stroke, cls, dash) {
      return '<rect class="' + cls + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h +
        '" rx="9" fill="' + fill + '" stroke="' + stroke + '"' + (dash ? ' stroke-dasharray="4 3"' : "") + "/>";
    }
    svg += '<g class="node">' + nodeRect(qX, midY - nodeH / 2, 200, nodeH, "var(--blue-soft)", "var(--blue)", "") +
      '<text x="' + (qX + 100) + '" y="' + (midY + 4) + '" text-anchor="middle" font-size="12">' +
      esc(trunc(receipt.query, 30)) + "<title>" + esc(receipt.query) + "</title></text></g>";
    rows.forEach(function (row, r) {
      var y = topPad + r * rowH;
      if (row.kind === "evidence") {
        var cited = !!citedSet[row.c.index];
        svg += '<g class="node ev-node" data-index="' + row.c.index + '">' +
          nodeRect(evX, y, nodeW, nodeH, cited ? "var(--green-soft)" : "#fff", cited ? "var(--green)" : "#cbd5e1", "") +
          '<text x="' + (evX + 14) + '" y="' + (y + 23) + '" font-size="12" font-weight="700">[' + row.c.index + "] " +
          esc(trunc(baseName(row.c.source), 34)) + "</text>" +
          '<text x="' + (evX + 14) + '" y="' + (y + 42) + '" font-size="11" fill="var(--muted)">score ' +
          esc(row.c.score) + " \\u00b7 " + esc(trunc(row.c.engram_id || "", 30)) + "</text>" +
          "<title>" + esc(row.c.source) + "</title></g>";
      } else {
        svg += '<g class="node">' + nodeRect(evX, y, nodeW, nodeH, "var(--red-soft)", "var(--red)", "", true) +
          '<text x="' + (evX + 14) + '" y="' + (y + 33) + '" font-size="12" fill="var(--red)">[' + row.index +
          "] cited but never sent</text></g>";
      }
    });
    svg += '<g class="node">' + nodeRect(aX, midY - nodeH / 2, 220, nodeH, "#fff", "var(--ink)", "") +
      '<text x="' + (aX + 110) + '" y="' + (midY + 4) + '" text-anchor="middle" font-size="12">' +
      esc(trunc(receipt.answer || "(no answer)", 32)) + "</text></g>";
    svg += "</svg>";
    html += svg;
    html += '<div class="legend">' +
      '<span><span class="swatch" style="border-color:#94a3b8"></span>retrieved (width = score)</span>' +
      '<span><span class="swatch" style="border-color:var(--green)"></span>cited in answer</span>' +
      '<span><span class="swatch" style="border-color:#cbd5e1;border-top-style:dashed"></span>retrieved, uncited</span>' +
      (invalidCited.length ? '<span><span class="swatch" style="border-color:var(--red);border-top-style:dashed"></span>cited but never sent</span>' : "") +
      "</div>";
  }
  html += "</div>";

  // ---- query + answer ----
  html += "<h2>Query</h2><div class='card'><div class='answer-body'>" + esc(receipt.query) + "</div></div>";
  html += "<h2>Answer</h2><div class='card'><div class='answer-body'>" +
    esc(receipt.answer || "(no answer recorded)").replace(/\\[(\\d{1,3})\\]/g, function (all, n) {
      var idx = parseInt(n, 10);
      var cls = invalidSet[idx] ? "invalid" : "valid";
      var title = invalidSet[idx] ? "cites evidence that was never sent" : "click to jump to evidence [" + idx + "]";
      return '<span class="cite-chip ' + cls + '" data-index="' + idx + '" title="' + title + '">[' + idx + "]</span>";
    }) + "</div></div>";

  // ---- evidence cards ----
  html += "<h2>Evidence (" + citations.length + " chunks sent to the model)</h2>";
  citations.forEach(function (c) {
    var cited = !!citedSet[c.index];
    var s = typeof c.score === "number" ? c.score : 0;
    html += '<div class="ev-card ' + (cited ? "cited" : "uncited") + '" id="evidence-' + c.index + '">' +
      '<div class="ev-head"><span class="ev-index">[' + c.index + "]</span>" +
      '<span class="ev-source" title="' + esc(c.source) + '">' + esc(baseName(c.source)) + "</span>" +
      '<span class="pill ' + (cited ? "ok" : "no_evidence") + '">' + (cited ? "cited" : "uncited") + "</span>" +
      '<span class="ev-id">' + esc(c.engram_id || "") + "</span>" +
      '<span class="score-wrap"><span class="score-bar"><div style="width:' + Math.round(Math.min(1, s) * 100) +
      '%"></div></span><span class="score-num">' + esc(c.score) + "</span></span></div>";
    var text = excerpts[c.index];
    if (text) {
      html += '<details class="ev-text"' + (cited ? " open" : "") + "><summary>excerpt sent to model</summary><pre>" +
        esc(text) + "</pre></details>";
    }
    html += "</div>";
  });

  // ---- provenance / integrity metadata ----
  var rm = receipt.retrieval_metadata || {};
  var fp = rm.retrieval_fingerprint || {};
  var gen = receipt.generation || {};
  html += "<h2>Bounded-Truth Provenance</h2><div class='card'><dl class='meta-grid'>" +
    "<div><dt>Receipt</dt><dd>" + esc(receipt.receipt_id) + "</dd></div>" +
    "<div><dt>Status</dt><dd><span class='pill " + esc(receipt.status || "") + "'>" + esc(receipt.status || "") + "</span></dd></div>" +
    "<div><dt>Model</dt><dd>" + esc(receipt.requested_model) + "</dd></div>" +
    "<div><dt>Integrity hash</dt><dd>" + esc(receipt.content_hash || "not recorded") + "</dd></div>" +
    "<div><dt>Collection snapshot</dt><dd>" + esc(fp.collection_snapshot || "n/a") + "</dd></div>" +
    "<div><dt>Embedding model</dt><dd>" + esc(fp.embedding_model_name || "n/a") + "</dd></div>" +
    "<div><dt>Retrieval profile</dt><dd>" + esc(fp.retrieval_profile || "n/a") + "</dd></div>" +
    "<div><dt>Retrieval latency</dt><dd>" + esc(rm.latency_s != null ? rm.latency_s + " s" : "n/a") + "</dd></div>" +
    "<div><dt>Generation</dt><dd>done_reason=" + esc(gen.done_reason) + (gen.truncated ? " (TRUNCATED)" : "") +
    " \\u00b7 prompt=" + esc(gen.prompt_eval_count) + " tok \\u00b7 completion=" + esc(gen.eval_count) + " tok</dd></div>" +
    "</dl>" +
    '<details class="ev-text"><summary>raw verification JSON</summary><pre>' +
    esc(JSON.stringify({ citation_check: receipt.citation_check, generation: receipt.generation, score_stats: receipt.score_stats }, null, 2)) +
    "</pre></details>" +
    '<p style="font-size:12px;margin:10px 0 0"><a href="' + esc(receipt.receipt_id) + '.json">raw receipt JSON</a></p>' +
    "</div>";

  html += "<h2>Boundary</h2><div class='card'><pre class='plain'>" + esc(receipt.claim_boundary || "") + "</pre></div>";

  document.getElementById("page").innerHTML = html;

  // ---- interactivity: chips + graph nodes jump to evidence; hover isolates a path ----
  function jump(idx) {
    var el = document.getElementById("evidence-" + idx);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(function () { el.classList.remove("flash"); }, 1600);
  }
  document.querySelectorAll(".cite-chip.valid, .ev-node").forEach(function (el) {
    el.addEventListener("click", function () { jump(el.getAttribute("data-index")); });
  });
  var graphEls = document.querySelectorAll(".ev-node, .edge");
  document.querySelectorAll(".ev-node").forEach(function (node) {
    var idx = node.getAttribute("data-index");
    node.addEventListener("mouseenter", function () {
      graphEls.forEach(function (el) {
        var keep = el === node || el.classList.contains("e-" + idx);
        el.classList.toggle("dim", !keep);
      });
    });
    node.addEventListener("mouseleave", function () {
      graphEls.forEach(function (el) { el.classList.remove("dim"); });
    });
  });

  // ---- PDF export via the browser print pipeline (works for Ctrl+P too) ----
  // For evidentiary use every excerpt must appear in the PDF, so all <details>
  // are opened before printing and restored afterwards.
  var reopenState = null;
  function preparePrint() {
    var stamp = document.getElementById("print-stamp");
    stamp.textContent = "Receipt " + (receipt.receipt_id || "") + " \\u00b7 " +
      (receipt.content_hash || "no integrity hash") + " \\u00b7 exported " + new Date().toISOString() +
      " \\u00b7 verify against the stored receipt JSON before relying on this document";
    reopenState = [];
    document.querySelectorAll("details").forEach(function (d) {
      reopenState.push([d, d.open]);
      d.open = true;
    });
  }
  function restorePrint() {
    (reopenState || []).forEach(function (pair) { pair[0].open = pair[1]; });
    reopenState = null;
  }
  window.addEventListener("beforeprint", preparePrint);
  window.addEventListener("afterprint", restorePrint);
  document.getElementById("export-pdf").addEventListener("click", function () { window.print(); });
})();
</script>
</body>
</html>"""


def render_evidence_receipt_html(receipt: dict[str, Any]) -> str:
    # "</" must be escaped so receipt content can never close the script tag.
    receipt_json = json.dumps(receipt, ensure_ascii=False).replace("</", "<\\/")
    return _RECEIPT_PAGE_TEMPLATE.replace("__RECEIPT_JSON__", receipt_json)


def _public_receipt_base_url() -> str:
    configured = os.getenv("MNEMOS_EVIDENCE_PUBLIC_BASE_URL")
    if configured:
        return configured.rstrip("/")
    host = request.host.split(":", 1)
    port = f":{host[1]}" if len(host) == 2 else ""
    return f"http://127.0.0.1{port}"


def _openai_error(message: str, *, status: int = 400):
    return (
        jsonify(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            }
        ),
        status,
    )


def _finalize_result(
    *,
    result: dict[str, Any],
    payload: dict[str, Any],
    receipt_dir: Path,
    receipt_id: str,
    created: int,
    query: str,
    requested_model: str,
    model: str,
    receipt_base_url: str,
) -> str:
    """Write the receipt for a completed result and return the final answer."""
    raw_answer = str(result.get("answer") or result.get("warning") or "")
    receipt_url = None
    if result.get("status") in RECEIPT_STATUSES:
        write_evidence_receipt(
            receipt_dir,
            receipt_id=receipt_id,
            created=created,
            query=query,
            requested_model=requested_model,
            actual_model=model,
            answer=raw_answer,
            result=result,
        )
        receipt_url = f"{receipt_base_url}/evidence/{receipt_id}"
    if should_append_footer(payload):
        return append_evidence_footer(raw_answer, result, receipt_url=receipt_url)
    return raw_answer


def _finalize_stream_suffix(
    *,
    result: dict[str, Any],
    receipt_dir: Path,
    receipt_id: str,
    created: int,
    query: str,
    requested_model: str,
    model: str,
    receipt_base_url: str,
    footer_enabled: bool,
) -> str:
    """Write the receipt after a live stream and return the trailing text.

    The answer body has already been streamed, so only the footer (and any
    generation-failure warning) remains to be emitted.
    """
    raw_answer = str(result.get("answer") or "")
    receipt_url = None
    if result.get("status") in RECEIPT_STATUSES:
        write_evidence_receipt(
            receipt_dir,
            receipt_id=receipt_id,
            created=created,
            query=query,
            requested_model=requested_model,
            actual_model=model,
            answer=raw_answer,
            result=result,
        )
        receipt_url = f"{receipt_base_url}/evidence/{receipt_id}"
    parts: list[str] = []
    if result.get("status") not in {None, "ok"}:
        warning = str(result.get("warning") or "").strip()
        if warning:
            parts.append(f"[MNEMOS proxy] {warning}")
    if footer_enabled:
        parts.append(
            build_evidence_footer(result, receipt_url=receipt_url, answer_text=raw_answer)
        )
    return "\n\n".join(parts)


def _blocking_stream_adapter(runner: QueryRunner) -> StreamQueryRunner:
    """Adapt a blocking query runner to the streaming event protocol."""

    def stream_runner(**kwargs: Any) -> Iterator[dict[str, Any]]:
        result = runner(**kwargs)
        if result.get("status") != "ok":
            yield {"event": "done", "result": result}
            return
        yield {"event": "retrieval", "result": result}
        answer = str(result.get("answer") or "")
        if answer:
            yield {"event": "delta", "content": answer}
        yield {"event": "done", "result": result}

    return stream_runner


def _openai_live_stream_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    retrieval_result: dict[str, Any],
    events: Iterator[dict[str, Any]],
    finalize: Callable[[dict[str, Any]], str],
) -> Response:
    def chunk(
        delta: dict[str, Any],
        meta: dict[str, Any],
        finish: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> str:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            "mnemos": meta,
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def event_stream():
        meta = _mnemos_meta(retrieval_result)
        yield chunk({"role": "assistant"}, meta)
        done_result = retrieval_result
        streamed_any = False
        for event in events:
            if event.get("event") == "delta" and event.get("content"):
                streamed_any = True
                yield chunk({"content": event["content"]}, meta)
            elif event.get("event") == "done":
                done_result = event.get("result") or done_result
        suffix = finalize(done_result)
        final_meta = _mnemos_meta(done_result)
        if suffix:
            text = f"\n\n{suffix}" if streamed_any else suffix
            yield chunk({"content": text}, final_meta)
        yield chunk({}, final_meta, finish="stop", usage=_usage_from_result(done_result))
        yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


def _ollama_live_stream_response(
    *,
    model: str,
    retrieval_result: dict[str, Any],
    events: Iterator[dict[str, Any]],
    finalize: Callable[[dict[str, Any]], str],
) -> Response:
    def line(
        content: str,
        done: bool,
        meta: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> str:
        body = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": done,
            **(extra or {}),
            "mnemos": meta,
        }
        return json.dumps(body, ensure_ascii=False) + "\n"

    def lines():
        meta = _mnemos_meta(retrieval_result)
        done_result = retrieval_result
        streamed_any = False
        for event in events:
            if event.get("event") == "delta" and event.get("content"):
                streamed_any = True
                yield line(event["content"], False, meta)
            elif event.get("event") == "done":
                done_result = event.get("result") or done_result
        suffix = finalize(done_result)
        final_meta = _mnemos_meta(done_result)
        if suffix:
            text = f"\n\n{suffix}" if streamed_any else suffix
            yield line(text, False, final_meta)
        yield line("", True, final_meta, extra=_generation_passthrough(done_result))

    return Response(lines(), mimetype="application/x-ndjson")


def _mnemos_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "citations": result.get("citations") or [],
        "claim_boundary": result.get("claim_boundary") or CLAIM_BOUNDARY,
        "warning": result.get("warning"),
    }


def _openai_chat_completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_from_result(result),
        "mnemos": _mnemos_meta(result),
    }


def _openai_stream_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> Response:
    metadata = _mnemos_meta(result)

    def events():
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            "mnemos": metadata,
        }
        yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"
        if answer:
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
                "mnemos": metadata,
            }
            yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
        stop_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": _usage_from_result(result),
            "mnemos": metadata,
        }
        yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(events(), mimetype="text/event-stream")


def _generation_passthrough(result: dict[str, Any]) -> dict[str, Any]:
    """Ollama-native generation fields worth forwarding when present."""
    info = generation_info(result)
    fields: dict[str, Any] = {}
    if info["done_reason"] is not None:
        fields["done_reason"] = info["done_reason"]
    if info["prompt_eval_count"] is not None:
        fields["prompt_eval_count"] = info["prompt_eval_count"]
    if info["eval_count"] is not None:
        fields["eval_count"] = info["eval_count"]
    return fields


def _ollama_chat_response(
    *,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": answer},
        "done": True,
        **_generation_passthrough(result),
        "mnemos": _mnemos_meta(result),
    }


def _ollama_stream_response(
    *,
    model: str,
    answer: str,
    result: dict[str, Any],
) -> Response:
    metadata = _mnemos_meta(result)

    def lines():
        if answer:
            yield json.dumps(
                {
                    "model": model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": answer},
                    "done": False,
                    "mnemos": metadata,
                },
                ensure_ascii=False,
            ) + "\n"
        yield json.dumps(
            {
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                **_generation_passthrough(result),
                "mnemos": metadata,
            },
            ensure_ascii=False,
        ) + "\n"

    return Response(lines(), mimetype="application/x-ndjson")


def create_app(
    *,
    query_runner: QueryRunner = run_query,
    stream_query_runner: StreamQueryRunner | None = None,
    ollama_tags_client: OllamaTagsClient | None = None,
    ollama_base_url: str | None = None,
    receipt_dir: str | Path | None = None,
) -> Flask:
    app = Flask(__name__)
    if stream_query_runner is None:
        # Injected blocking runners (tests, custom wiring) keep working: their
        # single result is replayed through the streaming event protocol.
        stream_query_runner = (
            run_query_stream
            if query_runner is run_query
            else _blocking_stream_adapter(query_runner)
        )
    resolved_ollama_url = normalize_base_url(
        ollama_base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )
    tags_client = ollama_tags_client or RequestsOllamaTagsClient(resolved_ollama_url)

    app.config["CLAIM_BOUNDARY"] = CLAIM_BOUNDARY
    app.config["OLLAMA_BASE_URL"] = resolved_ollama_url
    resolved_receipt_dir = Path(
        receipt_dir or os.getenv("MNEMOS_EVIDENCE_RECEIPT_DIR", str(DEFAULT_RECEIPT_DIR))
    )
    app.config["MNEMOS_EVIDENCE_RECEIPT_DIR"] = str(resolved_receipt_dir)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "service": "mnemos-ollama-openwebui-proxy",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    @app.get("/api/tags")
    def api_tags():
        return jsonify(tags_client.tags())

    @app.get("/v1/api/tags")
    def v1_api_tags():
        return jsonify(tags_client.tags())

    @app.get("/api/ps")
    @app.get("/v1/api/ps")
    def api_ps():
        return jsonify({"models": []})

    @app.get("/api/version")
    @app.get("/v1/api/version")
    def api_version():
        return jsonify({"version": "mnemos-proxy"})

    @app.get("/evidence/<receipt_id>.json")
    def evidence_receipt_json(receipt_id: str):
        path = _receipt_path(resolved_receipt_dir, receipt_id)
        if not path.exists():
            return jsonify({"error": "receipt_not_found"}), 404
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/evidence/<receipt_id>")
    def evidence_receipt(receipt_id: str):
        path = _receipt_path(resolved_receipt_dir, receipt_id)
        if not path.exists():
            return "MNEMOS evidence receipt not found.", 404
        receipt = json.loads(path.read_text(encoding="utf-8"))
        return Response(render_evidence_receipt_html(receipt), mimetype="text/html")

    @app.get("/v1/models")
    def v1_models():
        models = [
            {"id": model_id, "object": "model", "owned_by": "ollama"}
            for model_id in _model_ids_from_tags(tags_client.tags())
        ]
        return jsonify({"object": "list", "data": models})

    def _chat_run_kwargs(payload: dict[str, Any], *, temperature: float, num_predict: int):
        """Shared request parsing for the OpenAI and Ollama chat shapes."""
        messages = payload.get("messages") or []
        query, history = split_query_and_history(messages)
        requested_model = str(payload.get("model") or DEFAULT_MODEL)
        model = normalize_openwebui_model_id(requested_model)
        run_kwargs: dict[str, Any] = dict(
            query=query,
            model=model,
            top_k=int(payload.get("mnemos_top_k") or os.getenv("MNEMOS_PROXY_TOP_K", "5")),
            retrieval_mode=str(payload.get("mnemos_retrieval_mode") or "semantic"),
            fusion_policy=str(payload.get("mnemos_fusion_policy") or "balanced"),
            max_chars_per_hit=int(payload.get("mnemos_max_chars_per_hit") or 1200),
            temperature=temperature,
            num_predict=num_predict,
            history=history,
            condense_queries=_env_flag("MNEMOS_PROXY_QUERY_CONDENSE"),
        )
        return query, requested_model, model, run_kwargs

    @app.post("/v1/chat/completions")
    def v1_chat_completions():
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return _openai_error("messages must be a list")

        temperature = float(payload.get("temperature", 0.0) or 0.0)
        num_predict = int(payload.get("max_tokens") or payload.get("num_predict") or 700)
        query, requested_model, model, run_kwargs = _chat_run_kwargs(
            payload, temperature=temperature, num_predict=num_predict
        )
        if not query:
            return _openai_error("at least one user message with text content is required")

        created = int(time.time())
        completion_id = f"chatcmpl-mnemos-{uuid.uuid4().hex}"
        receipt_base_url = _public_receipt_base_url()

        if payload.get("stream") is True:
            events = stream_query_runner(**run_kwargs)
            first = next(events, None)
            if first is not None and first.get("event") == "retrieval":
                retrieval_result = first.get("result") or {}
                footer_enabled = should_append_footer(payload)

                def finalize(done_result: dict[str, Any]) -> str:
                    return _finalize_stream_suffix(
                        result=done_result,
                        receipt_dir=resolved_receipt_dir,
                        receipt_id=completion_id,
                        created=created,
                        query=query,
                        requested_model=requested_model,
                        model=model,
                        receipt_base_url=receipt_base_url,
                        footer_enabled=footer_enabled,
                    )

                return _openai_live_stream_response(
                    completion_id=completion_id,
                    created=created,
                    model=requested_model,
                    retrieval_result=retrieval_result,
                    events=events,
                    finalize=finalize,
                )
            # Terminal before generation (no evidence / MNEMOS error): reply
            # with a single-chunk stream carrying the withheld-answer footer.
            result = (first or {}).get("result") or {}
            answer = _finalize_result(
                result=result,
                payload=payload,
                receipt_dir=resolved_receipt_dir,
                receipt_id=completion_id,
                created=created,
                query=query,
                requested_model=requested_model,
                model=model,
                receipt_base_url=receipt_base_url,
            )
            return _openai_stream_response(
                completion_id=completion_id,
                created=created,
                model=requested_model,
                answer=answer,
                result=result,
            )

        result = query_runner(**run_kwargs)
        answer = _finalize_result(
            result=result,
            payload=payload,
            receipt_dir=resolved_receipt_dir,
            receipt_id=completion_id,
            created=created,
            query=query,
            requested_model=requested_model,
            model=model,
            receipt_base_url=receipt_base_url,
        )
        return jsonify(
            _openai_chat_completion_response(
                completion_id=completion_id,
                created=created,
                model=requested_model,
                answer=answer,
                result=result,
            )
        )

    def _handle_ollama_chat():
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return jsonify({"error": "messages must be a list"}), 400

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            options = {}

        query, requested_model, model, run_kwargs = _chat_run_kwargs(
            payload,
            temperature=float(options.get("temperature", 0.0) or 0.0),
            num_predict=int(options.get("num_predict") or 700),
        )
        if not query:
            return jsonify({"error": "at least one user message with text content is required"}), 400

        created = int(time.time())
        receipt_id = f"chatcmpl-mnemos-{uuid.uuid4().hex}"
        receipt_base_url = _public_receipt_base_url()

        if payload.get("stream") is True:
            events = stream_query_runner(**run_kwargs)
            first = next(events, None)
            if first is not None and first.get("event") == "retrieval":
                retrieval_result = first.get("result") or {}
                footer_enabled = should_append_footer(payload)

                def finalize(done_result: dict[str, Any]) -> str:
                    return _finalize_stream_suffix(
                        result=done_result,
                        receipt_dir=resolved_receipt_dir,
                        receipt_id=receipt_id,
                        created=created,
                        query=query,
                        requested_model=requested_model,
                        model=model,
                        receipt_base_url=receipt_base_url,
                        footer_enabled=footer_enabled,
                    )

                return _ollama_live_stream_response(
                    model=requested_model,
                    retrieval_result=retrieval_result,
                    events=events,
                    finalize=finalize,
                )
            result = (first or {}).get("result") or {}
            answer = _finalize_result(
                result=result,
                payload=payload,
                receipt_dir=resolved_receipt_dir,
                receipt_id=receipt_id,
                created=created,
                query=query,
                requested_model=requested_model,
                model=model,
                receipt_base_url=receipt_base_url,
            )
            return _ollama_stream_response(model=requested_model, answer=answer, result=result)

        result = query_runner(**run_kwargs)
        answer = _finalize_result(
            result=result,
            payload=payload,
            receipt_dir=resolved_receipt_dir,
            receipt_id=receipt_id,
            created=created,
            query=query,
            requested_model=requested_model,
            model=model,
            receipt_base_url=receipt_base_url,
        )
        return jsonify(_ollama_chat_response(model=requested_model, answer=answer, result=result))

    @app.post("/api/chat")
    def api_chat():
        return _handle_ollama_chat()

    @app.post("/v1/api/chat")
    def v1_api_chat():
        return _handle_ollama_chat()

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ollama_client = OllamaChatClient(args.ollama_base_url)
    app = create_app(
        ollama_base_url=args.ollama_base_url,
        query_runner=lambda **kwargs: run_query(
            **kwargs,
            ollama_client=ollama_client,
        ),
        stream_query_runner=lambda **kwargs: run_query_stream(
            **kwargs,
            ollama_client=ollama_client,
        ),
    )
    try:
        from waitress import serve
    except ImportError:
        app.run(host=args.host, port=args.port, threaded=True)
    else:
        # send_bytes=1 so SSE/NDJSON chunks flush immediately instead of being
        # batched, which would defeat token-level streaming.
        serve(app, host=args.host, port=args.port, threads=8, send_bytes=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
