# Chat Integration Evidence Contract

## Audience

This guide is for AI developers integrating a chat, assistant, RAG, or agentic UI with MNEMOS.

MNEMOS search results now include a normalized evidence package designed for citation-aware answer generation. Use this package whenever your chat system claims an answer came from retrieved memory.

## Why This Exists

MNEMOS is often used behind chat systems where users ask natural-language questions over uploaded documents, PDFs, local folders, or governed knowledge stores.

Without a formal evidence contract, chat systems may produce useful answers but weak provenance, for example:

- "based on MNEMOS" without naming the source document,
- citations invented by the model,
- no chunk/page/rank metadata,
- no clear way to distinguish retrieved evidence from model prior knowledge.

The evidence contract solves this by making source metadata available in a consistent shape for every search consumer.

## Search Response Fields

`POST /v1/mnemos/search` returns evidence in two places:

1. Per result: `results[].evidence`
2. Per search: `meta.evidence_summary`

Use `results[].evidence` when building prompt context or inline citations.

Use `meta.evidence_summary` when rendering a concise `Sources:` footer.

## Per-Result Evidence

Each result includes:

```json
{
  "score": 0.8809,
  "rank": 1,
  "engram": {
    "id": "engram-1",
    "content": "retrieved text...",
    "source": "file:///example-corpus/example_research_note.pdf",
    "metadata": {
      "filename": "example_research_note.pdf",
      "title": "Example Research Note",
      "chunk_index": 1,
      "chunk_count": 17
    }
  },
  "evidence": {
    "result_id": "engram-1",
    "engram_id": "engram-1",
    "label": "example_research_note.pdf chunk 1/17",
    "document_title": "Example Research Note",
    "filename": "example_research_note.pdf",
    "source_uri": "file:///example-corpus/example_research_note.pdf",
    "file_id": "file-1",
    "file_hash": "hash-1",
    "chunk_id": null,
    "chunk_index": 1,
    "chunk_count": 17,
    "page_start": 2,
    "page_end": 3,
    "char_start": 100,
    "char_end": 250,
    "ingestion_source": "local-folder",
    "canonical_store": "mnemos",
    "score": 0.8809,
    "rank": 1
  }
}
```

Keys are intentionally null-friendly. Do not assume page or character spans are present for every ingestion source.

## Evidence Summary

The search envelope also includes grouped source evidence:

```json
{
  "meta": {
    "evidence_summary": {
      "source_count": 1,
      "sources": [
        {
          "filename": "example_research_note.pdf",
          "title": "Example Research Note",
          "source_uri": "file:///example-corpus/example_research_note.pdf",
          "top_score": 0.8809,
          "chunks_returned": 3,
          "chunk_indices": [1, 2, 3],
          "ranks": [1, 2, 3]
        }
      ]
    }
  }
}
```

This is the easiest data structure for a chat UI footer.

## SDK Usage

The typed SDK exposes evidence on each `SearchHit`:

```python
from mnemos_sdk import MnemosClient, MnemosConfig

client = MnemosClient(MnemosConfig.from_env())
client.wait_until_ready()

hits = client.search(
    "what does the example research note say about retention controls?",
    top_k=3,
)

for hit in hits:
    evidence = hit.evidence or {}
    print(
        hit.rank,
        evidence.get("filename"),
        evidence.get("document_title"),
        evidence.get("chunk_index"),
        evidence.get("chunk_count"),
        evidence.get("score"),
    )
```

## Raw HTTP Usage

```python
import requests

response = requests.post(
    "http://localhost:8700/v1/mnemos/search",
    json={
        "query": "your question",
        "top_k": 5
    },
    timeout=30,
)
payload = response.json()

results = payload.get("results", [])
summary = (payload.get("meta") or {}).get("evidence_summary") or {}
```

Prefer the SDK for application code when available. Raw HTTP is useful for smoke tests and diagnostics.

## Chat Prompt Construction

When passing retrieved context to a model, include compact evidence markers outside the prose body. For example:

```text
You are answering using retrieved MNEMOS evidence.
Use only the evidence IDs below for citations. Do not invent sources.

[E1] example_research_note.pdf, chunk 1/17, score 0.8809
Title: Example Research Note
Text:
...

[E2] example_research_note.pdf, chunk 2/17, score 0.8524
Title: Example Research Note
Text:
...
```

Then instruct the model:

```text
If you use retrieved facts, cite the evidence IDs in the answer.
If the evidence is insufficient, say so instead of filling gaps from general knowledge.
```

## Recommended Chat Footer

For normal user-facing chat, render a concise footer from `meta.evidence_summary`:

```text
Sources:
- example_research_note.pdf, "Example Research Note", chunks 1, 2, 3 of 17; ranks 1, 2, 3; top score 0.8809.
```

If page spans are available:

```text
Sources:
- example_research_note.pdf, pages 2-3, chunk 1/17, score 0.8809.
```

## Rules For Chat Integrators

- Cite only `results[].evidence` or `meta.evidence_summary`.
- Do not let the model invent filenames, titles, chunk numbers, or scores.
- Do not claim "from MNEMOS" when the result set is empty or scores are below your app's confidence threshold.
- Preserve `rank` and `score` in hidden traces, logs, or visible footers.
- Prefer source grouping in user-facing footers; prefer per-result evidence IDs in model prompts.
- Treat null page/span fields as "unknown", not as page 0 or chunk 0.
- Keep evidence attached to the generated answer for debugging and audit.

## Adapter Integration Note

External adapters should preserve MNEMOS provenance without flattening away
rank, score, chunk, or source metadata. The minimum adapter contract is:

- preserve the complete `results[].evidence` object,
- preserve result `rank` and `score`,
- keep `meta.evidence_summary` available for `Sources:` footers,
- do not synthesize filenames, titles, chunk indices, or scores,
- avoid claiming MNEMOS grounding when evidence is absent.

## Minimal Citation Formatter

```python
def format_sources(payload: dict) -> str:
    summary = ((payload.get("meta") or {}).get("evidence_summary") or {})
    sources = summary.get("sources") or []
    if not sources:
        return ""

    lines = ["Sources:"]
    for source in sources:
        filename = source.get("filename") or source.get("source_uri") or "unknown source"
        title = source.get("title")
        chunks = source.get("chunk_indices") or []
        ranks = source.get("ranks") or []
        top_score = source.get("top_score")

        parts = [str(filename)]
        if title:
            parts.append(f'"{title}"')
        if chunks:
            parts.append("chunks " + ", ".join(str(chunk) for chunk in chunks))
        if ranks:
            parts.append("ranks " + ", ".join(str(rank) for rank in ranks))
        if top_score is not None:
            parts.append(f"top score {top_score}")
        lines.append("- " + "; ".join(parts) + ".")

    return "\n".join(lines)
```

## Validation Prompt

Use a query that should be answerable from one known ingested document:

```text
Using your existing MNEMOS knowledge, answer the question from retrieval and include sources.
```

Expected behavior in a citation-aware chat integration:

- The chat system searches MNEMOS.
- The answer is grounded in retrieved results.
- No unrelated workflow or ticket is created unless explicitly requested.
- The answer includes either inline evidence IDs or a `Sources:` footer.
- The footer names the source filename, document title, chunk indices/count if present, and rank or score.
