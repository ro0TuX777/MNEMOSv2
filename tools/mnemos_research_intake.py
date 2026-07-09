"""Research artifact intake for MNEMOS.

Indexes local research artifacts into MNEMOS and can ask Ollama to produce a
source-linked research packet. This is an MFS boundary tool: it uses
``mnemos_sdk`` for indexing and does not alter MNEMOS retrieval or enforcement
policy.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemos_sdk import MnemosClient, MnemosConfig  # noqa: E402
from tools.mnemos_ollama_chat import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaChatClient,
    normalize_base_url,
)

CLAIM_BOUNDARY = (
    "MFS_RESEARCH_INTAKE_R0_ARTIFACT_MEMORY: indexes supplied artifacts and "
    "optional Ollama summaries into MNEMOS context; it does not make integration "
    "decisions, alter retrieval policy, or enable R1/R2 enforcement."
)

TEXT_SUFFIXES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".json": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".csv": "data",
}


class MnemosIndexClient(Protocol):
    def index(self, documents: list[dict[str, Any]], *, tiers: list[str] | None = None) -> Any:
        ...


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "word"
    return TEXT_SUFFIXES.get(suffix, "text")


def extract_text(path: Path | str) -> str:
    """Extract readable text from supported research artifact types."""
    return extract_text_with_details(path)[0]


def extract_text_with_details(path: Path | str) -> tuple[str, dict[str, Any]]:
    """Extract text and return (text, extraction details) for receipt metadata."""
    pages, details = extract_pages_with_details(path)
    return "\n\n".join(pages).strip(), details


def extract_pages_with_details(path: Path | str) -> tuple[list[str], dict[str, Any]]:
    """Extract per-page text (single-element list for unpaged formats).

    Page boundaries feed ``page_start``/``page_end`` chunk lineage so evidence
    receipts can point at pages, not just files.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages, method, chars_per_page = _extract_pdf_pages(p)
        return pages, {
            "extraction_method": method,
            "extraction_chars_per_page": round(chars_per_page, 1),
            "page_count": len(pages),
        }
    if suffix == ".docx":
        return [_extract_docx_text(p).strip()], {"extraction_method": "docx_xml"}
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    return [text], {"extraction_method": "plain_text"}


# Below this pypdf text density (chars per page), a PDF is treated as scanned
# (image-only) and routed to the Docling OCR fallback.
PDF_OCR_MIN_CHARS_PER_PAGE = 100.0


def _extract_pdf_pages(path: Path) -> tuple[list[str], str, float]:
    """Extract PDF text; returns (page_texts, extraction_method, chars_per_page).

    Tries pypdf first (fast, no OCR). If the extracted text density is below
    ``PDF_OCR_MIN_CHARS_PER_PAGE`` the PDF is treated as scanned and re-parsed
    with Docling OCR (optional dependency).
    """
    pages = _extract_pdf_pages_pypdf(path)
    page_count = max(len(pages), 1)
    chars_per_page = sum(len(page.strip()) for page in pages) / page_count
    if chars_per_page >= PDF_OCR_MIN_CHARS_PER_PAGE:
        return pages, "pypdf", chars_per_page
    ocr_pages = _extract_pdf_pages_docling(path)
    ocr_chars = sum(len(page.strip()) for page in ocr_pages)
    return ocr_pages, "docling_ocr", ocr_chars / max(len(ocr_pages), page_count, 1)


def _extract_pdf_pages_pypdf(path: Path) -> list[str]:
    try:
        import pypdf
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PDF extraction requires pypdf to be installed") from exc
    reader = pypdf.PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_pdf_pages_docling(path: Path) -> list[str]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            f"{path.name} appears to be a scanned PDF (no extractable text layer). "
            "OCR fallback requires the optional 'docling' package: pip install docling"
        ) from exc
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    document = converter.convert(str(path)).document
    page_texts: dict[int, list[str]] = {}
    for item, _level in document.iterate_items():
        prov = getattr(item, "prov", None)
        page_no = prov[0].page_no if prov else 1
        text = getattr(item, "text", None)
        if text is None and type(item).__name__ == "TableItem":
            try:
                text = item.export_to_markdown(document)
            except TypeError:  # older docling-core takes no doc argument
                text = item.export_to_markdown()
        if text:
            page_texts.setdefault(page_no, []).append(text)
    if not page_texts:
        return []
    return ["\n".join(page_texts.get(n, [])) for n in range(1, max(page_texts) + 1)]


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        data = archive.read("word/document.xml")
    root = ElementTree.fromstring(data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = [node.text or "" for node in root.findall(".//w:t", ns)]
    return html.unescape(" ".join(part for part in parts if part).strip())


def chunk_text(text: str, *, max_words: int = 350, overlap_words: int = 50) -> list[str]:
    return [
        chunk
        for chunk, _start, _end in chunk_pages(
            [text], max_words=max_words, overlap_words=overlap_words
        )
    ]


def chunk_pages(
    pages: list[str], *, max_words: int = 350, overlap_words: int = 50
) -> list[tuple[str, int, int]]:
    """Chunk page texts into (chunk, page_start, page_end) with 1-based pages.

    Chunk content is identical to ``chunk_text`` over the joined pages, so
    deterministic chunk IDs are unaffected by page tracking.
    """
    words: list[str] = []
    word_pages: list[int] = []
    for page_no, page_text in enumerate(pages, start=1):
        for word in re.sub(r"\s+", " ", page_text).strip().split(" "):
            if word:
                words.append(word)
                word_pages.append(page_no)
    if not words:
        return []
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")
    chunks: list[tuple[str, int, int]] = []
    step = max_words - overlap_words
    index = 0
    while index < len(words):
        end = min(index + max_words, len(words))
        chunks.append((" ".join(words[index:end]), word_pages[index], word_pages[end - 1]))
        if index + max_words >= len(words):
            break
        index += step
    return chunks


def build_documents(
    files: list[Path | str],
    *,
    project: str,
    capability: str,
    status: str = "new",
    tags: list[str] | None = None,
    max_words: int = 350,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    clean_tags = [tag for tag in (tags or []) if tag]
    for raw_path in files:
        path = Path(raw_path)
        pages, extraction_details = extract_pages_with_details(path)
        chunks = chunk_pages(pages, max_words=max_words, overlap_words=overlap_words)
        source_uri = path.resolve().as_uri()
        artifact_type = _artifact_type(path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for chunk_index, (chunk, page_start, page_end) in enumerate(chunks):
            chunk_hash = _sha256_text(f"{source_uri}\n{chunk_index}\n{chunk}")[:20]
            metadata = {
                "source_path": str(path),
                "source_uri": source_uri,
                "filename": path.name,
                "artifact_type": artifact_type,
                "project": project,
                "capability": capability,
                "status": status,
                "tags": clean_tags,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "file_sha256": file_hash,
                "claim_boundary": CLAIM_BOUNDARY,
                **extraction_details,
            }
            if artifact_type == "pdf":
                metadata["page_start"] = page_start
                metadata["page_end"] = page_end
            documents.append(
                {
                    "id": f"research::{chunk_hash}",
                    "content": chunk,
                    "source": source_uri,
                    "metadata": metadata,
                }
            )
    return documents


def build_summary_prompt(
    *,
    project: str,
    capability: str,
    status: str,
    documents: list[dict[str, Any]],
) -> str:
    excerpts = []
    for doc in documents[:8]:
        meta = doc["metadata"]
        excerpts.append(
            {
                "source": meta["source_path"],
                "chunk_index": meta["chunk_index"],
                "text": doc["content"][:1600],
            }
        )
    payload = {
        "project": project,
        "capability": capability,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "excerpts": excerpts,
    }
    return (
        "Create a concise research intake packet in Markdown. Use only the "
        "provided excerpts. Include: Summary, Potential relevance, Risks, "
        "Suggested local test, Open questions, and Source notes. Do not claim "
        "that integration is approved.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _ollama_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create research intake packets from supplied MNEMOS "
                    "artifact excerpts. MNEMOS is the evidence source; Ollama "
                    "is only the local model runtime."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 900},
    }


def _answer_from_response(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(response.get("response"), str):
        return response["response"].strip()
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


def _write_packet(
    output_path: Path,
    *,
    project: str,
    capability: str,
    status: str,
    indexed: int,
    summary: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# MNEMOS Research Intake Packet",
                "",
                f"- Project: `{project}`",
                f"- Capability: `{capability}`",
                f"- Status: `{status}`",
                f"- Indexed chunks: `{indexed}`",
                f"- Claim boundary: `{CLAIM_BOUNDARY}`",
                "",
                summary.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _index_documents_in_batches(
    mnemos: MnemosIndexClient,
    documents: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[bool, int, str | None]:
    indexed = 0
    size = max(1, int(batch_size))
    for start in range(0, len(documents), size):
        batch = documents[start:start + size]
        response = mnemos.index(batch)
        if not bool(getattr(response, "ok", False)):
            return False, indexed, getattr(response, "error", "unknown_error")
        result = getattr(response, "data", {}).get("result", {})
        indexed += int(result.get("indexed", len(batch)))
    return True, indexed, None


def run_intake(
    *,
    files: list[Path | str],
    project: str,
    capability: str,
    status: str = "new",
    tags: list[str] | None = None,
    max_words: int = 350,
    overlap_words: int = 50,
    summarize: bool = False,
    output_path: Path | None = None,
    mnemos_client: MnemosIndexClient | None = None,
    ollama_client: OllamaChatClient | None = None,
    ollama_model: str = DEFAULT_MODEL,
    batch_size: int = 25,
) -> dict[str, Any]:
    documents = build_documents(
        files,
        project=project,
        capability=capability,
        status=status,
        tags=tags,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    indexed_paths = {doc["metadata"]["source_path"] for doc in documents}
    files_without_content = [
        str(Path(path)) for path in files if str(Path(path)) not in indexed_paths
    ]
    if not documents:
        return {
            "status": "no_documents",
            "indexed": 0,
            "claim_boundary": CLAIM_BOUNDARY,
            "files": [str(Path(path)) for path in files],
            "files_without_content": files_without_content,
        }

    mnemos = mnemos_client or MnemosClient(MnemosConfig.from_env())
    ok, indexed, error = _index_documents_in_batches(
        mnemos,
        documents,
        batch_size=batch_size,
    )
    if not ok:
        return {
            "status": "index_failed",
            "indexed": indexed,
            "error": error,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    summary = None
    if summarize:
        ollama = ollama_client or OllamaChatClient(
            normalize_base_url(os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        )
        prompt = build_summary_prompt(
            project=project,
            capability=capability,
            status=status,
            documents=documents,
        )
        summary = _answer_from_response(ollama.chat(_ollama_payload(ollama_model, prompt)))
        if output_path is not None:
            _write_packet(
                Path(output_path),
                project=project,
                capability=capability,
                status=status,
                indexed=indexed,
                summary=summary,
            )

    return {
        "status": "ok",
        "indexed": indexed,
        "document_count": len(documents),
        "summary": summary,
        "claim_boundary": CLAIM_BOUNDARY,
        "files": [str(Path(path)) for path in files],
        "files_without_content": files_without_content,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="PDF, DOCX, Markdown, text, code, or data files.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--status", default="new")
    parser.add_argument("--tag", action="append", default=[], help="Repeatable tag metadata.")
    parser.add_argument("--max-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--summarize-with-ollama", action="store_true")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_intake(
        files=[Path(item) for item in args.files],
        project=args.project,
        capability=args.capability,
        status=args.status,
        tags=args.tag,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        batch_size=args.batch_size,
        summarize=args.summarize_with_ollama,
        output_path=args.output,
        ollama_model=args.ollama_model,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Status: {result['status']}")
        print(f"Indexed chunks: {result['indexed']}")
        for skipped in result.get("files_without_content", []):
            print(f"WARNING: no text content extracted, not indexed: {skipped}")
        print(result["claim_boundary"])
        if result.get("summary"):
            print("\nSummary:\n")
            print(result["summary"])
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
