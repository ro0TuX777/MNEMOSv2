from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools import mnemos_research_intake as intake


def _write_docx(path: Path, text: str) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        + text
        + "</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


def test_extract_text_supports_markdown_and_docx(tmp_path):
    md = tmp_path / "paper_notes.md"
    md.write_text("# Paper Notes\n\nUseful retrieval idea.", encoding="utf-8")
    docx = tmp_path / "decision.docx"
    _write_docx(docx, "Decision memo text")

    assert "Useful retrieval idea" in intake.extract_text(md)
    assert intake.extract_text(docx) == "Decision memo text"


def test_build_documents_chunks_with_research_metadata(tmp_path):
    source = tmp_path / "repo.md"
    source.write_text(" ".join(f"word{i}" for i in range(45)), encoding="utf-8")

    docs = intake.build_documents(
        [source],
        project="MNEMOS",
        capability="agent memory",
        status="new",
        tags=["github", "retrieval"],
        max_words=20,
        overlap_words=5,
    )

    assert len(docs) == 3
    assert docs[0]["id"].startswith("research::")
    assert docs[0]["metadata"]["project"] == "MNEMOS"
    assert docs[0]["metadata"]["capability"] == "agent memory"
    assert docs[0]["metadata"]["artifact_type"] == "markdown"
    assert docs[0]["metadata"]["chunk_index"] == 0
    assert docs[0]["metadata"]["chunk_count"] == 3
    assert docs[0]["metadata"]["tags"] == ["github", "retrieval"]
    assert "word0" in docs[0]["content"]
    assert "word15" in docs[1]["content"]


def test_run_intake_indexes_documents_and_writes_summary_packet(tmp_path):
    source = tmp_path / "capability.md"
    source.write_text("This GitHub project proposes local agent memory retrieval.", encoding="utf-8")
    output = tmp_path / "packet.md"

    class RecordingMnemos:
        def __init__(self):
            self.documents = None

        def index(self, documents, *, tiers=None):
            self.documents = documents
            return type(
                "Resp",
                (),
                {
                    "ok": True,
                    "status": "healthy",
                    "error": None,
                    "data": {"result": {"indexed": len(documents), "tiers": {"qdrant": len(documents)}}},
                },
            )()

    class RecordingOllama:
        def __init__(self):
            self.payload = None

        def chat(self, payload):
            self.payload = payload
            return {"message": {"content": "## Summary\nUseful for MNEMOS.\n"}}

    mnemos = RecordingMnemos()
    ollama = RecordingOllama()
    result = intake.run_intake(
        files=[source],
        project="MNEMOS",
        capability="agent memory",
        status="reviewed",
        tags=["github"],
        summarize=True,
        output_path=output,
        mnemos_client=mnemos,
        ollama_client=ollama,
        ollama_model="llama3.1",
    )

    assert result["status"] == "ok"
    assert result["indexed"] == 1
    assert mnemos.documents[0]["metadata"]["status"] == "reviewed"
    assert "local agent memory retrieval" in json.dumps(ollama.payload)
    assert output.read_text(encoding="utf-8").startswith("# MNEMOS Research Intake Packet")
    assert "Useful for MNEMOS" in output.read_text(encoding="utf-8")


def test_run_intake_indexes_documents_in_batches(tmp_path):
    source = tmp_path / "many_chunks.md"
    source.write_text(" ".join(f"word{i}" for i in range(45)), encoding="utf-8")

    class RecordingMnemos:
        def __init__(self):
            self.batch_sizes = []

        def index(self, documents, *, tiers=None):
            self.batch_sizes.append(len(documents))
            return type(
                "Resp",
                (),
                {
                    "ok": True,
                    "status": "healthy",
                    "error": None,
                    "data": {"result": {"indexed": len(documents), "tiers": {"qdrant": len(documents)}}},
                },
            )()

    mnemos = RecordingMnemos()
    result = intake.run_intake(
        files=[source],
        project="MNEMOS",
        capability="batch indexing",
        mnemos_client=mnemos,
        max_words=10,
        overlap_words=0,
        batch_size=2,
    )

    assert result["status"] == "ok"
    assert result["indexed"] == 5
    assert mnemos.batch_sizes == [2, 2, 1]


def test_run_intake_reports_no_documents_for_empty_file(tmp_path):
    source = tmp_path / "empty.md"
    source.write_text("   ", encoding="utf-8")

    result = intake.run_intake(
        files=[source],
        project="SAM",
        capability="planning",
        mnemos_client=object(),
    )

    assert result["status"] == "no_documents"
    assert result["indexed"] == 0
    assert result["files_without_content"] == [str(source)]


def _write_blank_pdf(path: Path) -> None:
    """Write an image-free, text-free PDF that mimics a scanned document."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def test_scanned_pdf_routes_to_docling_ocr_fallback(tmp_path, monkeypatch):
    pdf = tmp_path / "scanned.pdf"
    _write_blank_pdf(pdf)

    monkeypatch.setattr(
        intake, "_extract_pdf_pages_docling", lambda path: ["OCR recovered text from scan."]
    )

    text, details = intake.extract_text_with_details(pdf)
    assert text == "OCR recovered text from scan."
    assert details["extraction_method"] == "docling_ocr"
    assert details["extraction_chars_per_page"] > 0
    assert details["page_count"] == 1


def test_text_pdf_uses_pypdf_without_ocr(tmp_path, monkeypatch):
    pdf = tmp_path / "digital.pdf"
    _write_blank_pdf(pdf)

    dense_text = "word " * 500
    monkeypatch.setattr(
        intake, "_extract_pdf_pages_pypdf", lambda path: [dense_text]
    )

    def _fail(path):
        raise AssertionError("docling fallback must not run for text PDFs")

    monkeypatch.setattr(intake, "_extract_pdf_pages_docling", _fail)

    text, details = intake.extract_text_with_details(pdf)
    assert "word" in text
    assert details["extraction_method"] == "pypdf"
    assert details["extraction_chars_per_page"] >= intake.PDF_OCR_MIN_CHARS_PER_PAGE


def test_build_documents_records_extraction_method(tmp_path, monkeypatch):
    pdf = tmp_path / "scanned.pdf"
    _write_blank_pdf(pdf)
    monkeypatch.setattr(
        intake,
        "_extract_pdf_pages_docling",
        lambda path: [" ".join(f"ocr{i}" for i in range(30))],
    )

    docs = intake.build_documents([pdf], project="MNEMOS", capability="ocr")

    assert docs
    assert docs[0]["metadata"]["extraction_method"] == "docling_ocr"
    assert docs[0]["metadata"]["extraction_chars_per_page"] > 0

    md = tmp_path / "notes.md"
    md.write_text("plain markdown notes", encoding="utf-8")
    docs = intake.build_documents([md], project="MNEMOS", capability="ocr")
    assert docs[0]["metadata"]["extraction_method"] == "plain_text"


def test_chunk_pages_tracks_page_lineage_and_matches_chunk_text():
    pages = [
        " ".join(f"p1w{i}" for i in range(30)),
        " ".join(f"p2w{i}" for i in range(30)),
        " ".join(f"p3w{i}" for i in range(30)),
    ]
    chunks = intake.chunk_pages(pages, max_words=40, overlap_words=10)

    assert chunks[0][1] == 1  # first chunk starts on page 1
    assert chunks[0][2] == 2  # 40-word window crosses into page 2
    assert chunks[-1][2] == 3  # last chunk ends on the final page

    # Chunk content must be identical to page-blind chunking so that
    # deterministic chunk IDs (and thus dedupe) are unchanged.
    joined = "\n\n".join(pages)
    assert [c for c, _, _ in chunks] == intake.chunk_text(joined, max_words=40, overlap_words=10)


def test_pdf_chunks_carry_page_start_and_page_end(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _write_blank_pdf(pdf)
    pages = [" ".join(f"p{n}w{i}" for i in range(200)) for n in (1, 2, 3)]
    monkeypatch.setattr(intake, "_extract_pdf_pages_pypdf", lambda path: pages)

    docs = intake.build_documents(
        [pdf], project="MNEMOS", capability="lineage", max_words=250, overlap_words=0
    )

    assert docs[0]["metadata"]["page_start"] == 1
    assert docs[0]["metadata"]["page_end"] == 2
    assert docs[-1]["metadata"]["page_end"] == 3
    assert docs[0]["metadata"]["page_count"] == 3

    md = tmp_path / "notes.md"
    md.write_text("no pages in markdown", encoding="utf-8")
    md_docs = intake.build_documents([md], project="MNEMOS", capability="lineage")
    assert "page_start" not in md_docs[0]["metadata"]
    assert "page_end" not in md_docs[0]["metadata"]


def test_run_intake_reports_files_without_content_alongside_indexed(tmp_path, monkeypatch):
    good = tmp_path / "good.md"
    good.write_text("Useful research content here.", encoding="utf-8")
    scanned = tmp_path / "scan_failed.pdf"
    _write_blank_pdf(scanned)
    monkeypatch.setattr(intake, "_extract_pdf_pages_docling", lambda path: [])

    class RecordingMnemos:
        def index(self, documents, *, tiers=None):
            return type(
                "Resp",
                (),
                {
                    "ok": True,
                    "status": "healthy",
                    "error": None,
                    "data": {"result": {"indexed": len(documents)}},
                },
            )()

    result = intake.run_intake(
        files=[good, scanned],
        project="MNEMOS",
        capability="ocr",
        mnemos_client=RecordingMnemos(),
    )

    assert result["status"] == "ok"
    assert result["indexed"] == 1
    assert result["files_without_content"] == [str(scanned)]
