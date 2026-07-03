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
