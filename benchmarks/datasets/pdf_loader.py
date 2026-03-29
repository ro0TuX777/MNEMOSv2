"""
MNEMOS Benchmark -- Real-World PDF Corpus Loader
===================================================

Extracts text from PDF files, chunks into passage-level Engram objects
with metadata derived from filenames and content analysis.

Usage:
    from benchmarks.datasets.pdf_loader import load_pdf_corpus
    engrams = load_pdf_corpus("/path/to/pdfs")
"""

import hashlib
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnemos.engram.model import Engram


# ---- Domain classification heuristics ----

DOMAIN_KEYWORDS = {
    "machine_learning": [
        "neural network", "transformer", "attention", "gradient", "backpropagation",
        "loss function", "embedding", "fine-tuning", "pretraining", "encoder",
        "decoder", "bert", "gpt", "llm", "diffusion", "generative",
        "classification", "regression", "supervised", "unsupervised",
    ],
    "nlp": [
        "language model", "tokenization", "semantic", "syntactic", "parsing",
        "named entity", "sentiment", "translation", "summarization", "question answering",
        "text generation", "corpus", "vocabulary", "word embedding",
    ],
    "computer_vision": [
        "image", "convolution", "cnn", "object detection", "segmentation",
        "pixel", "resolution", "visual", "camera", "lidar", "depth",
    ],
    "reinforcement_learning": [
        "reward", "policy", "agent", "environment", "exploration", "exploitation",
        "q-learning", "markov", "trajectory", "episode",
    ],
    "systems": [
        "distributed", "parallel", "latency", "throughput", "scalability",
        "memory", "cache", "gpu", "cuda", "inference", "serving", "deployment",
    ],
}


def _classify_domain(text: str) -> Tuple[str, List[str]]:
    """Classify text into a domain and extract relevant tags."""
    text_lower = text.lower()
    scores = {}
    matched_tags = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in text_lower]
        scores[domain] = len(hits)
        matched_tags[domain] = hits

    if not scores or max(scores.values()) == 0:
        return "general", ["research"]

    best_domain = max(scores, key=scores.get)
    tags = matched_tags[best_domain][:5]  # top 5 matched keywords as tags
    return best_domain, tags


def _extract_title(text: str) -> str:
    """Extract likely paper title from first page text."""
    lines = text.strip().split("\n")
    # Title is usually the first non-empty, non-short line
    for line in lines[:10]:
        clean = line.strip()
        if len(clean) > 20 and not clean.startswith("arXiv"):
            return clean[:200]
    return "Untitled"


def _chunk_text(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> List[str]:
    """
    Chunk text into passage-level segments.

    Strategy:
    - Split on paragraph boundaries (double newline)
    - Merge short paragraphs into chunks up to max_chars
    - Apply overlap for retrieval continuity
    - Skip chunks below min_chars
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars and current:
            if len(current) >= min_chars:
                chunks.append(current.strip())
            # Overlap: keep tail of current chunk
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    # Last chunk
    if current.strip() and len(current.strip()) >= min_chars:
        chunks.append(current.strip())

    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF loading. "
            "Install with: pip install PyMuPDF"
        )

    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def load_pdf_corpus(
    pdf_dir: str,
    max_chunk_chars: int = 1500,
    overlap_chars: int = 200,
    min_chunk_chars: int = 100,
    max_pdfs: Optional[int] = None,
) -> List[Engram]:
    """
    Load a directory of PDFs into Engram objects.

    Each PDF is:
    1. Extracted to text via PyMuPDF
    2. Chunked into ~1500-char passages with 200-char overlap
    3. Classified by domain (ML, NLP, CV, RL, Systems)
    4. Assigned metadata: source paper ID, chunk index, domain, title

    Args:
        pdf_dir: Path to directory containing PDF files
        max_chunk_chars: Maximum characters per chunk (default 1500)
        overlap_chars: Overlap between chunks (default 200)
        min_chunk_chars: Minimum chunk length to keep (default 100)
        max_pdfs: Optional limit on number of PDFs to process

    Returns:
        List of Engram objects ready for benchmarking
    """
    pdf_path = Path(pdf_dir)
    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]

    print(f"  Loading {len(pdf_files)} PDFs from {pdf_dir}...")

    all_engrams = []
    skipped = 0

    for pdf_idx, pdf_file in enumerate(pdf_files):
        try:
            text = extract_pdf_text(pdf_file)
        except Exception as e:
            print(f"    [WARN] Failed to extract {pdf_file.name}: {e}")
            skipped += 1
            continue

        if len(text.strip()) < 200:
            print(f"    [WARN] {pdf_file.name}: too little text ({len(text)} chars)")
            skipped += 1
            continue

        # Extract paper metadata
        paper_id = pdf_file.stem  # e.g., "2603.10155v1"
        title = _extract_title(text)
        domain, tags = _classify_domain(text)

        # Chunk
        chunks = _chunk_text(
            text,
            max_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chunk_chars,
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            doc_id = hashlib.sha256(
                f"{paper_id}:{chunk_idx}:{chunk_text[:100]}".encode()
            ).hexdigest()[:16]
            # Deterministic synthetic policy metadata for governance stress tests.
            seed_int = int(hashlib.sha256(f"{paper_id}:{chunk_idx}".encode()).hexdigest()[:8], 16)
            ts = datetime(2023, 1, 1) + timedelta(days=(seed_int % 1095))
            ts_iso = ts.isoformat() + "Z"
            ts_epoch = int(ts.timestamp())
            tenant = ["tenant-A", "tenant-B", "tenant-C", "tenant-D"][seed_int % 4]
            clearance = ["public", "internal", "restricted", "classified"][seed_int % 4]

            engram = Engram(
                id=doc_id,
                content=chunk_text,
                source=f"arxiv:{paper_id}",
                neuro_tags=tags + [domain],
                confidence=0.85,
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "domain": domain,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "department": domain,       # for filter compatibility
                    "tenant": tenant,            # for governance filter tests
                    "clearance": clearance,      # for governance filter tests
                    "timestamp": ts_iso,
                    "timestamp_epoch": ts_epoch,
                },
            )
            all_engrams.append(engram)

        if (pdf_idx + 1) % 10 == 0:
            print(f"    Processed {pdf_idx + 1}/{len(pdf_files)} PDFs "
                  f"({len(all_engrams)} chunks so far)")

    print(f"  [OK] {len(all_engrams)} engrams from {len(pdf_files) - skipped} PDFs "
          f"(skipped {skipped})")

    return all_engrams
