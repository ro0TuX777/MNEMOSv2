"""
Matryoshka migration tool - BGE single-vector -> Nomic MRL named-vector collection.

Implements docs/reports/matryoshka_migration_shadow_replay_spec.md:

  Phase 0  preflight  - connectivity, prefix sentinel check (ABORTS if the
                        embedding wrapper ignores Nomic task prefixes),
                        target collection creation (dense_64 + dense_768)
  Phase 1  reembed    - resumable batch re-embed; checkpoint after every
                        batch; per-batch throughput + ETA telemetry
  Phase 2  replay     - divergence-aware anchor-query replay (Jaccard@10,
                        per-query drift warnings, promotion verdict)
  Phase 3  finalize   - prints the exact .env.mnemos edits for cutover;
                        optional source-collection cleanup gated behind
                        --cleanup --confirm-delete <exact collection name>

Non-destructive: the source collection is never modified. Cleanup requires
an explicit, exact-name confirmation flag.

Usage:
  python tools/mnemos_matryoshka_migrate.py --phase all
  python tools/mnemos_matryoshka_migrate.py --dry-run --dry-run-limit 500
  python tools/mnemos_matryoshka_migrate.py --phase reembed --batch-size 500
  python tools/mnemos_matryoshka_migrate.py --phase replay --anchors benchmarks/truthsets/anchor_queries_v1.json
  python tools/mnemos_matryoshka_migrate.py --cleanup --confirm-delete mnemos_engrams
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_TARGET_MODEL = "nomic-ai/nomic-embed-text-v1.5"
PREFIX_SCHEME = "nomic_v1.5_task_prefixes"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MRL_DIM = 64
FULL_DIM = 768
DEFAULT_CHECKPOINT = PROJECT_ROOT / "migration_checkpoint.json"
DEFAULT_ANCHORS = PROJECT_ROOT / "benchmarks" / "truthsets" / "anchor_queries_v1.json"
RAW_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "outputs" / "summaries"

JACCARD_WARN_THRESHOLD = 0.3   # per-query high-drift operator warning
JACCARD_PROCEED_MEDIAN = 0.6   # spec promotion matrix (per class median)
SCORE_DROP_WARN_THRESHOLD = 0.10  # absolute mean top-k score delta considered material
SCORE_DROP_TOP_K = 3


# ---- pure logic ---------------------------------------------------------


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard similarity of two id sets (1.0 when both empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def rank_displacement(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Mean absolute rank movement across overlapping IDs."""
    rank_a = {engram_id: idx for idx, engram_id in enumerate(a, start=1)}
    rank_b = {engram_id: idx for idx, engram_id in enumerate(b, start=1)}
    overlap = set(rank_a) & set(rank_b)
    if not overlap:
        return None
    return sum(abs(rank_a[i] - rank_b[i]) for i in overlap) / len(overlap)


def mean_top_score(rows: Sequence[Tuple[str, Optional[float]]], top_k: int = SCORE_DROP_TOP_K) -> Optional[float]:
    scores = [score for _, score in rows[:top_k] if score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def nomic_score_significantly_lower(
    bge_score: Optional[float],
    nomic_score: Optional[float],
    threshold: float = SCORE_DROP_WARN_THRESHOLD,
) -> bool:
    """True when both engines reported scores and Nomic trails materially."""
    if bge_score is None or nomic_score is None:
        return False
    return (bge_score - nomic_score) >= threshold


def mrl_slice(matrix: Any, dim: int = MRL_DIM) -> np.ndarray:
    """
    Nomic v1.5 MRL resizing recipe: layer-norm, truncate, L2-normalize.
    Plain truncation without the layer-norm degrades the nested space.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=1, keepdims=True)
    var = matrix.var(axis=1, keepdims=True)
    normed = (matrix - mean) / np.sqrt(var + 1e-5)
    sliced = normed[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return sliced / np.clip(norms, 1e-12, None)


def l2_normalize(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def replay_verdict(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Spec promotion matrix, per query class:
      median jaccard >= 0.6 and no labeled regression  -> PROCEED
      median jaccard <  0.6 and no labeled regression  -> REVIEW
      any labeled regression                            -> BLOCK (absolute)
    One regressing class blocks promotion for all.
    """
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_query:
        by_class.setdefault(row.get("query_class", "semantic"), []).append(row)

    class_verdicts: Dict[str, Dict[str, Any]] = {}
    overall = "PROCEED"
    for cls, rows in sorted(by_class.items()):
        sims = sorted(r["jaccard_at_10"] for r in rows)
        median = sims[len(sims) // 2] if len(sims) % 2 else (sims[len(sims) // 2 - 1] + sims[len(sims) // 2]) / 2
        regressed = any(r.get("labeled_regression") for r in rows)
        if regressed:
            verdict = "BLOCK"
        elif median >= JACCARD_PROCEED_MEDIAN:
            verdict = "PROCEED"
        else:
            verdict = "REVIEW"
        class_verdicts[cls] = {"median_jaccard_at_10": round(median, 4), "labeled_regression": regressed, "verdict": verdict}
        if verdict == "BLOCK":
            overall = "BLOCK"
        elif verdict == "REVIEW" and overall != "BLOCK":
            overall = "REVIEW"
    return {"overall": overall, "per_class": class_verdicts}


@dataclass
class MigrationState:
    """Resumable migration checkpoint, persisted after every committed batch."""

    migration_id: str = ""
    source_collection: str = ""
    target_collection: str = ""
    source_model: str = DEFAULT_SOURCE_MODEL
    target_model: str = DEFAULT_TARGET_MODEL
    current_model_version: str = DEFAULT_TARGET_MODEL
    prefix_scheme: str = PREFIX_SCHEME
    batch_size: int = 500
    total_engrams: int = 0
    processed_count: int = 0
    processed_ids: List[str] = field(default_factory=list)
    next_offset: Optional[Any] = None     # qdrant scroll next_page_offset
    failed_ids: List[str] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)  # engrams/sec, last 20
    start_time: str = ""
    updated_at: str = ""
    phase0_prefix_check: Optional[bool] = None
    phase1_complete: bool = False
    phase2_verdict: Optional[str] = None

    def save(self, path: Path) -> None:
        """Atomic write (tmp + replace) so a crash never corrupts the checkpoint."""
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "MigrationState":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("current_model_version"):
            data["current_model_version"] = data.get("target_model", DEFAULT_TARGET_MODEL)
        data.setdefault("processed_ids", [])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def record_batch(
        self,
        batch_count: int,
        elapsed_s: float,
        next_offset: Optional[Any],
        processed_ids: Optional[Sequence[str]] = None,
    ) -> float:
        self.processed_count += batch_count
        if processed_ids:
            self.processed_ids.extend(str(i) for i in processed_ids)
        self.next_offset = next_offset
        throughput = batch_count / elapsed_s if elapsed_s > 0 else 0.0
        self.throughput_samples = (self.throughput_samples + [round(throughput, 1)])[-20:]
        return throughput

    def eta_seconds(self) -> Optional[float]:
        if not self.throughput_samples or not self.total_engrams:
            return None
        rate = sum(self.throughput_samples) / len(self.throughput_samples)
        remaining = max(self.total_engrams - self.processed_count, 0)
        return remaining / rate if rate > 0 else None


# ---- embedders ----------------------------------------------------------


class NomicEmbedder:
    """Prefix-enforcing wrapper. Callers pass raw text; prefixes are internal."""

    def __init__(self, model_name: str = DEFAULT_TARGET_MODEL, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return l2_normalize(self._model.encode([DOC_PREFIX + t for t in texts], show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._model.encode([QUERY_PREFIX + text], show_progress_bar=False))[0]

    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """Unprefixed embedding - used ONLY by the prefix sentinel check."""
        return l2_normalize(self._model.encode(texts, show_progress_bar=False))


def prefix_sentinel_check(embedder) -> bool:
    """
    Phase 0 gate: prefixed and unprefixed embeddings of the same sentinel
    must differ. Identical vectors mean the wrapper (or a cached layer) is
    ignoring prefixes - silent quality degradation for Nomic v1.5.
    """
    sentinel = "mnemos prefix sentinel: gravitational waves were detected in 2015"
    with_prefix = embedder.embed_documents([sentinel])[0]
    without_prefix = embedder.embed_raw([sentinel])[0]
    return not np.allclose(with_prefix, without_prefix, atol=1e-6)


# ---- phases -------------------------------------------------------------


def _client(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, timeout=60)


def phase0_preflight(client, state: MigrationState, embedder: NomicEmbedder) -> None:
    from qdrant_client.models import Distance, VectorParams

    print("== Phase 0: preflight ==")
    if not prefix_sentinel_check(embedder):
        print("Aborting: Embedding engine is ignoring prefixes. Prefix injection is mandatory for Nomic v1.5.")
        sys.exit(2)
    state.phase0_prefix_check = True
    print("  [ok] prefix sentinel check passed (prefixed != unprefixed)")

    count = client.count(state.source_collection, exact=True).count
    state.total_engrams = count
    print(f"  [ok] source collection '{state.source_collection}': {count} points")

    existing = {c.name for c in client.get_collections().collections}
    if state.target_collection not in existing:
        client.create_collection(
            collection_name=state.target_collection,
            vectors_config={
                "dense_64": VectorParams(size=MRL_DIM, distance=Distance.COSINE),
                "dense_768": VectorParams(size=FULL_DIM, distance=Distance.COSINE),
            },
        )
        print(f"  [ok] created target collection '{state.target_collection}' (dense_64 + dense_768)")
    else:
        print(f"  [ok] target collection '{state.target_collection}' exists (resuming)")


def phase1_reembed(client, state: MigrationState, embedder: NomicEmbedder, checkpoint: Path) -> None:
    from qdrant_client.models import PointStruct

    print("== Phase 1: re-embed (resumable) ==")
    if state.phase1_complete:
        print("  [ok] already complete per checkpoint; skipping")
        return

    while True:
        points, next_offset = client.scroll(
            collection_name=state.source_collection,
            limit=state.batch_size,
            offset=state.next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        batch_started = time.perf_counter()
        texts, metas = [], []
        for p in points:
            payload = p.payload or {}
            texts.append(payload.get("content", ""))
            metas.append((p.id, payload))
        engram_ids = [str(payload.get("_mnemos_id", pid)) for pid, payload in metas]

        try:
            dense_768 = embedder.embed_documents(texts)
            dense_64 = mrl_slice(dense_768, MRL_DIM)
            client.upsert(
                collection_name=state.target_collection,
                points=[
                    PointStruct(
                        id=pid,
                        vector={"dense_64": dense_64[i].tolist(), "dense_768": dense_768[i].tolist()},
                        payload=payload,  # copied verbatim - edges/governance preserved
                    )
                    for i, (pid, payload) in enumerate(metas)
                ],
            )
        except Exception as exc:  # collect, don't abort a long run
            state.failed_ids.extend(engram_ids)
            print(f"  [fail] batch failed ({len(metas)} ids recorded for retry): {exc}")
            state.next_offset = next_offset
            state.save(checkpoint)
            if next_offset is None:
                break
            continue

        throughput = state.record_batch(
            len(points),
            time.perf_counter() - batch_started,
            next_offset,
            processed_ids=engram_ids,
        )
        state.save(checkpoint)
        eta = state.eta_seconds()
        eta_str = f", ETA {eta/60:.1f} min" if eta else ""
        print(f"  batch ok: {state.processed_count}/{state.total_engrams} "
              f"({throughput:.0f} engrams/sec{eta_str})")

        if next_offset is None:
            break

    state.phase1_complete = not state.failed_ids
    state.save(checkpoint)
    if state.failed_ids:
        print(f"  ! {len(state.failed_ids)} failed ids in checkpoint - rerun phase 1 or retry manually")
    else:
        print(f"  [ok] re-embed complete: {state.processed_count} engrams")


def phase1_dry_run(client, state: MigrationState, embedder: NomicEmbedder, dry_run_limit: int) -> Dict[str, Any]:
    """
    Non-destructive throughput probe: scroll source payloads, run the exact
    Nomic document embedding + MRL slicing path, and skip target writes.
    """
    print("== Dry run: re-embed throughput probe (no writes) ==")
    if dry_run_limit <= 0:
        raise ValueError("--dry-run-limit must be > 0")

    count = client.count(state.source_collection, exact=True).count
    state.total_engrams = count
    remaining = min(dry_run_limit, count)
    offset = None
    processed = 0
    samples: List[float] = []
    started = time.perf_counter()

    while remaining > 0:
        limit = min(state.batch_size, remaining)
        points, next_offset = client.scroll(
            collection_name=state.source_collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        batch_started = time.perf_counter()
        texts = [(p.payload or {}).get("content", "") for p in points]
        dense_768 = embedder.embed_documents(texts)
        _ = mrl_slice(dense_768, MRL_DIM)
        elapsed = time.perf_counter() - batch_started
        throughput = len(points) / elapsed if elapsed > 0 else 0.0
        samples.append(throughput)
        processed += len(points)
        remaining -= len(points)
        print(f"  dry batch ok: {processed}/{min(dry_run_limit, count)} ({throughput:.0f} engrams/sec)")

        if next_offset is None:
            break
        offset = next_offset

    total_elapsed = time.perf_counter() - started
    mean_throughput = processed / total_elapsed if total_elapsed > 0 and processed else 0.0
    median_batch = sorted(samples)[len(samples) // 2] if samples else 0.0
    result = {
        "source_collection": state.source_collection,
        "target_model": state.target_model,
        "sampled_engrams": processed,
        "total_engrams": count,
        "elapsed_s": round(total_elapsed, 4),
        "mean_engrams_per_sec": round(mean_throughput, 2),
        "median_batch_engrams_per_sec": round(median_batch, 2),
        "writes_performed": 0,
    }
    print("  dry-run summary:")
    print(f"    sampled: {processed}/{count} engrams")
    print(f"    mean throughput: {mean_throughput:.2f} engrams/sec")
    print(f"    median batch throughput: {median_batch:.2f} engrams/sec")
    print("    writes performed: 0")
    return result


def _result_rows(response: Any) -> List[Tuple[str, Optional[float]]]:
    rows: List[Tuple[str, Optional[float]]] = []
    for point in getattr(response, "points", response):
        payload = getattr(point, "payload", None) or {}
        engram_id = str(payload.get("_mnemos_id", getattr(point, "id", "")))
        score = getattr(point, "score", None)
        rows.append((engram_id, float(score) if score is not None else None))
    return rows


def _ids(rows: Sequence[Tuple[str, Optional[float]]]) -> List[str]:
    return [engram_id for engram_id, _ in rows]


def _search_source_top10(client, state: MigrationState, bge_model, query: str) -> List[Tuple[str, Optional[float]]]:
    vec = l2_normalize(bge_model.encode([query], show_progress_bar=False))[0]
    res = client.query_points(collection_name=state.source_collection, query=vec.tolist(),
                              limit=10, with_payload=True)
    return _result_rows(res)


def _search_target_top10(client, state: MigrationState, embedder: NomicEmbedder, query: str) -> List[Tuple[str, Optional[float]]]:
    from qdrant_client.models import Prefetch

    full = embedder.embed_query(query)
    coarse = mrl_slice(full.reshape(1, -1), MRL_DIM)[0]
    res = client.query_points(
        collection_name=state.target_collection,
        prefetch=Prefetch(query=coarse.tolist(), using="dense_64", limit=30),  # 3x oversample
        query=full.tolist(),
        using="dense_768",
        limit=10,
        with_payload=True,
    )
    return _result_rows(res)


def phase2_replay(client, state: MigrationState, embedder: NomicEmbedder,
                  anchors_path: Path, checkpoint: Path) -> Dict[str, Any]:
    print("== Phase 2: divergence-aware replay ==")
    from sentence_transformers import SentenceTransformer

    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"]
    bge_model = SentenceTransformer(state.source_model)

    per_query: List[Dict[str, Any]] = []
    for anchor in anchors:
        bge_rows = _search_source_top10(client, state, bge_model, anchor["query"])
        nomic_rows = _search_target_top10(client, state, embedder, anchor["query"])
        bge_ids = _ids(bge_rows)
        nomic_ids = _ids(nomic_rows)
        sim = jaccard(bge_ids, nomic_ids)
        displacement = rank_displacement(bge_ids, nomic_ids)
        bge_mean_score = mean_top_score(bge_rows)
        nomic_mean_score = mean_top_score(nomic_rows)
        significant_score_drop = nomic_score_significantly_lower(bge_mean_score, nomic_mean_score)

        expected = set(anchor.get("expected_ids") or [])
        regression = False
        recall_a = recall_b = None
        if expected:
            recall_a = len(expected & set(bge_ids)) / len(expected)
            recall_b = len(expected & set(nomic_ids)) / len(expected)
            regression = recall_b < recall_a

        row = {
            "query_id": anchor.get("query_id"),
            "query_class": anchor.get("class", "semantic"),
            "jaccard_at_10": round(sim, 4),
            "rank_displacement": round(displacement, 4) if displacement is not None else None,
            "bge_mean_top3_score": round(bge_mean_score, 4) if bge_mean_score is not None else None,
            "nomic_mean_top3_score": round(nomic_mean_score, 4) if nomic_mean_score is not None else None,
            "significant_score_drop": significant_score_drop,
            "recall_bge": recall_a,
            "recall_nomic": recall_b,
            "labeled_regression": regression,
            "bge_top10": bge_ids,
            "nomic_top10": nomic_ids,
        }
        per_query.append(row)

        if sim < JACCARD_WARN_THRESHOLD and significant_score_drop:
            print("  " + "!" * 64)
            print(f"  ! HIGH DRIFT + SCORE DROP: anchor '{anchor.get('query_id')}' jaccard@10={sim:.2f} (<{JACCARD_WARN_THRESHOLD})")
            print(f"  ! mean top-{SCORE_DROP_TOP_K} score: BGE={bge_mean_score:.4f}, Nomic={nomic_mean_score:.4f}")
            if regression:
                print(f"  ! AND labeled recall regression ({recall_a} -> {recall_b}) - promotion will be BLOCKED")
            print("  " + "!" * 64)
        elif sim < JACCARD_WARN_THRESHOLD:
            print(f"  drift notice: anchor '{anchor.get('query_id')}' jaccard@10={sim:.2f}; "
                  "score drop guard not triggered")
        else:
            print(f"  anchor '{anchor.get('query_id')}': jaccard@10={sim:.2f}"
                  + (f", recall {recall_a}->{recall_b}" if expected else ""))

    verdict = replay_verdict(per_query)
    state.phase2_verdict = verdict["overall"]
    state.save(checkpoint)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifact = {
        "migration_id": state.migration_id,
        "timestamp": timestamp,
        "source_model": state.source_model,
        "target_model": state.target_model,
        "per_query": per_query,
        "verdict": verdict,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"matryoshka_shadow_{timestamp}_raw.json"
    raw_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    decision_path = SUMMARY_DIR / f"matryoshka_shadow_{timestamp}_decision.md"
    decision_path.write_text(
        f"# Matryoshka Shadow Replay Decision\n\n- Verdict: **{verdict['overall']}**\n"
        f"- Per class: {json.dumps(verdict['per_class'], indent=2)}\n"
        f"- Raw: {raw_path.name}\n",
        encoding="utf-8",
    )
    print(f"  verdict: {verdict['overall']}  (artifacts: {raw_path.name}, {decision_path.name})")
    return verdict


def phase3_finalize(state: MigrationState) -> None:
    print("== Phase 3: cutover instructions ==")
    if state.phase2_verdict == "BLOCK":
        print("  PROMOTION BLOCKED by replay verdict - do not cut over. Investigate regressing anchors first.")
        return
    if state.phase2_verdict == "REVIEW":
        print("  REVIEW verdict - inspect diverged anchors before applying the edits below.")
    print(f"""
  To finalize the cutover, edit .env.mnemos (then restart the stack):

      MNEMOS_EMBEDDING_MODEL={state.target_model}
      MNEMOS_QDRANT_COLLECTION={state.target_collection}

  PowerShell session equivalent:

      $env:MNEMOS_EMBEDDING_MODEL = "{state.target_model}"
      $env:MNEMOS_QDRANT_COLLECTION = "{state.target_collection}"

  Bash/zsh session equivalent:

      export MNEMOS_EMBEDDING_MODEL="{state.target_model}"
      export MNEMOS_QDRANT_COLLECTION="{state.target_collection}"

  The derived-view cache starts a fresh model-isolated lane automatically
  (embedding_model_name is part of the cache key).

  Source collection '{state.source_collection}' was NOT modified. After
  validating the cutover, remove it with:

      python tools/mnemos_matryoshka_migrate.py --cleanup --confirm-delete {state.source_collection}
""")


def cleanup(client, state: MigrationState, confirm_delete: Optional[str]) -> None:
    if confirm_delete != state.source_collection:
        print(f"ABORT: --confirm-delete must exactly match the source collection "
              f"name ('{state.source_collection}'). Nothing was deleted.")
        sys.exit(2)
    client.delete_collection(state.source_collection)
    print(f"[ok] deleted source collection '{state.source_collection}'")


# ---- CLI ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="BGE -> Nomic MRL migration (Phase 7)")
    parser.add_argument("--phase", choices=["all", "preflight", "reembed", "replay", "finalize"], default="all")
    parser.add_argument("--app", default=None, help="Operator label for compatibility with app-scoped runbooks")
    parser.add_argument("--qdrant-url", default=os.getenv("MNEMOS_QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--source-collection", default=os.getenv("MNEMOS_QDRANT_COLLECTION", "mnemos_engrams"))
    parser.add_argument("--target-collection", default="mnemos_engrams_nomic_mrl")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Measure re-embedding throughput without target writes")
    parser.add_argument("--dry-run-limit", type=int, default=500, help="Maximum engrams to sample during --dry-run")
    parser.add_argument("--shadow-only", action="store_true", help="Compatibility alias for --phase replay")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--confirm-delete", default=None)
    args = parser.parse_args()
    if args.shadow_only:
        args.phase = "replay"

    if args.checkpoint.exists():
        state = MigrationState.load(args.checkpoint)
        print(f"Resuming migration {state.migration_id} "
              f"({state.processed_count}/{state.total_engrams} processed)")
    else:
        state = MigrationState(
            migration_id=f"matryoshka_{time.strftime('%Y%m%d_%H%M%S')}",
            source_collection=args.source_collection,
            target_collection=args.target_collection,
            target_model=args.target_model,
            current_model_version=args.target_model,
            batch_size=args.batch_size,
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    client = _client(args.qdrant_url)

    if args.cleanup:
        cleanup(client, state, args.confirm_delete)
        return

    embedder = NomicEmbedder(state.target_model, device=args.device)

    if args.dry_run:
        if not prefix_sentinel_check(embedder):
            print("Aborting: Embedding engine is ignoring prefixes. Prefix injection is mandatory for Nomic v1.5.")
            sys.exit(2)
        if args.app:
            print(f"App label: {args.app}")
        phase1_dry_run(client, state, embedder, args.dry_run_limit)
        return

    if args.phase in ("all", "preflight"):
        phase0_preflight(client, state, embedder)
        state.save(args.checkpoint)
    if args.phase in ("all", "reembed"):
        phase1_reembed(client, state, embedder, args.checkpoint)
    if args.phase in ("all", "replay"):
        phase2_replay(client, state, embedder, args.anchors, args.checkpoint)
    if args.phase in ("all", "finalize"):
        phase3_finalize(state)


if __name__ == "__main__":
    main()
