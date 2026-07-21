# Plan — Purge duplicate engrams from the research index

Status: draft for review. No index mutation has been performed.
Date: 2026-07-22
Related commits: `ed48e5f` (demo panel), `e28a1c0` (upload dedup fix + tool)

## Problem

The research intake UI numbered re-uploads (`name-1.pdf`, `name-2.pdf`, ...)
keyed only on filename, so identical documents were stored, extracted,
chunked, and indexed multiple times. The upload directory has been cleaned
(587 → 287 files, one canonical copy per distinct content hash), and
`_save_uploads` now dedupes new uploads at the boundary.

The **index** was not touched. Qdrant (collection `mnemos_engrams`) and the
lexical tier still hold the chunks from all 300 removed duplicate files. Those
stale engrams keep crowding retrieval — the symptom that started this: a
receipt showing `coverage 0.20 / partial_evidence_cited` because the top-5
passages were duplicate copies of the same two sources.

Goal: remove exactly the stale engrams, leave every canonical engram intact,
and confirm retrieval no longer returns duplicate passages of one source.

## Key facts (verified in code)

- Engram id is deterministic — `tools/mnemos_research_intake.py:245`, `:266`:
  `research::` + `sha256(f"{source_uri}\n{chunk_index}\n{chunk}")[:20]`,
  where `source_uri = path.resolve().as_uri()`.
- Duplicate files shared byte-identical content but differed in filename, so
  they differ **only** in `source_uri`. Their chunk text and chunk_index are
  identical to the canonical file's.
- Therefore each stale id can be reconstructed from `(removed_filename,
  canonical_file_content)` without the deleted file being present.
- Delete path — `service/app.py:1824` `delete_engram` removes from every
  semantic tier and the lexical tier; route `DELETE /v1/mnemos/engrams/<id>`
  (`service/app.py:2558`). Lookup: `GET /v1/mnemos/engrams/<id>`
  (`service/app.py:2548`).
- There is **no** "list engrams" or "list by source" endpoint; enumeration
  must come from id reconstruction (Option A) or a full rebuild (Option B).
- Collection name: `mnemos_engrams`, overridable via
  `MNEMOS_QDRANT_COLLECTION` (`mnemos/config.py:60`, `:244`).

## Option A — Surgical delete (recommended)

Delete only the reconstructed stale ids. No rebuild; canonical engrams keep
their ids, scores, and snapshot lineage.

Steps:
1. Recover the removed→canonical map. Re-run the dedupe tool's grouping logic
   in report mode against a restored listing, OR reconstruct from naming
   (`constitution-1.pdf` → `constitution.pdf`). The dedupe dry-run already
   prints every `keep`/`would remove` pair; capture that list before it is
   lost. (Action item: the apply run was not logged to a file — regenerate the
   pairing from the numbering convention and verify each canonical exists.)
2. For each removed filename, build documents from the **canonical** file's
   bytes but with `source_uri` set from the removed filename, using the exact
   `build_documents` chunking (same `max_words`/`overlap_words` defaults).
   Collect the resulting `research::…` ids.
3. Pre-flight: `GET` a sample of those ids and confirm they exist; `GET` a
   sample of canonical ids and confirm they will be **kept** (not in the delete
   set). Abort if any canonical id appears in the delete set.
4. `DELETE` each stale id. Record counts per tier from the response.
5. Verify (below).

Pros: minimal blast radius, fast, reversible per-id (re-index a file to
restore). Cons: depends on id derivation exactly matching the original run —
mitigated by the pre-flight `GET` existence check in step 3.

Risk to check before running: confirm the original indexing used the same
chunk parameters now in `build_documents` (350/50). If any prior run used
different values, reconstructed ids won't match and step 3 will show the ids as
absent — in which case fall back to Option B.

## Option B — Re-index into a fresh collection (fallback)

If id reconstruction proves unreliable, rebuild clean:
1. Index the 287 canonical files into a new collection
   (`MNEMOS_QDRANT_COLLECTION=mnemos_engrams_v2`) via the batch indexer
   (`tools/mnemos_research_intake._index_documents_in_batches`).
2. Warm and smoke-test retrieval against the new collection.
3. Cut over the service env to the new collection; keep the old one until
   verification passes.
4. Drop the old collection.

Pros: guaranteed-clean slate, also clears any unrelated drift. Cons: heavier,
touches all engrams, temporary double storage, needs a service restart/cutover.

## Verification (both options)

- Re-run the original capture query ("Shortly after the ratification of the
  U.S. Constitution, the Founding generation added what?") with
  `retrieval_mode: semantic`. Expect the top hits to resolve to **distinct**
  sources; no three-identical-score cluster from one document.
- Spot-check 2–3 other multi-copy docs (e.g. `1706.03762.pdf`, ×6) return a
  single source, not repeats.
- `GET /v1/mnemos/stats` engram count drops by roughly the removed-chunk total
  (Option A) or matches a fresh 287-file index (Option B).
- Audit ledger shows the delete/index actions.

## Out of scope

- No change to retrieval routing (the `semantic`→`hybrid` auto-upgrade seen
  earlier is a separate question).
- No change to the demo frontend; its panel documents the pre-cleanup capture
  and explains the duplication, which remains accurate as a historical receipt.

## Recommendation

Proceed with Option A after the step-1 pairing is regenerated and the step-3
pre-flight confirms reconstructed ids exist and canonical ids are excluded.
Fall back to Option B if the pre-flight shows id mismatches.
