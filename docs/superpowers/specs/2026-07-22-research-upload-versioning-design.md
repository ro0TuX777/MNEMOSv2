# Design — Duplicate and version handling for research uploads

Status: draft for review. No code written yet.
Date: 2026-07-22
Related: `e28a1c0` (Layer 1 exact-dedup, shipped),
`docs/superpowers/plans/2026-07-22-research-index-duplicate-purge.md`
(one-off cleanup of the pre-fix backlog).

## Problem

"Duplicate upload" is two problems that need opposite handling:

| | Same bytes | Different bytes |
|---|---|---|
| **Same name** | exact re-upload → reuse | **updated version → supersede** |
| **Different name** | copy/rename → reuse | new document → index |

Layer 1 (`_save_uploads`, shipped) covers the whole "same bytes" row via
content hash. The unsolved cell is **same name, different bytes**: a user
re-exports or edits `report.pdf` and uploads it again. Today it is stored as
`report-1.pdf`, indexed as a *separate* document, and the previous version's
engrams remain in the index competing in retrieval. This is worse than the
exact-dupe backlog because the stale content is a wrong/old version, not a
harmless copy.

Identity signals, ranked by trustworthiness:

- **Content sha256** — the only reliable identity signal; immune to metadata.
- **Filename** — weak; two distinct docs can share a name, and an update keeps
  the name.
- **Size** — cheap pre-filter only (different size ⇒ different content). Never
  identity. Already used by the cleanup tool to avoid hashing everything.
- **Date** — `file.lastModified` is client-controlled and resets on
  copy/download, so it can invert real age. Use it only to *order* versions and
  to inform a user prompt, never to decide identity. Pair with a server-side
  upload timestamp as the trustworthy clock.

Rule: **hash decides identity; size pre-filters; date only orders and informs.**

## Current wiring (verified)

- `POST /api/intake` (`tools/mnemos_research_ui.py:899`) → `_save_uploads`
  then `run_intake(files, project, capability, status, tags, ...)`.
  `project` and `capability` are already required form fields, so a
  project-scoped identity key is available at this boundary.
- `build_documents` (`tools/mnemos_research_intake.py:225`) assigns
  deterministic engram ids: `research::` + `sha256(f"{source_uri}\n{chunk_index}\n{chunk}")[:20]`,
  `source_uri = path.resolve().as_uri()` — so the id embeds the *filename*.
- Index delete path exists: `DELETE /v1/mnemos/engrams/<id>`
  (`service/app.py:2558`), removes from all tiers.
- There is no upload/index manifest today; nothing records what filename →
  which engram ids. This is the missing piece the design adds.

## Proposed scheme

### Manifest (foundation)

Add a small JSON manifest next to the upload dir, e.g.
`data/research_uploads/.manifest.json`, one record per stored document:

```
{
  "sha256": "...",            # content identity
  "stored_path": "constitution.pdf",
  "identity_key": "mnemos::constitution.pdf",   # project-scoped, normalized name
  "size": 123456,
  "uploaded_at": "2026-07-22T...Z",   # server clock, trustworthy
  "client_mtime": 1710000000,          # file.lastModified, informational
  "version": 3,
  "engram_ids": ["research::...", ...],  # what this version put in the index
  "superseded_by": null                  # sha256 of the version that replaced it
}
```

The manifest turns supersession from "reconstruct ids by re-chunking" (fragile,
the failure mode called out in the purge plan) into a lookup.

`engram_ids` are captured at index time: `build_documents` already computes
them deterministically, so `run_intake` returns them (small additive change to
its result dict) and the endpoint records them in the manifest.

### Decision at the upload boundary

Identity key = `f"{project}::{normalized_filename}"` (lowercased, trimmed).
Scoping by project removes most ambiguity: two unrelated `report.pdf`s in
different projects never collide.

| Hash match | Identity-key match | Meaning | Action |
|---|---|---|---|
| yes | — | exact duplicate | reuse stored file + engrams; no re-index (Layer 1) |
| no | no | new document | index normally |
| no | yes | **updated version** | index new content; delete the prior version's `engram_ids`; bump `version`; set old record's `superseded_by` |
| no | yes, but multiple candidates / cross-project name clash in same project | ambiguous | **prompt the user**: update existing vs. add as new — show size + dates to decide |

The ambiguous row is the only case needing a UI round-trip, and it is rare
(same name, same project, genuinely different document). Default on cancel is
the safe non-destructive choice: store as a new numbered document, index it,
leave the old one alone.

### Supersession = the purge machinery, scoped to one document

Retiring the old version reuses exactly the `DELETE /v1/mnemos/engrams/<id>`
loop from the index-purge plan, but bounded to one document's recorded
`engram_ids` — no reconstruction, no full rebuild. The two efforts share this
delete path.

## Optional follow-ups (not required for the core fix)

- **Layer 3 — near-duplicate advisory.** Exact-hash misses "same paper,
  re-exported with one metadata byte changed." A text fingerprint
  (SimHash/MinHash over extracted text, or a hash of normalized first-page
  text) flags likely duplicates for the user to confirm. Only worth building if
  re-exports prove common.
- **Stable-across-versions engram ids.** Deriving the id from
  `identity_key + chunk_index + chunk` instead of `source_uri` would make
  re-indexing an updated document *upsert* unchanged chunks and only require
  deleting trailing chunks when a doc shrinks — turning supersession into a
  clean diff. Changes retrieval/provenance semantics, so it needs its own
  review; the manifest approach above works without it.

## Frontend changes

- Send `file.size` and `file.lastModified` with each upload (currently only
  `size` is read, for the chip label, and neither is sent). Cheap; feeds the
  manifest and the ambiguity prompt.
- Add the confirmation dialog for the ambiguous row only. Everything else is
  server-side and silent.

## Verification

- Re-upload identical bytes → no new engrams (Layer 1 regression check).
- Upload an edited `report.pdf` in the same project → index count reflects new
  chunks, old version's `engram_ids` are gone from `GET /v1/mnemos/engrams/*`,
  manifest shows `version: 2` and `superseded_by` set on the old record.
- Upload same-named but different document across two projects → both indexed,
  no supersession.
- Retrieval no longer returns two versions of the same document.

## Recommendation / phasing

1. **Manifest + engram_ids capture** (additive, no behavior change).
2. **Version supersession** at the upload boundary (the core fix; reuses the
   purge delete path).
3. **Ambiguity prompt** frontend + `file.size`/`lastModified` plumbing.
4. Optional: Layer 3, stable engram ids.

Phases 1–2 deliver the actual retrieval-hygiene benefit; phase 3 covers the
rare ambiguous case; phase 4 is optimization.
