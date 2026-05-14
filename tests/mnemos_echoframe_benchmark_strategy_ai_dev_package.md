# MNEMOS × EchoFrame Evaluation: Benchmark Strategy and AI Dev Instructions

**Document Version:** v0.1  
**Prepared For:** MNEMOS AI Dev  
**Project Context:** Evaluation of EchoFrame as a candidate evidence-packet and context-throttle layer for MNEMOS  
**Primary Rule:** Baseline MNEMOS first; EchoFrame prototype second; comparative promotion decision last.

---

## Abstract

MNEMOS is the source-of-truth memory framework responsible for governed retrieval, evidence provenance, source traceability, contradiction handling, and memory lifecycle policy. EchoFrame is being evaluated as a possible read-path optimization layer that may sit after MNEMOS retrieval and governance filtering, but before final LLM prompt/context assembly.

EchoFrame should not be treated as a replacement for MNEMOS ingestion, storage, retrieval ownership, vector indexing, contradiction policy, or governance doctrine. Its candidate role is narrower: transform already-selected MNEMOS evidence into a deterministic, compressed, inspectable evidence packet that reduces prompt token load while preserving answer fidelity, provenance, and governance visibility.

The purpose of this work is to establish a before/after benchmark that measures the current MNEMOS read path, then compares it against an EchoFrame-style adapter under identical test conditions. EchoFrame should only be considered for promotion if it demonstrates meaningful token reduction without degrading evidence correctness, source traceability, contradiction handling, stale-evidence detection, or approval-required governance behavior.

---

## 1. Project Objective

Implement a benchmark-first evaluation package for determining whether an EchoFrame-style evidence codec is a good candidate for MNEMOS.

The AI Dev should:

1. Locate the current MNEMOS retrieval-to-prompt/context assembly path.
2. Build a reproducible baseline benchmark around the existing MNEMOS behavior.
3. Create or reuse a fixed evaluation corpus and fixed query set.
4. Capture baseline metrics before introducing EchoFrame behavior.
5. Implement EchoFrame only as an optional experimental read-path adapter.
6. Re-run the same benchmark with the adapter enabled.
7. Produce a comparative report with pass/fail gates.

The immediate deliverable is not a production EchoFrame integration. The immediate deliverable is a benchmark harness and evaluation report that determines whether EchoFrame merits further integration.

---

## 2. Architectural Positioning

### 2.1 Correct Integration Model

EchoFrame should be evaluated as a candidate module inside or alongside MNEMOS.

```text
User Query
  ↓
MNEMOS Retrieval
  ↓
MNEMOS Governance / Evidence Filtering
  ↓
EchoFrame Candidate Adapter
  ↓
Compressed / Structured Evidence Packet
  ↓
Prompt Assembly
  ↓
LLM / Agent Runtime
  ↓
Answer + Evidence Trace + Receipts
```

### 2.2 Incorrect Integration Model

Do not invert the architecture by making EchoFrame the wrapper around MNEMOS.

```text
EchoFrame
  ↓
Wraps or controls MNEMOS storage/retrieval/governance
```

This is out of scope. MNEMOS remains the authoritative memory framework.

---

## 3. Strict Scope Boundaries

### 3.1 EchoFrame May Touch

For the prototype only, EchoFrame-style logic may touch:

- Evidence packet rendering
- Context compression
- Decode-level selection
- Token-budget allocation
- Read-path packet metadata
- Optional stability / hysteresis scoring
- Experimental benchmark output

### 3.2 EchoFrame Must Not Touch Initially

EchoFrame must not modify or own:

- MNEMOS ingestion
- Canonical memory storage
- Engram source-of-truth records
- Vector store ownership
- Lexical index ownership
- Contradiction policy
- Memory lifecycle policy
- Governance doctrine
- Approval policy
- Durable production schemas
- Protected governance paths

Any later proposal to touch these areas requires a separate promotion package.

---

## 4. Benchmark Strategy

The benchmark must compare current MNEMOS behavior against MNEMOS plus the EchoFrame candidate adapter.

### 4.1 Phase A — Baseline MNEMOS

Measure the current MNEMOS read path without EchoFrame.

Capture:

- Query ID
- Query text
- Retrieved evidence IDs
- Source document IDs
- Chunk IDs / scope IDs where available
- Rendered context packet or prompt evidence block
- Input token count
- Output token count if available
- Latency for retrieval
- Latency for prompt/context assembly
- LLM answer if the harness invokes the model
- Citation/provenance completeness
- Evidence gaps
- Contradiction flags
- Stale-evidence flags
- Approval-required or escalation flags
- Any governance metadata already emitted

### 4.2 Phase B — EchoFrame Candidate Adapter

Insert the EchoFrame-style adapter after MNEMOS retrieval and governance filtering.

The adapter should accept selected evidence and produce a deterministic evidence packet.

Minimum packet fields:

```json
{
  "query_id": "string",
  "packet_id": "string",
  "renderer": "baseline|echoframe_candidate",
  "selected_evidence": [],
  "scope_packets": [],
  "omitted_evidence": [],
  "token_estimate": 0,
  "decode_decisions": [],
  "provenance_map": [],
  "governance_flags": [],
  "evidence_gaps": [],
  "contradictions": [],
  "stale_evidence": [],
  "explainability": {}
}
```

The adapter must preserve a mapping from every packeted claim back to the original MNEMOS evidence source.

### 4.3 Phase C — Comparative Evaluation

Run the same query set through both paths:

```text
Path 1: MNEMOS baseline
Path 2: MNEMOS + EchoFrame candidate adapter
```

The output report must compare quality, token load, latency, stability, provenance, and governance behavior.

---

## 5. Required Benchmark Categories

Start with 20–30 cases. Do not begin with hundreds of examples. The first benchmark should be small enough to inspect manually and strict enough to expose failure modes.

Required categories:

| Category | Purpose |
|---|---|
| Exact fact retrieval | Validate preservation of precise values, names, thresholds, dates, config keys, function names, or obligations |
| Multi-hop source resolution | Validate that definitions and references across documents survive packetization |
| Contradiction handling | Validate that conflicting evidence is surfaced, not flattened |
| Code/API recall | Validate preservation of signatures, classes, invariants, config flags, and failure modes |
| Policy/obligation recall | Validate preservation of must/shall/required language |
| Low-risk general query | Validate that the adapter does not over-decompress simple queries |
| High-risk governance query | Validate that high-risk or approval-required questions force higher-fidelity evidence packets |
| Repeated query stability | Validate reduction in evidence/context thrash across repeated or similar queries |
| Stale evidence case | Validate stale or superseded evidence is not silently treated as current |
| Insufficient evidence case | Validate that unknowns and evidence gaps are preserved |

---

## 6. Metrics to Capture

### 6.1 Fidelity Metrics

- Answer correctness
- Exact fact preservation
- Required evidence included
- Critical evidence omitted
- Unsupported claims introduced
- Contradictions preserved
- Unknowns preserved

### 6.2 Provenance Metrics

- Percentage of claims mapped to source evidence
- Source document IDs preserved
- Chunk/scope IDs preserved
- Citation completeness
- Evidence-gap completeness
- Stale evidence flag completeness

### 6.3 Efficiency Metrics

- Input token count
- Evidence packet token count
- Token reduction percentage
- Retrieval latency
- Packet rendering latency
- Total time to prompt-ready packet
- Optional TTFT if model execution is included

### 6.4 Stability Metrics

- Evidence set variance across repeated runs
- Packet variance across repeated runs
- Decode-level variance across repeated runs
- Context churn rate
- Omitted/included evidence oscillation

### 6.5 Governance Metrics

- Approval-required states preserved
- Contradictions preserved
- Evidence gaps preserved
- Stale evidence preserved
- Protected-domain warnings preserved
- Read-path decisions explained
- Receipts or audit entries emitted where applicable

---

## 7. EchoFrame Candidate Concepts to Prototype

The first prototype should be minimal. Avoid implementing the full EchoFrame paper immediately.

### 7.1 Evidence Scope

Represent retrieved evidence as scopes when possible.

A scope may map to:

- Source document section
- Code file / module
- Function or class
- Policy section
- Legal/compliance clause
- Governance doctrine section
- Existing MNEMOS evidence unit

### 7.2 Keyframe

A keyframe is the canonical structured representation of a scope.

Examples:

```json
{
  "scope_id": "string",
  "scope_type": "policy_section|code_module|contract_clause|memory_record|other",
  "summary": "string",
  "invariants": [],
  "obligations": [],
  "thresholds": [],
  "definitions": [],
  "api_signatures": [],
  "failure_modes": [],
  "source_refs": []
}
```

### 7.3 Delta

A delta is a minimal change or update relative to a keyframe.

For the first prototype, deltas may be represented but do not need a full patch engine.

```json
{
  "delta_id": "string",
  "scope_id": "string",
  "change_type": "added|modified|removed|superseded",
  "field": "string",
  "before": "string|null",
  "after": "string|null",
  "source_refs": []
}
```

### 7.4 Decode Levels

Use simple decode levels before implementing complex pressure dynamics.

Suggested initial levels:

| Level | Meaning |
|---|---|
| D0 | Scope title / identifier only |
| D1 | Short canonical summary |
| D2 | Structured fields needed for the query |
| D3 | High-fidelity evidence with direct quoted or near-source detail |
| D4 | Full raw supporting excerpt where required |

The benchmark should record why each scope received its decode level.

### 7.5 Hysteresis / Stability

Do not begin with complex math unless the codebase already supports it cleanly.

Initial version may implement simple stability rules:

- Avoid dropping previously selected high-value evidence too quickly.
- Keep high-risk evidence loaded longer across related turns.
- Require stronger reason to remove evidence than to add it.
- Log every retained, upgraded, downgraded, or omitted scope.

---

## 8. Pass/Fail Gates

EchoFrame should not proceed unless it passes the following gates.

| Gate | Requirement | Fail Condition |
|---|---|---|
| Gate A — Fidelity | No material loss in answer correctness | Correct baseline answer becomes incorrect, incomplete, or misleading |
| Gate B — Provenance | Every packeted claim maps back to MNEMOS source evidence | Claims appear without source mapping |
| Gate C — Governance Preservation | Evidence gaps, contradictions, stale flags, and approval-required states survive packetization | Governance signals are dropped or hidden |
| Gate D — Token Efficiency | Meaningful input-token reduction versus baseline | Token reduction is negligible or achieved only by unsafe omission |
| Gate E — Stability | Repeated/similar queries show lower or equal context thrash | Adapter increases evidence oscillation |
| Gate F — Explainability | Decode decisions are inspectable | No reason is recorded for packet shape or decode level |

Gates A, B, and C are non-negotiable. If any of these fail, stop the integration path and produce a failure report.

---

## 9. Suggested File/Directory Layout

Use the actual MNEMOS repository conventions where they already exist. If no convention exists, use the following draft layout.

```text
mnemos/
  experimental/
    echoframe_adapter/
      __init__.py
      adapter.py
      packet_schema.py
      scope_model.py
      decode_policy.py
      provenance_map.py
      stability_policy.py
      token_estimator.py
      report_writer.py

benchmarks/
  echoframe_eval/
    corpus/
    queries/
      eval_queries.jsonl
    expected/
      expected_evidence.jsonl
    outputs/
      baseline/
      echoframe_candidate/
    reports/
      comparison_report.md
      comparison_report.json
    run_baseline.py
    run_echoframe_candidate.py
    compare_results.py

tests/
  test_echoframe_adapter_packet_schema.py
  test_echoframe_adapter_provenance.py
  test_echoframe_adapter_decode_policy.py
  test_echoframe_benchmark_comparison.py
```

---

## 10. Required CLI Behavior

Add or provide scripts that support the following flow.

### 10.1 Run Baseline

```bash
python benchmarks/echoframe_eval/run_baseline.py \
  --queries benchmarks/echoframe_eval/queries/eval_queries.jsonl \
  --out benchmarks/echoframe_eval/outputs/baseline
```

### 10.2 Run EchoFrame Candidate

```bash
python benchmarks/echoframe_eval/run_echoframe_candidate.py \
  --queries benchmarks/echoframe_eval/queries/eval_queries.jsonl \
  --out benchmarks/echoframe_eval/outputs/echoframe_candidate
```

### 10.3 Compare Results

```bash
python benchmarks/echoframe_eval/compare_results.py \
  --baseline benchmarks/echoframe_eval/outputs/baseline \
  --candidate benchmarks/echoframe_eval/outputs/echoframe_candidate \
  --report-md benchmarks/echoframe_eval/reports/comparison_report.md \
  --report-json benchmarks/echoframe_eval/reports/comparison_report.json
```

---

## 11. Output Report Requirements

The comparison report must include:

1. Executive summary
2. Test corpus description
3. Query category coverage
4. Baseline metrics
5. EchoFrame candidate metrics
6. Token reduction summary
7. Fidelity comparison
8. Provenance comparison
9. Governance signal comparison
10. Stability comparison
11. Latency comparison
12. Failure cases
13. Manual inspection notes
14. Promotion recommendation
15. PASS/WARN/FAIL gate table

Suggested gate table:

| Gate | Status | Evidence | Notes |
|---|---|---|---|
| Gate A — Fidelity | PASS/WARN/FAIL | Link/path to result artifacts | Notes |
| Gate B — Provenance | PASS/WARN/FAIL | Link/path to result artifacts | Notes |
| Gate C — Governance Preservation | PASS/WARN/FAIL | Link/path to result artifacts | Notes |
| Gate D — Token Efficiency | PASS/WARN/FAIL | Link/path to result artifacts | Notes |
| Gate E — Stability | PASS/WARN/FAIL | Link/path to result artifacts | Notes |
| Gate F — Explainability | PASS/WARN/FAIL | Link/path to result artifacts | Notes |

---

## 12. Acceptance Criteria

The sprint is complete when:

1. Baseline MNEMOS benchmark runs successfully.
2. Baseline artifacts are emitted to disk.
3. EchoFrame candidate adapter can be enabled and disabled without affecting MNEMOS default behavior.
4. Candidate artifacts are emitted to disk.
5. Comparison script produces both Markdown and JSON reports.
6. Every packeted claim has a provenance mapping or is explicitly marked as unsupported/derived.
7. Evidence gaps, contradictions, stale evidence, and approval-required flags are preserved in candidate output.
8. Token counts are captured for both baseline and candidate paths.
9. At least 20 benchmark cases exist across the required categories.
10. Unit tests validate packet schema, provenance preservation, decode policy, and comparison logic.
11. The final report provides a clear recommendation: reject, continue sandbox, or promote to next integration phase.

---

## 13. Non-Goals

This sprint must not attempt to:

- Replace MNEMOS retrieval
- Replace MNEMOS storage
- Replace MNEMOS governance
- Introduce a new production vector store
- Rewrite ingestion
- Add production lifecycle policy changes
- Add autonomous mutation
- Modify protected governance paths
- Claim EchoFrame is production-ready
- Optimize for token reduction at the expense of fidelity

---

## 14. Recommended Promotion Logic

Use the following decision model:

```text
If Gates A, B, or C fail:
  Reject or keep as research only.

If A/B/C pass but D/E/F are weak:
  Continue sandbox evaluation.

If A/B/C pass and at least two of D/E/F pass:
  Promote to a deeper integration design package.

If all gates pass:
  Prepare Phase 1 integration plan for governed read-path adapter.
```

---

## 15. Developer Notes

- Keep the implementation deterministic wherever possible.
- Prefer JSONL artifacts for benchmark traces.
- Do not hide omitted evidence; log it.
- Do not collapse contradictions into a single synthesized answer.
- Preserve unknowns and evidence gaps.
- Treat token savings as secondary to correctness and provenance.
- Add feature flags so the EchoFrame candidate path is disabled by default.
- Make every decode-level decision inspectable.
- Do not promote without a written comparison report.

---

## 16. Suggested First Commit

The first commit should contain only benchmark scaffolding and baseline instrumentation.

Suggested commit title:

```text
Add MNEMOS EchoFrame baseline benchmark scaffold
```

Suggested first deliverables:

```text
benchmarks/echoframe_eval/queries/eval_queries.jsonl
benchmarks/echoframe_eval/run_baseline.py
benchmarks/echoframe_eval/compare_results.py
benchmarks/echoframe_eval/reports/README.md
```

EchoFrame adapter implementation should come after baseline output is available and reviewed.

---

## 17. Final Instruction to AI Dev

Do not begin by implementing EchoFrame. Begin by proving how MNEMOS behaves today.

The target sequence is:

```text
Baseline → Instrument → Measure → Adapter → Compare → Gate Decision
```

The success condition is not that EchoFrame is integrated. The success condition is that we can make an evidence-based decision about whether EchoFrame improves MNEMOS without weakening MNEMOS governance, provenance, or factual fidelity.
