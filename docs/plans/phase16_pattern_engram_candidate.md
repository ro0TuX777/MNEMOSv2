# MNEMOS Phase 16–21: PatternEngramCandidate Extraction Harness

**Canonical repo:** `<repo-root>`
**Whitepaper:** `<repo-root>\docs\whitepaper.md` (v3.2)
**Paper basis:** ExpeL (conceptual), R²-Mem `2605.13486v1` (operational), From Storage to Experience (taxonomy), A-MEM (linking), Governing Evolving Memory (safety)
**Status:** Plan — not yet implemented
**Phase sequence:** 16 → 17 → 18 → 19 → 20 → 21

---

## Phase 0: Documentation Discovery (COMPLETE)

### Allowed APIs (verified against source)

**`PatternEngramCandidate`** — `mnemos/cognitive/pattern_engram.py` (fully implemented, 146 lines)
- Required: `pattern_summary: str`, `supporting_cycle_ids: List[str]`, `risk_if_wrong: str`
- Optional: `confidence_score`, `support_score`, `contradiction_score`, `pattern_type`, `recommended_scope`, `applies_when`, `does_not_apply_when`, `proposed_learning_class`
- Promotion lifecycle: `candidate` → `recommend_promotion(gate_id, confidence_threshold)` → `promotion_recommended` → `approve_promotion(governance_review_id, explicit_approval=True)` → `approved`
- Hardcoded invariants: `authoritative_for_retrieval=False`, `affects_ranking=False`, `mutates_policy=False`

**`CognitiveCycleRecord`** — `mnemos/cognitive/cycle.py`
- Fields: `cycle_id`, `trigger_type`, `query_or_event` (capped 240 chars), `attention_decisions: List[AttentionDecision]`, `retrieval_actions`, `reasoning_actions`, `governance_evaluations: List[GovernanceEvalSummary]`, `learning_writes: List[LearningWrite]`, `forecast_actions`, `forensic_ledger_refs`, `final_status`, `cycle_latency_ms`

**`CycleAssembler`** — `mnemos/cognitive/assembler.py`
- `add_learning_write(*, target_memory_type, operation, write_class, engram_id, delta_summary, triggered_by)`
- `add_reasoning_action(name, *, inputs, outputs, latency_ms, status, skip_reason)`
- `build(*, selected_route, final_status, outcome_observation_plan) → CognitiveCycleRecord`

**`ForecastOutcomeRecord`** — `mnemos/cognitive/forecast_outcome.py`
- Fields: `forecast_id`, `source_signal`, `forecast_type`, `predicted_condition`, `confidence_score`, `selected_action`, `actual_outcome`, `forecast_error_delta`, `operator_feedback`, `learning_recommendation`, `future_policy_candidate`
- Lifecycle: `resolve(actual_outcome, forecast_error_delta, operator_feedback, learning_recommendation, future_policy_candidate)`

**`learning_boundary.py`** write classes:
- `SEMANTIC_CANDIDATE_WRITE` — for `descriptive` pattern types (advisory, non-authoritative)
- `PROCEDURAL_CHANGE_CANDIDATE` — for `operational_recommendation` pattern types (advisory)
- `BLOCKED_PROCEDURAL_MUTATION` — for `*_mutation` pattern types (blocked)
- `classify_pattern_candidate(pattern_type: str) → LearningBoundaryDecision`

**Service integration** — `service/app.py`
- `app.py:623` — `cognitive_cycle: Optional[bool]` parameter
- `app.py:684–689` — `CycleAssembler` instantiation guard
- `app.py:1080` — ledger emit `operation="cognitive_cycle_emit"`
- `app.py:1731` — Flask route `GET /v1/mnemos/cognitive/cycles`

### Anti-patterns identified
- Do NOT create a plain `PatternEngram` (authoritative) class before Phase 20 — none exists yet; premature creation bypasses the governance gate
- Do NOT set `authoritative_for_retrieval=True` on any `PatternEngramCandidate` — it is hardcoded `False`
- Do NOT add `PatternEngramCandidate` objects to the main Qdrant/pgvector engram index — candidates are advisory; a separate store is required
- Do NOT use `approve_promotion(explicit_approval=False)` — raises `PermissionError`
- Do NOT import `LearningBoundaryDecision`, `ADVISORY_ONLY_WRITE_CLASSES`, or `AUTHORITATIVE_WRITE_CLASSES` from `mnemos.cognitive` — they are not in `__all__`; import from `mnemos.cognitive.learning_boundary` directly
- Do NOT add a 12th attention dimension to `build_attention_decisions()` without updating the 11-dimension test assertions in `tests/test_attention_contract.py`
- R²-Mem ADD/UPVOTE/DOWNVOTE/EDIT operations do NOT exist — those are ExpeL-only; R²-Mem's bank is static after offline construction

---

## Phase 16: Cycle Evaluation Rubric

**Goal:** Score `CognitiveCycleRecord` objects as high or low quality using a MNEMOS-adapted R²-Mem rubric. This is the offline evaluator — it produces `CycleEvaluationRecord` per cycle, with aggregate score and `good`/`bad`/`neutral` label.

### What to implement

**New file:** `mnemos/cognitive/cycle_evaluator.py`

**1. `CycleEvaluationRecord` (frozen dataclass)**
```python
@dataclass(frozen=True)
class CycleEvaluationRecord:
    cycle_id: str
    trigger_type: str
    rubric_scores: Dict[str, int]       # dimension_name -> 0..3
    aggregate_score: int                 # sum of rubric_scores values
    quality_label: str                   # "good" | "bad" | "neutral"
    reasons: Dict[str, str]             # dimension_name -> reason string
    evaluated_at: str                    # UTC ISO
    evaluator_version: str = "v1"
    
    def to_dict(self) -> Dict[str, Any]: ...
```

**2. MNEMOS rubric dimensions (adapted from R²-Mem §3.2 / Appendix D.1)**

Each dimension scored 0–3:

| Dimension | Scores 3 when | Scores 0 when |
|---|---|---|
| `routing_precision` | CLASS_A/B/C label matches `final_status=completed` and attention `query_classification` decision is consistent | Route mismatch: CLASS_C was used for a factoid with completed=true but returned 1 result |
| `candidate_efficiency` | `candidate_envelope` pressure is low; pool size ≤ 2× top_k | Candidate pool is 10× top_k without governance reducing it |
| `governance_appropriateness` | Governance mode matches query risk profile (enforced for high-risk, off for low-risk CLASS_A) | Advisory mode used when veto count > 0 but enforced was available |
| `forecast_utilization` | High-confidence forecast advisory was acted upon (pre_warm triggered or routing degraded appropriately) | High-confidence advisory emitted but routing was unchanged and latency degraded |
| `attention_coverage` | All 11 attention dimensions present in `attention_decisions` | Fewer than 7 dimensions present (incomplete cycle) |
| `learning_write_integrity` | All `learning_writes` carry valid `write_class` from `LEARNING_WRITE_CLASSES` | Any write with `write_class=None` or unknown class |

**Aggregate score:** sum across all applicable dimensions (max 18 = 6 × 3).

**Quality thresholds** (mirroring R²-Mem stable range `Klow=5, Khigh=10`):
- `score > 13` → `"good"`
- `score < 7` → `"bad"`
- `7 ≤ score ≤ 13` → `"neutral"` (discarded during extraction)

**3. `CycleEvaluator` class**
```python
class CycleEvaluator:
    KLOW: int = 7
    KHIGH: int = 13
    EVALUATOR_VERSION: str = "v1"

    def evaluate(self, cycle: CognitiveCycleRecord) -> CycleEvaluationRecord: ...
    def evaluate_batch(self, cycles: List[CognitiveCycleRecord]) -> List[CycleEvaluationRecord]: ...
    def filter_by_quality(
        self,
        records: List[CycleEvaluationRecord],
        quality: str,   # "good" | "bad"
    ) -> List[CycleEvaluationRecord]: ...
```

**Documentation references:**
- R²-Mem paper `2605.13486v1` §3.2 + Appendix D.1 — rubric structure and JSON schema
- `mnemos/cognitive/cycle.py` — `CognitiveCycleRecord`, `GovernanceEvalSummary`, `LearningWrite`, `AttentionDecision` field names
- `mnemos/cognitive/learning_boundary.py` — `LEARNING_WRITE_CLASSES` constant for `learning_write_integrity` dimension
- Test pattern reference: `tests/test_cognitive_cycle.py` lines 1–100 — `_make_assembler()` helper and class-per-type grouping

### Verification checklist
- [ ] `CycleEvaluator().evaluate(cycle)` returns `CycleEvaluationRecord` with all 6 dimension keys in `rubric_scores`
- [ ] `aggregate_score` equals `sum(rubric_scores.values())`
- [ ] Score ≥ 14 → `quality_label == "good"` (confirm threshold)
- [ ] Score ≤ 6 → `quality_label == "bad"`
- [ ] Score 7–13 → `quality_label == "neutral"`
- [ ] Cycle with `attention_decisions=[]` scores 0 on `attention_coverage`
- [ ] Cycle with all `write_class=None` learning_writes scores 0 on `learning_write_integrity`
- [ ] `evaluate_batch` returns one record per input cycle
- [ ] `filter_by_quality("good")` excludes `neutral` and `bad`
- [ ] `pytest tests/test_cycle_evaluator.py` passes (new file, minimum 20 tests)

### Anti-pattern guards
- Do NOT call an LLM inside `CycleEvaluator` — all scoring is deterministic rule-based
- Do NOT import from `mnemos.cognitive` directly for `LearningBoundaryDecision` — import from `mnemos.cognitive.learning_boundary`
- Do NOT score `ForecastOutcomeRecord` here — that is Phase 17 input

---

## Phase 17: Situation Abstractor + Pattern Learner

**Goal:** For each high/low quality `CycleEvaluationRecord`, produce a `PatternEngramCandidate` with a structured IF-THEN `pattern_summary`, entity-free `situation` description, and `applies_when` / `does_not_apply_when` guards. This is the Reflection Learner stage from R²-Mem (§3.3).

### What to implement

**New file:** `mnemos/cognitive/pattern_learner.py`

**1. `SituationAbstractor`** — deterministic (no LLM), extracts entity-free abstract situation from a cycle

Input: `CognitiveCycleRecord` + `CycleEvaluationRecord`

Output: `SituationSummary` (frozen dataclass):
```python
@dataclass(frozen=True)
class SituationSummary:
    trigger_type: str           # from cycle
    route_class: str            # "CLASS_A" | "CLASS_B" | "CLASS_C" | "unknown"
    governance_mode: str        # from cycle's governance_eval
    forecast_active: bool       # True if any forecast_actions present
    candidate_pressure: str     # "low" | "medium" | "high" (from working_memory_snapshot)
    weak_dimensions: List[str]  # rubric dimensions that scored 0 or 1
    strong_dimensions: List[str] # rubric dimensions that scored 3
    quality_label: str          # "good" | "bad"
    situation_text: str         # entity-free prose, e.g.:
                                # "A CLASS_B multi-hop search with advisory governance
                                #  where routing precision was weak and forecast was unused."
```

`situation_text` is template-assembled from the above fields — no LLM call.

**Template examples (modeled on R²-Mem Appendix E):**
- `"A {route_class} {trigger_type} with {governance_mode} governance where {', '.join(weak_dims)} {were|was} weak."`
- `"A {route_class} search where forecast was active but {', '.join(weak_dims)} degraded the outcome."`

**2. `PatternLearner`**

Input: `CognitiveCycleRecord` + `CycleEvaluationRecord` + `SituationSummary`

Output: `PatternEngramCandidate`

```python
class PatternLearner:
    def extract(
        self,
        cycle: CognitiveCycleRecord,
        evaluation: CycleEvaluationRecord,
        situation: SituationSummary,
    ) -> PatternEngramCandidate: ...
    
    def _build_if_then(
        self,
        situation: SituationSummary,
        evaluation: CycleEvaluationRecord,
    ) -> str: ...   # Returns the IF-THEN pattern_summary string
    
    def _infer_pattern_type(self, situation: SituationSummary) -> str: ...
    # Returns "descriptive" for good cycles
    # Returns "operational_recommendation" for bad cycles with routing/forecast weak dims
    # Never returns "*_mutation" types (those remain BLOCKED)
    
    def _compute_confidence(self, evaluation: CycleEvaluationRecord) -> float: ...
    # good cycles: confidence = aggregate_score / 18 (max)
    # bad cycles: confidence = 1.0 - (aggregate_score / 18)
```

**IF-THEN rule templates (adapted from R²-Mem `{situation, experience}` schema):**

For **good** cycles (positive pattern):
> `"IF a {route_class} query arrives with {active_conditions} THEN {strategy from strong_dimensions} produces high-quality results."`

For **bad** cycles (negative pattern / failure mode):
> `"IF a {route_class} query arrives with {triggering_conditions} THEN avoid {weak_dimensions_strategy}; instead apply {remediation_from_rubric_reasons}."`

**`PatternEngramCandidate` construction** (copy from `mnemos/cognitive/pattern_engram.py` constructor):
```python
PatternEngramCandidate(
    pattern_summary=if_then_string,
    supporting_cycle_ids=[cycle.cycle_id],
    risk_if_wrong=f"Incorrect routing or governance for {situation.route_class} queries",
    confidence_score=confidence,
    support_score=1.0 if quality_label == "good" else 0.0,
    contradiction_score=0.0 if quality_label == "good" else 1.0,
    pattern_type=inferred_pattern_type,
    recommended_scope="local",
    applies_when=situation.situation_text,
    does_not_apply_when=f"Query is not {situation.route_class}",
    proposed_learning_class=PATTERN_TYPE_TO_WRITE_CLASS[inferred_pattern_type],
)
```

**3. `PatternConsolidator`** — A-MEM-inspired deduplication and linking

After extracting a new candidate, compare against the existing candidate pool:
- If `situation_text` similarity > 0.85 (Jaccard on tokens): merge `supporting_cycle_ids`, increment `support_score`
- If `situation_text` similarity > 0.85 but quality labels conflict: add to `contradicting_engram_ids`
- Otherwise: treat as new candidate

```python
class PatternConsolidator:
    def consolidate(
        self,
        new_candidate: PatternEngramCandidate,
        existing_pool: List[PatternEngramCandidate],
    ) -> Tuple[PatternEngramCandidate, ConsolidationAction]: ...

ConsolidationAction = Literal["new", "merged", "contradicted"]
```

**Documentation references:**
- R²-Mem `2605.13486v1` Appendix E.2 — concrete IF-THEN examples and `{thinking, summary, situation, experience}` schema
- `mnemos/cognitive/pattern_engram.py` — `PatternEngramCandidate` constructor, `PATTERN_TYPES`, `PROMOTION_CANDIDATE`
- `mnemos/cognitive/learning_boundary.py` — `PATTERN_TYPE_TO_WRITE_CLASS` dict (import directly, not via `__all__`)
- `mnemos/cognitive/cycle.py` — `WorkingMemorySnapshot` fields for candidate_pressure derivation

### Verification checklist
- [ ] `SituationAbstractor` produces non-empty `situation_text` for all `CognitiveCycleRecord` inputs
- [ ] `situation_text` contains no entity names from `query_or_event` (entity-free check)
- [ ] `PatternLearner.extract()` always returns `PatternEngramCandidate` with `promotion_status == "candidate"`
- [ ] `PatternLearner` never produces `pattern_type` of `"policy_mutation"`, `"routing_mutation"`, or `"template_mutation"` (these are blocked)
- [ ] `PatternEngramCandidate.authoritative_for_retrieval` is always `False` (hardcoded invariant)
- [ ] Good cycle → `support_score ≥ 0.7`; bad cycle → `contradiction_score ≥ 0.7`
- [ ] `PatternConsolidator` merges near-duplicate situations (shared `supporting_cycle_ids` grows)
- [ ] `PatternConsolidator` surfaces contradicted pairs (different quality labels, same situation)
- [ ] `pytest tests/test_pattern_learner.py` passes (new file, minimum 25 tests)
- [ ] Confirmed: `classify_pattern_candidate(pattern_type)` from `mnemos.cognitive.learning_boundary` returns matching `write_class` for all generated candidates

### Anti-pattern guards
- Do NOT call an LLM inside `SituationAbstractor` — template assembly only
- Do NOT use `pattern_type="policy_mutation"` or any `*_mutation` type from `PatternLearner` — `_infer_pattern_type` must never return these
- Do NOT merge candidates with different `pattern_type` values in `PatternConsolidator`
- Do NOT mutate existing `PatternEngramCandidate` objects — they are `@dataclass` (mutable), but `PatternConsolidator` must return new instances via `replace()` or reconstruction

---

## Phase 18: Pattern Candidate Store + Offline Accumulation Runner

**Goal:** Persist `PatternEngramCandidate` objects to a dedicated advisory store (separate from the main engram index), and provide an offline runner that processes cycle history to accumulate candidates.

### What to implement

**New file:** `mnemos/cognitive/pattern_store.py`

**`PatternCandidateStore`** — in-memory store with JSON persistence

```python
class PatternCandidateStore:
    def __init__(self, persist_path: Optional[str] = None): ...
    
    # Write (advisory only — never touches main engram index)
    def add(self, candidate: PatternEngramCandidate) -> str: ...        # returns candidate_id
    def update(self, candidate_id: str, candidate: PatternEngramCandidate) -> None: ...
    
    # Read
    def get(self, candidate_id: str) -> Optional[PatternEngramCandidate]: ...
    def list_by_status(self, status: str) -> List[PatternEngramCandidate]: ...
    def list_by_pattern_type(self, pattern_type: str) -> List[PatternEngramCandidate]: ...
    def list_all(self) -> List[PatternEngramCandidate]: ...
    
    # Advisory recall (Phase 19 uses this)
    def find_relevant(
        self,
        situation_text: str,
        top_k: int = 3,
        min_confidence: float = 0.6,
    ) -> List[PatternEngramCandidate]: ...
    # Uses Jaccard similarity on situation_text tokens — no embedding call
    
    # Persistence
    def save(self) -> None: ...    # writes to persist_path as JSON
    def load(self) -> None: ...    # reads from persist_path
    def to_dict(self) -> Dict[str, Any]: ...
```

**New file:** `tools/run_pattern_accumulation.py` — offline accumulation runner

```python
# Usage:
# python tools/run_pattern_accumulation.py [--dry-run] [--limit N] [--since ISO_DATE]
#
# Reads cycle history from GET /v1/mnemos/cognitive/cycles (or from a local JSON file)
# Evaluates each cycle with CycleEvaluator
# Extracts PatternEngramCandidate for good/bad cycles
# Consolidates via PatternConsolidator
# Persists to PatternCandidateStore
# Emits a JSON report: pattern_accumulation_<timestamp>.json
```

**Report schema:**
```json
{
  "run_id": "uuid",
  "ran_at": "ISO",
  "cycles_evaluated": 42,
  "good_cycles": 18,
  "bad_cycles": 9,
  "neutral_cycles": 15,
  "candidates_new": 12,
  "candidates_merged": 5,
  "candidates_contradicted": 2,
  "candidates_total": 27,
  "dry_run": false,
  "gate": {
    "min_candidates_required": 5,
    "passed": true
  }
}
```

**Phase gate (integrated into runner):**
- Gate passes if: `candidates_new + candidates_merged ≥ 5` on a batch of ≥ 10 evaluated cycles
- Gate fails if: zero candidates extracted or all cycles are `neutral`

**Documentation references:**
- `service/app.py:1162–1174` — `get_cognitive_cycles()` method (pattern to replicate in runner for fetching cycle history)
- R²-Mem `2605.13486v1` Algorithm 1 — offline experience bank construction loop (adapt for MNEMOS cycle batches)
- `mnemos/cognitive/pattern_engram.py:140–146` — `to_dict()` for JSON persistence

### Verification checklist
- [ ] `PatternCandidateStore.add()` stores candidate and returns valid UUID
- [ ] `PatternCandidateStore.find_relevant("CLASS_B multi-hop", top_k=3)` returns ≤ 3 candidates sorted by similarity
- [ ] `PatternCandidateStore.save()` + `load()` round-trips: all candidate fields preserved
- [ ] `run_pattern_accumulation.py --dry-run` produces report JSON without writing to store
- [ ] `run_pattern_accumulation.py --limit 20` processes exactly 20 cycles
- [ ] Report gate passes when ≥ 5 candidates extracted from ≥ 10 cycles
- [ ] Store `list_by_status("candidate")` never returns `status="approved"` entries
- [ ] `pytest tests/test_pattern_store.py` passes (new file, minimum 20 tests)
- [ ] `benchmarks/results/pattern_accumulation_<timestamp>.json` artifact generated from runner

### Anti-pattern guards
- Do NOT call `MnemosRuntime.index()` or any Qdrant/pgvector write path for candidates — `PatternCandidateStore` is a separate advisory store only
- Do NOT include candidates with `pattern_type` in `BLOCKED_PROCEDURAL_MUTATION` types in any store write — they can be accumulated and surfaced but must be labeled clearly as blocked
- Do NOT persist `PatternEngramCandidate.authoritative_for_retrieval` as `True` in JSON — validation check in `load()`

---

## Phase 19: Advisory Recall Integration

**Goal:** At search time, when `cognitive_cycle=True`, retrieve relevant `PatternEngramCandidate` objects and surface them as advisory hints in the `CognitiveCycleRecord`. No auto-promotion, no routing mutation.

### What to implement

**Modify:** `mnemos/cognitive/assembler.py` — add `add_advisory_patterns()` method

```python
# Add to CycleAssembler (after existing add_* methods):
def add_advisory_patterns(
    self,
    candidates: List["PatternEngramCandidate"],
) -> None:
    """Record advisory PatternEngramCandidates recalled for this cycle. Zero-cost if empty."""
    self._advisory_patterns: List[Dict[str, Any]] = [
        {
            "candidate_id": c.candidate_id,
            "pattern_summary": c.pattern_summary,
            "pattern_type": c.pattern_type,
            "confidence_score": round(c.confidence_score, 4),
            "promotion_status": c.promotion_status,
            "applies_when": c.applies_when,
        }
        for c in candidates
        if c.promotion_status in (PROMOTION_CANDIDATE, PROMOTION_RECOMMENDED)
    ]
```

**Modify:** `mnemos/cognitive/cycle.py` — add `advisory_patterns` field to `CognitiveCycleRecord`

```python
# Add to CognitiveCycleRecord dataclass (after learning_writes):
advisory_patterns: List[Dict[str, Any]] = field(default_factory=list)
# Schema: [{candidate_id, pattern_summary, pattern_type, confidence_score, promotion_status, applies_when}]
# Advisory only — never modifies routing, governance, or ranking
```

**Modify:** `service/app.py:684–689` block — wire advisory recall

```python
# In the cognitive_cycle guard block (app.py ~684–689):
# After instantiating _assembler, before first add_* calls:
if cognitive_cycle and _pattern_store is not None:
    _situation = _situation_abstractor.from_query(query, active_profile=_profile)
    _advisory = _pattern_store.find_relevant(_situation, top_k=3, min_confidence=0.6)
    if _advisory:
        _assembler.add_advisory_patterns(_advisory)
```

Note: `_pattern_store` is optional — if not configured (`MNEMOS_PATTERN_STORE_PATH` unset), recall is silently skipped. This keeps the default path zero-cost.

**New env var:**
```
MNEMOS_PATTERN_STORE_PATH=<path-to-store.json>   # default: None (feature disabled)
```

**New attention dimension (12th)** — `pattern_advisory`:
- Add to `build_attention_decisions()` in `mnemos/cognitive/attention.py`
- Decision: `"recalled:{n} candidates"` or `"no_store_configured"` or `"no_relevant_candidates"`
- **Requires updating test assertions** in `tests/test_attention_contract.py` for the 11-dimension count

**Documentation references:**
- `service/app.py:684–689` — existing `CycleAssembler` instantiation block (exact integration point)
- `mnemos/cognitive/assembler.py` — all `add_*` method signatures for consistent naming
- `mnemos/cognitive/attention.py` — `build_attention_decisions()` function (add 12th dimension here)
- `tests/test_attention_contract.py` — update assertion for dimension count (11 → 12)
- R²-Mem `2605.13486v1` Algorithm 2 — online experience recall (Top-K cosine similarity retrieval; adapt as Jaccard for MNEMOS deterministic path)

### Verification checklist
- [ ] `CognitiveCycleRecord.advisory_patterns` defaults to `[]` — zero cost when store not configured
- [ ] `add_advisory_patterns([])` results in empty `advisory_patterns` list (not absent key)
- [ ] `advisory_patterns` entries never include `promotion_status="approved"` candidates (those are PatternEngrams, Phase 20)
- [ ] Service returns `"advisory_patterns": []` in cognitive_cycle response when `MNEMOS_PATTERN_STORE_PATH` is not set
- [ ] Service returns ≤ 3 advisory candidates when store is configured
- [ ] Routing decisions, governance mode, and ranking are unchanged whether or not advisory_patterns is populated (confirmed via diff test)
- [ ] `tests/test_attention_contract.py` updated: 11 → 12 dimension assertions pass
- [ ] `pytest tests/test_pattern_recall.py` passes (new file, minimum 15 tests)
- [ ] `GET /v1/mnemos/capabilities` reflects `"pattern_store_enabled": bool` in response

### Anti-pattern guards
- Do NOT use advisory candidates to modify `selected_route` or any governance decision — advisory_patterns is read-only context
- Do NOT add `advisory_patterns` to the forensic ledger payload — only `candidate_id` list as a metadata annotation is acceptable to avoid ledger bloat
- Do NOT block search requests when `_pattern_store.find_relevant()` raises — wrap in try/except, log warning, continue with empty candidates

---

## Phase 20: Promotion Governance Gate + PatternEngram

**Goal:** Define the authoritative `PatternEngram` class (promoted from candidate), implement the governed promotion workflow with forensic ledger tracing, and expose review/approval endpoints.

### What to implement

**New file:** `mnemos/cognitive/promoted_pattern.py`

```python
@dataclass
class PatternEngram:
    """
    Authoritative pattern engram. Only reachable via explicit governance approval.
    Constructed exclusively via PatternEngram.from_approved_candidate().
    """
    pattern_id: str                     # from approved candidate_id
    pattern_summary: str
    pattern_type: str                   # "descriptive" | "operational_recommendation" only
    applies_when: str
    does_not_apply_when: str
    risk_if_wrong: str
    confidence_score: float
    supporting_cycle_ids: List[str]
    supporting_engram_ids: List[str]
    contradicting_engram_ids: List[str]
    governance_review_id: str           # required — from approve_promotion()
    promoted_from_candidate_id: str
    promoted_at: str                    # UTC ISO
    authoritative_for_retrieval: bool = False   # Still False — advisory, not index-inserted
    write_class: str = SEMANTIC_WRITE           # SEMANTIC_WRITE for descriptive
    
    # Class method — the ONLY constructor path
    @classmethod
    def from_approved_candidate(
        cls,
        candidate: PatternEngramCandidate,
        *,
        governance_review_id: str,
    ) -> "PatternEngram":
        if candidate.promotion_status != PROMOTION_APPROVED:
            raise PermissionError("Candidate must be in PROMOTION_APPROVED state")
        ...
    
    def to_dict(self) -> Dict[str, Any]: ...
```

**Important:** `PatternEngram.authoritative_for_retrieval` remains `False` — authoritative `PatternEngram` objects are surfaced via dedicated endpoints, not injected into the Qdrant/pgvector retrieval path. This can be revisited in a future phase with an explicit governance gate.

**Modify:** `mnemos/cognitive/pattern_store.py` — add promoted engram tracking

```python
# Add to PatternCandidateStore:
def promote(
    self,
    candidate_id: str,
    *,
    governance_review_id: str,
    ledger_ref: Optional[str] = None,
) -> "PatternEngram": ...

def list_promoted(self) -> List["PatternEngram"]: ...
```

**New service endpoints** — add to `service/app.py`:

```
GET    /v1/mnemos/cognitive/candidates             — list PatternEngramCandidates by status
GET    /v1/mnemos/cognitive/candidates/{id}        — get one candidate
POST   /v1/mnemos/cognitive/candidates/{id}/recommend  — set promotion_recommended
POST   /v1/mnemos/cognitive/candidates/{id}/approve    — promote to PatternEngram
POST   /v1/mnemos/cognitive/candidates/{id}/reject     — reject candidate
GET    /v1/mnemos/cognitive/patterns               — list promoted PatternEngrams
```

**Forensic ledger events** (new `operation` values):
- `"pattern_candidate_recommended"` — when `recommend_promotion()` is called
- `"pattern_candidate_approved"` — when `approve_promotion()` is called
- `"pattern_candidate_rejected"` — when rejected
- All events: metadata = `{candidate_id, pattern_type, governance_review_id}`

**Documentation references:**
- `mnemos/cognitive/pattern_engram.py:95–115` — `recommend_promotion()` and `approve_promotion()` method bodies (copy promotion state machine exactly)
- `mnemos/cognitive/pattern_engram.py:117–128` — safety invariant properties (replicate in `PatternEngram`)
- `service/app.py:1731–1743` — `GET /v1/mnemos/cognitive/cycles` Flask route (use as template for new cognitive endpoints)
- `service/app.py:1080` — ledger emit pattern for new `operation` values
- `mnemos/cognitive/learning_boundary.py` — `SEMANTIC_WRITE` for `write_class` of promoted descriptive patterns

### Verification checklist
- [ ] `PatternEngram.from_approved_candidate()` raises `PermissionError` if `promotion_status != "approved"`
- [ ] `PatternEngram.authoritative_for_retrieval` is `False` (hardcoded — confirm in test)
- [ ] `POST /v1/mnemos/cognitive/candidates/{id}/approve` returns 400 if candidate is not in `promotion_recommended` state
- [ ] `POST /v1/mnemos/cognitive/candidates/{id}/approve` without `governance_review_id` body field returns 400
- [ ] Ledger records `"pattern_candidate_approved"` event with `candidate_id` and `governance_review_id` in metadata
- [ ] `GET /v1/mnemos/cognitive/patterns` returns promoted `PatternEngram` objects
- [ ] Promoted `PatternEngram` does NOT appear in `GET /v1/mnemos/search` results (not in engram index)
- [ ] `pytest tests/test_promoted_pattern.py` passes (new file, minimum 20 tests)
- [ ] `pytest tests/test_pattern_endpoints.py` passes (new file, minimum 15 endpoint tests)

### Anti-pattern guards
- Do NOT insert `PatternEngram` objects into Qdrant or pgvector — they go to `PatternCandidateStore` promoted pool only
- Do NOT allow `PatternEngram.from_approved_candidate()` to bypass the `PROMOTION_APPROVED` check via keyword arg — the guard is unconditional
- Do NOT expose a bulk-approve endpoint — each candidate must be approved individually with a `governance_review_id`
- Do NOT set `write_class=PROCEDURAL_CHANGE_CANDIDATE` on a `PatternEngram` — promoted engrams use `SEMANTIC_WRITE`

---

## Phase 21: End-to-End Validation Harness + Whitepaper

**Goal:** Phase gate validation harness (mirroring `tools/run_coala_cycle_validation.py`), CI integration, and whitepaper §4.11 addition.

### What to implement

**New file:** `tools/run_pattern_phase_gate.py`

8 validation scenarios (mirroring `run_coala_cycle_validation.py`'s 8 CoALA paths):

| Scenario | What it validates |
|---|---|
| `eval_good_class_a` | Good CLASS_A cycle → `CycleEvaluator` → `quality_label="good"` |
| `eval_bad_class_b` | Bad CLASS_B cycle with weak routing → `quality_label="bad"` |
| `learner_descriptive` | Good cycle → `PatternLearner` → `PatternEngramCandidate` with `pattern_type="descriptive"` |
| `learner_operational` | Bad cycle with forecast unused → `pattern_type="operational_recommendation"` |
| `consolidator_merge` | Two near-duplicate situations → merged candidate with 2 `supporting_cycle_ids` |
| `consolidator_contradict` | Same situation, opposite quality → `contradicting_engram_ids` populated |
| `recall_advisory` | Store populated → `find_relevant()` → top-3 advisory candidates returned |
| `promotion_gate` | Full lifecycle: candidate → recommend → approve → `PatternEngram` with ledger refs |

**Gate assertions (validated by harness):**
- `evaluator_determinism`: same cycle → same score on repeated calls
- `safety_invariant`: no candidate in any state has `authoritative_for_retrieval=True`
- `promotion_boundary`: `from_approved_candidate()` raises on non-approved input
- `blocked_types_never_promoted`: `policy_mutation`/`routing_mutation`/`template_mutation` never reach `PatternEngram`
- `ledger_traceability`: every promotion event has a forensic ledger ref

**New tests (minimum counts):**
- `tests/test_cycle_evaluator.py` — 20 tests
- `tests/test_pattern_learner.py` — 25 tests
- `tests/test_pattern_store.py` — 20 tests
- `tests/test_pattern_recall.py` — 15 tests
- `tests/test_promoted_pattern.py` — 20 tests
- `tests/test_pattern_endpoints.py` — 15 tests
- Total new tests: ≥ 115

**Whitepaper updates:**
- Bump version to `3.3` in header
- Add changelog row for `2026-0X-XX` PatternEngramCandidate phase
- Add `§4.11 PatternEngramCandidate Extraction Harness` (after existing §4.10 CoALA Cognitive Cycle)
- Update `§4.10` safety invariants list to reference Phase 20's promotion gate
- Update provenance capability table with pattern candidate rows

**`§4.11` content outline:**
```
§4.11 PatternEngramCandidate Extraction Harness (v3.3)
- Paper basis: ExpeL + R²-Mem + Governing Evolving Memory
- Pipeline diagram: cycle history → evaluator → learner → candidate store → advisory recall → governed promotion
- CycleEvaluator: rubric dimensions, quality thresholds
- PatternLearner: situation abstractor, IF-THEN rule construction
- PatternCandidateStore: advisory-only, separate from engram index
- Advisory recall: cognitive_cycle=true adds advisory_patterns to response
- Promotion governance gate: recommend → approve → PatternEngram
- Safety invariants (preserved from v3.2)
- New endpoints table
- Phase 21 gate evidence: benchmarks/results/pattern_phase_gate.json
```

### Verification checklist
- [ ] `python tools/run_pattern_phase_gate.py` exits 0 with all 8 scenarios passing
- [ ] `benchmarks/results/pattern_phase_gate.json` artifact written
- [ ] `pytest tests/test_cycle_evaluator.py tests/test_pattern_learner.py tests/test_pattern_store.py tests/test_pattern_recall.py tests/test_promoted_pattern.py tests/test_pattern_endpoints.py` — all ≥ 115 tests pass
- [ ] `pytest tests/test_cognitive_cycle.py tests/test_attention_contract.py tests/test_learning_boundary.py` — existing tests still pass (no regressions)
- [ ] Whitepaper `§4.11` added, version bumped to `3.3`, changelog row present
- [ ] `GET /v1/mnemos/capabilities` returns `"pattern_harness": {"enabled": bool, "store_configured": bool}`
- [ ] CI gate added: `tools/run_pattern_phase_gate.py --fail-on-gate` (mirror of `run_wave4_hygiene.py --fail-on-gate`)

### Anti-pattern guards
- Do NOT merge `test_pattern_endpoints.py` tests into `test_cognitive_cycle.py` — keep test files per-component
- Do NOT bump whitepaper to `v4.0` — this is an additive overlay phase; `v3.3` is correct
- Do NOT remove the `§4.10` safety invariants section when adding `§4.11` — update, don't replace

---

## Execution Order

```
Phase 16 → Phase 17 → Phase 18 → Phase 19 → Phase 20 → Phase 21
```

Each phase is self-contained. Phases 16–17 have no service integration and can be tested in isolation. Phase 18 introduces the offline runner. Phase 19 requires Phase 18 (pattern store). Phase 20 requires Phase 17 (PatternEngramCandidate) and Phase 18 (store). Phase 21 validates all prior phases.

Phases 16–17 can be executed in the same session. Phases 18–19 form a natural second session. Phases 20–21 form the final session.

---

## File Map (all new files)

```
mnemos/cognitive/
  cycle_evaluator.py        Phase 16 — CycleEvaluationRecord, CycleEvaluator
  pattern_learner.py        Phase 17 — SituationSummary, SituationAbstractor, PatternLearner, PatternConsolidator
  pattern_store.py          Phase 18 — PatternCandidateStore
  promoted_pattern.py       Phase 20 — PatternEngram

mnemos/cognitive/__init__.py     Phases 16-20 — extend __all__ incrementally

mnemos/cognitive/assembler.py    Phase 19 — add add_advisory_patterns()
mnemos/cognitive/cycle.py        Phase 19 — add advisory_patterns field to CognitiveCycleRecord
mnemos/cognitive/attention.py    Phase 19 — add 12th dimension "pattern_advisory"

service/app.py                   Phase 19-20 — pattern store wiring + new endpoints

tools/
  run_pattern_accumulation.py    Phase 18 — offline runner
  run_pattern_phase_gate.py      Phase 21 — validation harness

tests/
  test_cycle_evaluator.py        Phase 16
  test_pattern_learner.py        Phase 17
  test_pattern_store.py          Phase 18
  test_pattern_recall.py         Phase 19
  test_promoted_pattern.py       Phase 20
  test_pattern_endpoints.py      Phase 20

benchmarks/results/
  pattern_accumulation_<ts>.json Phase 18 artifact
  pattern_phase_gate.json        Phase 21 gate artifact

docs/whitepaper.md               Phase 21 — §4.11 + v3.3 bump
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Evaluator type | Deterministic rubric (no LLM) | MNEMOS is a service; LLM dependency in eval loop would add latency and external dependency |
| Situation abstractor | Template-based (no LLM) | Entity-free situation text is achievable from structured cycle fields without an LLM |
| Experience bank similarity | Jaccard on tokens | No embedding call needed for candidate recall; avoids GPU dependency in advisory path |
| R²-Mem bank staticness | Offline batch only | Matches R²-Mem's static bank model; online mutation deferred to future phase |
| ExpeL UPVOTE/DOWNVOTE | Not implemented | These operations are ExpeL-only; R²-Mem uses threshold filtering — simpler and sufficient |
| PatternEngram retrieval | Not index-inserted | Authoritative PatternEngrams go to promoted pool only; index insertion is a future governance gate decision |
| Promotion flow | `candidate` → `recommended` → `approved` | Existing `PatternEngramCandidate` lifecycle; Phase 20 adds `PatternEngram` as terminal promoted form |
