# Phase 4: EchoFrame Upstream GitHub Fit-Gap Review

## 1. Executive Summary
This report evaluates the upstream EchoFrame GitHub repository against the MNEMOS-native EchoFrame shadow implementation (`mnemos/experimental/echoframe_shadow/`). The review was conducted strictly as a read-only architectural analysis to determine the best integration strategy. Given that the MNEMOS-native implementation already achieved a ~92% token reduction with 0% failure and 1.0 stability on real workloads without external dependencies, the upstream repository introduces significant architectural complexity and dependency conflicts without a corresponding return on investment for MNEMOS's specific use case. The recommendation is to proceed with **Option A — Keep MNEMOS-native implementation only**.

## 2. Upstream EchoFrame Repository Overview
The upstream EchoFrame project operates as a highly complex "Evidence-based Context Compiler". It treats the context window as a constrained bandwidth channel utilizing dynamical control theory. Key components include:
- **Evidence IR**: A strict schema using Pydantic representing Scopes, Keyframes, Deltas, and Provenance.
- **Decompression Controller**: An algorithmic engine utilizing Hooke-inspired tension, Schreiner pressure mathematics, and JEPA (Joint-Embedding Predictive Architecture) to calculate semantic strain and discrete decompression allocations.
- **Domain Compilers**: Specialized AST-based parsers for coding environments (strictly pinned to Python 3.11.x) and schema mappers for Legal/Support.

## 3. MNEMOS-Native EchoFrame Implementation Overview
The MNEMOS-native implementation is a lightweight, zero-dependency shadow adapter deeply integrated into the existing `MnemosRuntime` pipeline. 
- It uses heuristic, string-based classification (`E0`-`E3`) instead of complex machine-learning embeddings for strain.
- It leverages existing MNEMOS constructs (e.g., `Engram` objects, Governance records) directly, avoiding the need for a secondary `Evidence IR` transformation layer.
- It successfully isolates hysteresis state via `session_id` using a simple rule-based dictionary.

## 4. Architecture Comparison
| Feature | Upstream EchoFrame | MNEMOS-Native Shadow |
| :--- | :--- | :--- |
| **Evidence Extraction** | Pydantic Schema, `ast` Compilers | String extraction, Heuristic Classifiers |
| **Semantic Scoring** | JEPA Latent Predictor (Cosine Sim) | Regex/Keyword Classification |
| **Hysteresis** | Dynamical Systems Math (Pressure) | Discrete turn-based state dictionary |
| **Dependencies** | `numpy`, `pydantic`, `jsonschema` | `json`, `uuid`, Python Stdlib |

## 5. Feature Fit-Gap Matrix
| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Evidence IR / scope model** | GAP | Upstream uses complex Pydantic models; MNEMOS relies on simple strings and `Engram` dicts. |
| **keyframe / delta model** | GAP | Upstream tracks diffs; MNEMOS natively minimizes context text without AST patching. |
| **decode levels** | MATCH | Upstream uses D0-D4; MNEMOS correctly mimics this with E0-E3. |
| **packet renderer** | PARTIAL_MATCH | Both render evidence minimally, but Upstream formats per domain schema. |
| **fact extraction** | PARTIAL_MATCH | MNEMOS uses heuristic extraction; Upstream uses dedicated domain compilers. |
| **semantic scoring** | GAP | JEPA vs keyword heuristic classification. |
| **hysteresis controller** | PARTIAL_MATCH | Both isolate multi-turn state efficiently. |
| **source-level evidence minimization** | MATCH | Both systems successfully minimize raw text retrieval bloat. |
| **provenance model** | PARTIAL_MATCH | MNEMOS uses direct source hashes; Upstream has a more involved `corpus_identity` class. |
| **governance signal preservation** | MATCH | Both strictly preserve governance flags and suppression vetoes. |
| **contradiction handling** | MATCH | Both bubble up detected contradictions cleanly. |
| **unknown / evidence-gap preservation** | MATCH | Both preserve missing-evidence warnings. |
| **multi-turn state isolation** | MATCH | Proper tracking across queries. |
| **telemetry / receipts** | MATCH | Standard JSON payload logging. |
| **configuration model** | PARTIAL_MATCH | MNEMOS native uses OS environ variables; Upstream uses complex config dicts. |
| **failure handling** | MATCH | Both implement fail-closed / fail-open capabilities safely. |
| **test harnesses** | MATCH | The `run_phase3f_soak.py` handles multi-turn soak validation perfectly. |
| **license compatibility** | NOT_APPLICABLE | Assuming internal access, but upstream poses risk via third-party libraries. |
| **dependency footprint** | CONFLICT | Upstream requires `numpy`, `pydantic`, forcing heavy footprint on MNEMOS. |

## 6. Code Reuse Candidates
* **`jepa.py` (Latent Predictor)**
  - *Location*: `echoframe_core/controller/jepa.py`
  - *Purpose*: Calculates semantic strain.
  - *Reuse Value*: Low. MNEMOS heuristics already achieved 92% token reduction.
  - *Recommendation*: **Reject**. Introduces `numpy` dependency unnecessarily.
* **`pressure.py` (Hysteresis Controller)**
  - *Location*: `echoframe_core/controller/pressure.py`
  - *Purpose*: Dynamical mathematics for context stability.
  - *Reuse Value*: Low. Our basic string hysteresis achieves 1.0 stability natively.
  - *Recommendation*: **Reject**.
* **AST Compilers**
  - *Purpose*: Diff tracking for code payloads.
  - *Reuse Value*: None.
  - *Recommendation*: **Reject**. Hard-pinned to Python 3.11.x; severe deployment blocker.

## 7. Incompatibility / Risk Areas
- **Python Version Locking**: Upstream requires exactly `3.11.x` to guarantee AST canonical formatting. This severely limits MNEMOS cross-environment deployment (e.g., SAM Windows/Mac nodes).
- **Redundant Governance**: The upstream repository attempts to model compliance and policy rules inside the Pydantic schemas, which conflicts directly with the authoritative `mnemos/governance/governor.py`.

## 8. Dependency and License Review
- **Dependencies**: `pydantic`, `numpy`, `jsonschema`.
- **Risk**: Adding `numpy` and `pydantic` heavily increases the Docker footprint and initialization latency of the MNEMOS runtime container, violating the local-first, minimal-overhead goals of the architecture. 

## 9. Security / Supply Chain Review
Importing an entirely new schema parser (Pydantic + JSONSchema) for AST code manipulation introduces parsing vulnerability risks (e.g., recursive AST depth limits), which is entirely avoided by MNEMOS's current string-chunk-based retrieval tier.

## 10. Test Coverage Comparison
MNEMOS's Phase 3F-R soak test provides a far more representative "real-world" multi-turn session simulation (100+ events) than the static unit-testing structures observed in the upstream repository's `tests/` directory.

## 11. Integration Options
- **Option A — Keep MNEMOS-native implementation only**
  - Retains full 92% token reduction and 1.0 stability without adding *any* dependencies or structural risk. Keeps MNEMOS as the definitive source of truth.
- **Option B — Selectively port small upstream components**
  - Unnecessary. The upstream features (like JEPA scoring) solve problems that MNEMOS's simple heuristics already solve adequately for our current scale.
- **Option C — Wrap EchoFrame as an optional external dependency**
  - Introduces API latency, IPC complexity, and dual-maintenance overhead.
- **Option D — Replace MNEMOS-native shadow code with upstream EchoFrame**
  - High risk. Would necessitate rewriting `retrieval_router.py` and `governor.py` to speak "Evidence IR" instead of native "Engrams", jeopardizing the entire Phase 2 release stability.
- **Option E — Reject upstream code but retain EchoFrame concepts**
  - Functionally similar to Option A, but Option A explicitly acknowledges that our native implementation *already* embodies the concepts securely.

## 12. Recommendation
Based on the evidence from the Phase 3F-R representative soak test, the MNEMOS-native shadow implementation provides all the token-compression benefits of EchoFrame without the architectural risks, dependency bloat, and Python-version pinning of the upstream repository. It seamlessly interoperates with MNEMOS's critical governance and provenance pipelines.

**RECOMMEND OPTION A:**
Keep MNEMOS-native implementation only.

## 13. Next-Step Implementation Plan
Given the recommendation for Option A, we should bypass any upstream porting and proceed immediately to **Phase 5 — Limited A/B Shadow Review**. This will expose the MNEMOS-native EchoFrame pipeline to live traffic, generating comparison telemetry against genuine user interaction traces.
