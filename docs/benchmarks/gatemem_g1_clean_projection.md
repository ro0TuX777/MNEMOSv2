# GateMem G1 Clean Input Projection and Normalizer

Date: 2026-06-24

Status: `GATEMEM_G1_CLEAN_INPUT_PROJECTION_GATE_PASS`

```text
GATEMEM_G1_CLEAN_INPUT_PROJECTION_AUTHORIZED
OFFLINE_ONLY
NO_RUNTIME_INTEGRATION
NO_UPSTREAM_GATEMEM_MODIFICATION
NO_HOSTED_JUDGE
NO_LEADERBOARD_SUBMISSION
NO_DELETION_CAPABILITY_CLAIM
```

## Implemented boundary

G1 is original MNEMOS research code under `prototype/gatemem_g1/`. It uses only
the Python standard library and imports neither GateMem nor MNEMOS runtime code.

```text
external GateMem episode + checkpoint
        |
        | allowlist only
        v
clean projection
  - episode/checkpoint identity
  - ordered visible turns through as_of_turn_id
  - requester principal identity and role
  - query text
  - separately supplied permitted metadata
        |
        | injected offline probes only
        v
retrieval/disclosure shadow observation
        |
        v
GateMem-compatible prediction row
```

No retriever, authorization service, answer model, MNEMOS client, network
transport, scorer, or judge is wired into the prototype.

## Evaluator isolation

The projector accesses only these checkpoint fields:

```text
checkpoint_id
episode_id
as_of_turn_id
asker.principal_id
asker.role
query_text
```

It never reads or enumerates:

```text
query_type
attack_type
expected_action
judge_spec
leak_targets
```

The raw checkpoint may contain those fields because they exist in released
GateMem JSON. Tests use a guarded mapping that raises if the projector attempts
to access one. Separately supplied permitted metadata is recursively rejected
if a scoring-only key appears at any depth.

The projector stops at `as_of_turn_id`, preserves source order, rejects missing
or duplicate turn IDs, and computes canonical SHA-256 over only permitted
input. GateMem's native `Checkpoint` class is not used.

## Shadow observation contract

`observe_shadow` accepts two injected callables:

1. an offline retrieval probe returning record IDs and ephemeral artifacts;
2. an offline disclosure probe returning `allowed`, `redacted`, or `denied`.

The boundary verifies unique retrieval IDs, prevents disclosure of an ID that
was not retrieved, and prevents denied decisions from identifying disclosed
records. This proves adapter mechanics only; no current MNEMOS retrieval or
authorization claim follows from an injected test probe.

## Deletion behavior

Deletion relevance is conservatively inferred from visible deletion or
historical-reference language in the query plus deletion language in prior
turns. No GateMem scoring label is consulted. This heuristic may miss or
over-classify cases and is not a deletion policy.

Two modes exist:

| Mode | Retrieval behavior | Normalized action | Claim |
|---|---|---|---|
| `unsupported` | Probe is not called | `refuse` | Explicitly unsupported |
| `simulated_shadow` | Offline probe may run; response content and used IDs are discarded | `refuse` | Simulation only |

Neither mode emits `no_memory` for a deletion-relevant observation. Each
prediction includes `deletion_capability_claim: false` and the explicit
evaluation status. `no_memory` is reserved for an empty eligible retrieval on
a non-deletion observation.

## Prediction normalization

The normalizer emits the GateMem external schema:

```json
{
  "checkpoint_id": "...",
  "action": "answer | answer_redacted | refuse | no_memory",
  "answer": "...",
  "answer_structured": {
    "gatemem_g1": {
      "shadow_only": true,
      "deletion_evaluation_status": "not_applicable | unsupported | simulated_shadow",
      "deletion_capability_claim": false
    }
  },
  "used_record_ids": []
}
```

Writers reject output paths inside the MNEMOS repository. Operators must also
keep the chosen external output path outside the pristine GateMem clone.
`tools/normalize_gatemem_g1_predictions.py` accepts only the exact frozen
observation schema; unknown fields, including scoring annotations, fail closed.

## Projection command

```powershell
python tools/run_gatemem_g1_projection.py `
  --episodes G:\MNEMOS-research\gatemem_upstream\bench\data\medical\episodes.jsonl `
  --checkpoints G:\MNEMOS-research\gatemem_upstream\bench\data\medical\checkpoints.jsonl `
  --output G:\MNEMOS-research\gatemem_g1_artifacts\medical_clean_projections.jsonl
```

The actual G1 run produced:

| Evidence | Value |
|---|---:|
| Clean projections | 579 |
| Unique projection digests | 579 |
| Turns per projection | 15–225 |
| Rows containing prohibited scoring fields | 0 |
| Artifact size | 15,858,818 bytes |
| Artifact SHA-256 | `decb4f6bbc27e3a0a86058dcb4881b65c89015ef16dc9b7a7b9b09456413a88e` |

The derived file remains external at:

```text
G:\MNEMOS-research\gatemem_g1_artifacts\medical_clean_projections.jsonl
```

No full-corpus retrieval observation or prediction file was produced because
G1 does not authorize or implement a real retriever/disclosure connection.

## Verification

The synthetic acceptance gate covers clean projection, future-turn exclusion,
scoring-field absence, determinism, metadata guards, observation mechanics,
prediction schema, deletion refusal behavior, invented-disclosure rejection,
and runtime/network import isolation.

```text
python -m pytest tests/test_gatemem_g1.py -q
python tools/run_gatemem_g1_gate.py
```

See `benchmarks/results/gatemem_g1_gate.md` and
`benchmarks/results/gatemem_g1_gate.json`.

## Advancement boundary

This PASS validates offline benchmark plumbing only. It does not authorize a
MNEMOS runtime adapter, GateMem agent, production route, hosted judge, external
scoring run, leaderboard submission, or deletion engineering. A true deletion
lifecycle requires its own ADR and authorization.
