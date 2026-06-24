# GateMem G0 Environment

Date: 2026-06-24 (Pacific/Auckland)

Status: `GATEMEM_G0_PARTIAL_DELETION_GOVERNANCE_GAP_IDENTIFIED`

## Isolation boundary

GateMem was cloned as an external research dependency at:

```text
G:\MNEMOS-research\gatemem_upstream
```

Generated environment evidence is kept outside that clone at:

```text
G:\MNEMOS-research\gatemem_g0_artifacts
```

This placement deliberately differs from the suggested
`environment.freeze.txt` command. That filename is not ignored upstream, so
writing it in the clone would violate the clean-clone gate. GateMem's ignored
`.venv/` and `outputs/` directories remain inside the clone. No GateMem source,
data, prompt, or evaluator file was modified.

## Pinned upstream

| Field | Value |
|---|---|
| Repository | <https://github.com/rzhub/GateMem> |
| Commit | `603f9f4b4ba4b77f043c20f85687fa016fd720b0` |
| Branch | `main` |
| Commit time | `2026-06-21T15:46:56+08:00` |
| Clone time | `2026-06-24T09:06:33+12:00` |
| Source-code license | MIT (`LICENSE`) |
| Dataset license metadata | CC-BY-4.0 (`CITATION.cff`) |
| Dataset character | Synthetic; four domains |

The repository-level `LICENSE` and dataset-level `CITATION.cff` establish a
split license boundary: MIT for the software repository and CC-BY-4.0 for the
dataset. `DATASET_CARD.md` does not contain a separate explicit license
section. Future redistribution should preserve both notices and attribution.

## Host and Python environment

| Field | Value |
|---|---|
| OS | Microsoft Windows 11 Pro `10.0.22631`, 64-bit |
| Architecture | AMD64 |
| Python | CPython `3.11.9` |
| Virtual environment | `G:\MNEMOS-research\gatemem_upstream\.venv` |
| Requirements SHA-256 | `EB0A301FAE188952B6A82300983C8F2C67E536D24CC009C17A2F9F87F8CBBCD3` |
| Frozen package count | 83 |
| Freeze SHA-256 | `7CA390A207A761A7B3535660287762C6BA7E3590AF0B541D09762127E7F00B06` |
| Freeze location | `G:\MNEMOS-research\gatemem_g0_artifacts\environment.freeze.txt` |

GateMem's `requirements.txt` is unpinned. The captured freeze is therefore the
reproducible lock evidence for this G0 machine. Representative resolved
versions include `numpy==2.4.6`, `PyYAML==6.0.3`, `requests==2.34.2`,
`faiss-cpu==1.14.3`, `torch==2.12.1`, `transformers==5.12.1`, and
`sentence-transformers==5.6.0`.

Environment creation and core imports completed successfully:

```text
py -3.11 -m venv .venv                                  exit 0
python -m pip install --upgrade pip wheel setuptools    exit 0
pip install -r requirements.txt                         exit 0
python --version                                        Python 3.11.9
python -c "import numpy, yaml, requests; ..."            core-imports-ok
```

No hosted-model API key was configured. No LLM judge was run. Installation did
not require a model download beyond Python packages and the CPU Torch wheel.

## Required smoke checks

The two prescribed commands were run exactly. Both exited `1` before agent
construction because `run_eval.py` eagerly constructs its default OpenAI
router and requires `OPENAI_API_KEY`, including for the always-refuse example
agent:

```text
python bench/scripts/run_eval.py --data_dir bench/data/medical --agent example --run_name gatemem_g0_example_smoke
python bench/scripts/run_eval.py --data_dir bench/data/medical --agent long_context --run_name gatemem_g0_long_context_smoke
```

Failure: `bench.llm.router.LLMError: Missing API key. Please set environment
variable OPENAI_API_KEY before running.` This is an upstream CLI-default versus
G0 no-key constraint, not an environment or dependency failure. The output
directories exist but contain zero files.

Supplemental offline diagnostics added `--llm_provider stub`, used distinct run
names, completed all 579 medical checkpoints, and exited `0`:

| Agent | Run directory | Exit | Runtime |
|---|---|---:|---:|
| example | `outputs/gatemem_g0_example_stub_diagnostic` | 0 | 0.192 s |
| long_context | `outputs/gatemem_g0_long_context_stub_diagnostic` | 0 | 0.390 s |

These are pipeline smokes only. They are not MNEMOS results and not meaningful
agent-quality evidence: GateMem's stub uses hidden `expected_action`, while the
example agent always refuses.

Logs remain external:

```text
G:\MNEMOS-research\gatemem_g0_artifacts\example_smoke.log
G:\MNEMOS-research\gatemem_g0_artifacts\long_context_smoke.log
G:\MNEMOS-research\gatemem_g0_artifacts\example_stub_diagnostic.log
G:\MNEMOS-research\gatemem_g0_artifacts\long_context_stub_diagnostic.log
```

`git status --porcelain` was empty after environment setup and after every
smoke/diagnostic run.

## Attribution boundary

GateMem data, prompts, evaluator logic, outputs, and vendored code remain in
the external clone and are not copied into MNEMOS. Any future adapter must be
original MNEMOS code and preserve GateMem's MIT software notice, CC-BY-4.0
dataset attribution, and requested citation:

> Zhe Ren et al. “GateMem: Benchmarking Memory Governance in
> Multi-Principal Shared-Memory Agents.” 2026. <https://arxiv.org/abs/2606.18829>

