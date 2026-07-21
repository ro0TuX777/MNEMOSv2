# Engram AI Dev Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `%TEMP%` artifact bundle that equips a fresh AI Dev to understand why Engram is splitting from MNEMOS, what MNEMOS architecture to inspect, and how to begin building Engram.

**Architecture:** Produce a small handoff bundle outside the repository with one lead overview and four focused companion files. The bundle will reference exact MNEMOS paths for engram, Qdrant, TurboQuant, and embedding components; cite the round-two memo carefully; and propose a lightweight Engram build sequence without dragging MNEMOS governance ceremony into the default coding-memory product.

**Tech Stack:** PowerShell, Markdown, local filesystem under `%TEMP%`, MNEMOS repository source paths, round-one/round-two memo artifacts.

## Global Constraints

- the final handoff bundle must live outside the repository under `%TEMP%`
- mention MNEMOS purpose and repository path
- mention the round-two memo as one decision input
- include clickable pointers to relevant MNEMOS files and artifacts
- identify the MNEMOS features Engram should inspect first: engram-related architecture, Qdrant integration, TurboQuant-related components, embedding engine components
- include a proposed Engram build sequence
- do not duplicate entire source files or large docs inline
- do not overstate the A/B findings
- do not commit the `%TEMP%` handoff bundle into the MNEMOS repository

---

### Task 1: Gather Source Anchors and Create the Temp Bundle Directory

**Files:**
- Create: `%TEMP%\\engram_ai_dev_package\\`
- Create: `%TEMP%\\engram_ai_dev_package\\artifact-manifest.md`
- Inspect: `mnemos/engram/model.py`
- Inspect: `mnemos/config.py`
- Inspect: `mnemos/retrieval/qdrant_tier.py`
- Inspect: `mnemos/compression/turbo_quant.py`
- Inspect: `service/app.py`
- Inspect: `C:\Users\vin\AppData\Local\Temp\mnemos-local-project-memory-ab-round2-results-2026-07-16.md`
- Inspect: `C:\Users\vin\.codex\attachments\67074bd0-730d-47c4-8fef-2e9b7559a4e6\pasted-text.txt`

**Interfaces:**
- Consumes: MNEMOS repository and the attached product-distinction text
- Produces: temp bundle directory and a concrete manifest of the exact source artifacts the package will cite

- [ ] **Step 1: Create the Engram temp bundle directory**

Run:

```powershell
$bundle = Join-Path $env:TEMP 'engram_ai_dev_package'
if (Test-Path $bundle) { Remove-Item -LiteralPath $bundle -Recurse -Force }
New-Item -ItemType Directory -Path $bundle | Out-Null
```

Expected:

```text
The directory %TEMP%\engram_ai_dev_package exists and is empty before bundle files are written.
```

- [ ] **Step 2: Inspect the attached product-distinction text and round-two memo**

Run:

```powershell
Get-Content 'C:\Users\vin\.codex\attachments\67074bd0-730d-47c4-8fef-2e9b7559a4e6\pasted-text.txt'
Get-Content 'C:\Users\vin\AppData\Local\Temp\mnemos-local-project-memory-ab-round2-results-2026-07-16.md'
```

Expected:

```text
The package has the strategic split language and the round-two evidence language needed for the overview file.
```

- [ ] **Step 3: Inspect the exact MNEMOS architecture anchor files**

Run:

```powershell
Get-Content mnemos\engram\model.py
Get-Content mnemos\config.py
Get-Content mnemos\retrieval\qdrant_tier.py
Get-Content mnemos\compression\turbo_quant.py
Get-Content service\app.py
```

Expected:

```text
The package has direct source anchors for Engram model structure, retrieval configuration, Qdrant wiring, TurboQuant, and runtime integration.
```

- [ ] **Step 4: Write the artifact manifest with exact paths the package will reference**

Manifest content:

```markdown
# Engram Package Artifact Manifest

- Product distinction input:
  - `C:\Users\vin\.codex\attachments\67074bd0-730d-47c4-8fef-2e9b7559a4e6\pasted-text.txt`
- Round-two memo:
  - `C:\Users\vin\AppData\Local\Temp\mnemos-local-project-memory-ab-round2-results-2026-07-16.md`
- MNEMOS repository root:
  - `G:\MNEMOS`
- Engram model:
  - `G:\MNEMOS\mnemos\engram\model.py`
- Qdrant configuration:
  - `G:\MNEMOS\mnemos\config.py`
  - `G:\MNEMOS\mnemos\retrieval\qdrant_tier.py`
- TurboQuant:
  - `G:\MNEMOS\mnemos\compression\turbo_quant.py`
- Embedding and runtime wiring:
  - `G:\MNEMOS\service\app.py`
  - `G:\MNEMOS\mnemos\retrieval\fusion.py`
  - `G:\MNEMOS\mnemos\retrieval\pgvector_tier.py`
```

Expected:

```text
The manifest gives the package author a fixed evidence base and gives reviewers a quick audit pointer list.
```

### Task 2: Write the Overview and Feature Map Files

**Files:**
- Create: `%TEMP%\\engram_ai_dev_package\\00-overview.md`
- Create: `%TEMP%\\engram_ai_dev_package\\01-mnemos-feature-map.md`
- Test: manual review for product-split clarity and exact file pointers

**Interfaces:**
- Consumes: artifact manifest and inspected source anchors
- Produces: the strategic overview and the reusable-MNEMOS feature map

- [ ] **Step 1: Write `00-overview.md`**

Required sections:

```markdown
# Engram Overview

## What MNEMOS Is
## Why Engram Is Separate
## What Round Two Tells Us
## What The Next AI Dev Should Do First
```

Key points to include verbatim or near-verbatim:

```text
MNEMOS is increasingly becoming evidence memory for high-trust AI workflows.
Engram is intended to be developer working memory for coding agents.
Round two showed packet-backed evidence can help reasoning, but did not show a
time or token win for the tested narrow coding task.
That supports building a separate coding-memory application rather than forcing
MNEMOS's evidence-packet workflow into everyday coding tasks.
```

Expected:

```text
The overview file gives a fresh AI Dev the product rationale without requiring the full prior conversation.
```

- [ ] **Step 2: Write `01-mnemos-feature-map.md`**

Required top-level structure:

```markdown
# MNEMOS Feature Map For Engram

## Borrow Directly
## Adapt For Engram
## Do Not Carry Over By Default
```

Required exact path references:

```markdown
- Engram data model: [model.py](G:/MNEMOS/mnemos/engram/model.py)
- Runtime wiring: [app.py](G:/MNEMOS/service/app.py)
- Qdrant config: [config.py](G:/MNEMOS/mnemos/config.py)
- Qdrant retrieval tier: [qdrant_tier.py](G:/MNEMOS/mnemos/retrieval/qdrant_tier.py)
- TurboQuant: [turbo_quant.py](G:/MNEMOS/mnemos/compression/turbo_quant.py)
- Fusion/embedding bridge: [fusion.py](G:/MNEMOS/mnemos/retrieval/fusion.py)
- Additional embedding backend example: [pgvector_tier.py](G:/MNEMOS/mnemos/retrieval/pgvector_tier.py)
```

Expected:

```text
The feature map clearly separates architectural reuse from MNEMOS-specific governance weight.
```

### Task 3: Write the Build Sequence and Risks Files

**Files:**
- Create: `%TEMP%\\engram_ai_dev_package\\02-engram-build-sequence.md`
- Create: `%TEMP%\\engram_ai_dev_package\\03-risks-and-boundaries.md`
- Test: manual review for sequence clarity and anti-scope-creep guidance

**Interfaces:**
- Consumes: overview decisions and feature-map buckets
- Produces: Engram build order and explicit “do not inherit by default” guidance

- [ ] **Step 1: Write `02-engram-build-sequence.md`**

Required sequence:

```markdown
1. Local-first project scanner
2. Code/doc/test engram extraction
3. Hybrid lexical plus embedding retrieval
4. Compact task-scoped context bundle builder
5. Lightweight MCP or IDE sidecar
6. Optional export bridge back into MNEMOS review mode
```

Required framing:

```text
Engram optimizes for coding-agent context quality, local project context, and
prompt compression, not evidence-packet ceremony.
```

Expected:

```text
The new AI Dev can move directly from strategic context to a concrete R0 build order.
```

- [ ] **Step 2: Write `03-risks-and-boundaries.md`**

Required content:

```markdown
# Risks and Boundaries

- Do not rebuild MNEMOS under a new name.
- Do not carry full evidence packet ceremony into the default coding workflow.
- Keep governance/export as optional later-mode behavior.
- Preserve the useful architectural parts: engram model, retrieval tiering,
  compression, and embedding interfaces.
```

Expected:

```text
The file warns the next AI Dev away from the most likely product-shape failure.
```

### Task 4: Write the Index File and Verify the Bundle

**Files:**
- Create: `%TEMP%\\engram_ai_dev_package\\04-artifact-index.md`
- Verify: `%TEMP%\\engram_ai_dev_package\\00-overview.md`
- Verify: `%TEMP%\\engram_ai_dev_package\\01-mnemos-feature-map.md`
- Verify: `%TEMP%\\engram_ai_dev_package\\02-engram-build-sequence.md`
- Verify: `%TEMP%\\engram_ai_dev_package\\03-risks-and-boundaries.md`

**Interfaces:**
- Consumes: all prior bundle files
- Produces: final readable package with a pointer index

- [ ] **Step 1: Write `04-artifact-index.md`**

Required structure:

```markdown
# Engram AI Dev Package Index

- Overview
- MNEMOS feature map
- Engram build sequence
- Risks and boundaries
- Source artifact manifest
- Round-two memo
- Product-distinction input
```

Required exact pointers:

```markdown
- [00-overview.md](<%TEMP% path>)
- [01-mnemos-feature-map.md](<%TEMP% path>)
- [02-engram-build-sequence.md](<%TEMP% path>)
- [03-risks-and-boundaries.md](<%TEMP% path>)
- [artifact-manifest.md](<%TEMP% path>)
- [Round-two memo](C:/Users/vin/AppData/Local/Temp/mnemos-local-project-memory-ab-round2-results-2026-07-16.md)
- [Product distinction input](C:/Users/vin/.codex/attachments/67074bd0-730d-47c4-8fef-2e9b7559a4e6/pasted-text.txt)
```

Expected:

```text
The next AI Dev can open one file and navigate the full handoff package immediately.
```

- [ ] **Step 2: Verify the package for completeness and scope**

Run:

```powershell
Get-ChildItem $env:TEMP\engram_ai_dev_package
Get-Content $env:TEMP\engram_ai_dev_package\00-overview.md
Get-Content $env:TEMP\engram_ai_dev_package\01-mnemos-feature-map.md
Get-Content $env:TEMP\engram_ai_dev_package\02-engram-build-sequence.md
Get-Content $env:TEMP\engram_ai_dev_package\03-risks-and-boundaries.md
Get-Content $env:TEMP\engram_ai_dev_package\04-artifact-index.md
```

Expected:

```text
The bundle exists, is readable, cites the round-two memo carefully, and gives a fresh AI Dev enough context to start Engram work without pulling in the full MNEMOS evidence-product shape.
```
