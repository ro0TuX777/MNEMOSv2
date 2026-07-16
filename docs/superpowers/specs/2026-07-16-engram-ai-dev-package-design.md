# Engram AI Dev Package Design

Date: 2026-07-16
Status: Proposed

## Goal

Create a focused artifact package for a fresh AI Dev session that explains why
the new application `Engram` should be built as a distinct coding-memory
product, points the next agent at the relevant reusable MNEMOS architecture,
and provides a practical build sequence for Engram.

The package is a handoff bundle, not a production feature. It should help a new
agent quickly understand:

- what MNEMOS is today;
- why Engram is being split out;
- which MNEMOS components are candidates for reuse;
- what not to carry over from MNEMOS by default; and
- how to begin building Engram in a sensible order.

## Context

The attached product discussion establishes the strategic distinction:

- `MNEMOS = evidence memory for high-trust AI workflows`
- `Engram = developer working memory for coding agents`

Recent A/B work in this repository reinforced that distinction. The round-two
memo showed that packet-backed evidence can guide reasoning, but did not show a
time or token win for a narrow coding bug when using the current MNEMOS
evidence-packet workflow. That result does not invalidate MNEMOS; instead, it
supports building a separate coding-memory application optimized for developer
and IDE-agent workflows.

## Package Scope

The package should:

- live outside the repository under `%TEMP%`;
- be readable by a fresh AI Dev without prior session context;
- mention MNEMOS purpose and repository path;
- mention the round-two memo as one decision input;
- point to relevant MNEMOS source files and artifacts with exact paths;
- identify the specific MNEMOS features Engram should inspect first:
  - engram-related architecture
  - Qdrant integration
  - TurboQuant-related components
  - embedding engine components
- include a proposed Engram build sequence.

The package should not:

- try to fully spec Engram implementation details;
- duplicate entire source files or large docs inline;
- overstate the A/B findings;
- commit the handoff bundle into the MNEMOS repository.

## Deliverable Shape

The recommended deliverable is a small bundle under `%TEMP%`, not one giant
document. The bundle should contain:

1. `00-overview.md`
2. `01-mnemos-feature-map.md`
3. `02-engram-build-sequence.md`
4. `03-risks-and-boundaries.md`
5. `04-artifact-index.md`

This structure keeps the handoff readable and lets a future AI Dev jump
directly to the most relevant slice of context.

## Content Model

### 00 Overview

This file should explain:

- what MNEMOS currently is;
- why Engram is being created separately;
- the product split between evidence memory and coding memory;
- the practical reason for the split, including the round-two outcome:
  packet-backed evidence was useful, but the workflow did not show a token or
  proven time advantage for the tested coding task.

### 01 MNEMOS Feature Map

This file should list the relevant MNEMOS components and group them into:

- `borrow directly`
- `adapt for Engram`
- `do not carry over by default`

It should point the next AI Dev to exact repository files or directories for:

- engram-related logic and data structures;
- Qdrant integration and collection configuration;
- TurboQuant-related implementation points;
- embedding engine configuration and execution paths;
- retrieval and sidecar patterns that may be useful for Engram.

The purpose is not to decide the final Engram architecture, but to reduce
discovery cost for the next agent.

### 02 Engram Build Sequence

This file should propose a lightweight Engram implementation order, starting
with a small R0 and growing only as needed. The suggested sequence should be:

1. local-first project scanner;
2. code/doc/test engram extraction;
3. hybrid lexical plus embedding retrieval;
4. compact task-scoped context bundle builder;
5. lightweight MCP or IDE sidecar;
6. optional export bridge back into MNEMOS-style review mode later.

It should emphasize that Engram optimizes for coding-agent context quality and
prompt compression, not evidence-packet ceremony.

### 03 Risks and Boundaries

This file should clarify what Engram should not inherit from MNEMOS by default,
including:

- audit-heavy response framing;
- full evidence packet ceremony;
- approval boilerplate in normal coding workflows;
- large inventories, hashes, and exclusion receipts except in optional review
  mode.

It should also note the main design risk: accidentally rebuilding MNEMOS under a
new name instead of producing a genuinely lighter coding-memory system.

### 04 Artifact Index

This file should be a compact pointer list to the most relevant inputs:

- the attached product-distinction text;
- the round-two memo;
- the MNEMOS repository root;
- the exact code locations referenced in the feature map;
- any especially important design or experiment artifacts.

## Output Quality Requirements

The bundle should be:

- concise;
- easy to scan;
- written for a capable fresh AI Dev;
- honest about evidence strength;
- opinionated enough to create momentum without pretending every design choice
  is already settled.

The round-two memo should be referenced carefully: it supports the decision to
split Engram from MNEMOS, but it does not prove that all forms of coding memory
are inefficient. It only shows that the tested MNEMOS evidence-packet shape did
not demonstrate a token or time win on that narrow debugging task.

## Acceptance Criteria

The package is successful if a fresh AI Dev can, on first read:

- explain why Engram is distinct from MNEMOS;
- locate the MNEMOS components most relevant to Engram;
- distinguish what to copy versus what to leave behind;
- describe an initial Engram build sequence; and
- begin planning or implementation without rediscovering the strategic
  rationale from scratch.

## Artifact Location

The final handoff bundle should be saved outside the repository under `%TEMP%`,
for example:

`%TEMP%\engram_ai_dev_package\`

The design spec itself should remain in the repository as the durable record of
why the package was assembled and what it should contain.
