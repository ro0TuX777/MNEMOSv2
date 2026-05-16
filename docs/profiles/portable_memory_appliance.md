# Portable Memory Appliance Profile

## Overview
The **Portable Memory Appliance** is a lightweight, local-first retrieval profile for the MNEMOS service. It replaces the containerized Qdrant requirement with an embedded, SIMD-accelerated vector index (`Turbovec`) alongside a high-performance metadata sidecar (`SQLite FTS5`). 

This profile provides equivalent semantic capabilities, source traceability, and hybrid search (via Python-native Reciprocal Rank Fusion) without the overhead of external database dependencies or Docker.

## Profile Architecture
- **Dense Vector Store**: Turbovec (Rust-native, SIMD-accelerated, quantized index)
- **Lexical/Metadata Store**: SQLite3 with FTS5 virtual tables
- **Hybrid Fusion**: Local Reciprocal Rank Fusion (RRF)
- **Persistence Unit**: `index.tvim`, `metadata.sqlite`, and `manifest.json` stored locally.
- **Dependency**: Requires the `turbovec` Python extension.

## Environment Variables
To activate this profile, configure your environment or `.env` file as follows:

```env
# Activate the portable profile (Default is 'core_memory_appliance')
MNEMOS_PROFILE=portable_memory_appliance

# Enable Turbovec tier explicitly
MNEMOS_TURBOVEC_ENABLED=true

# Force the backend router to use turbovec
MNEMOS_RETRIEVAL_BACKEND=turbovec
```

## Profile State & Capabilities
- **Status**: `operator-ready experimental`
- **Decision Bound**: `TURBOVEC_OPERATOR_EXPERIMENTAL_READY_WITH_PLATFORM_WARNINGS`
- **Default Eligible**: `no` (Qdrant remains default)
- **Shadow Eligible**: `yes`

## Support Matrix
| Platform | Support Status | Notes |
|---|---|---|
| **Windows Native** | Supported Experimental Path | Seamless installation via source or local build. |
| **Linux/WSL2** | **Blocked / Warn** | Blocked unless manually compiled using Nightly Rust + `maturin`. |
| **Standard Linux / Docker** | Not Yet Supported | Blocked due to missing `manylinux` wheel distributions. |
| **Qdrant Core Memory Appliance** | **Recommended Default** | Frictionless default path for broad deployments. |

## Core Distinctions
Unlike the `Core Memory Appliance` (Qdrant-backed) or the `Governance Native` profile (pgvector-backed), the Portable Memory Appliance is strictly designed for local, single-node execution. It shines in air-gapped environments or low-footprint operator nodes where spinning up Docker containers is not feasible.
