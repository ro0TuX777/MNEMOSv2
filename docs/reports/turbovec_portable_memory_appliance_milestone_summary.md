# Portable Memory Appliance Milestone Summary
**Turbovec + SQLite Integration Closeout**

## 1. Executive Summary
The **Portable Memory Appliance** (Turbovec + SQLite) has been successfully implemented, validated, and packaged as an official, opt-in experimental retrieval profile for MNEMOS. It delivers equivalent semantic retrieval, full source traceability, and rapid native Reciprocal Rank Fusion (RRF) hybrid search without the external footprint of a containerized Qdrant instance. However, due to its strict Nightly Rust/AVX-512 compilation requirements, its promotion to a default state is blocked until CI/CD pipelines can reliably emit `manylinux` wheel distributions.

## 2. Final Decision States
* **TQ-1**: `TURBOVEC_PORTABLE_PROFILE_CANDIDATE`
  * Turbovec successfully achieved performance parity with Qdrant on the canonical subset.
* **TQ-2**: `TURBOVEC_OPERATOR_EXPERIMENTAL_READY_WITH_PLATFORM_WARNINGS`
  * Fully passed the 10K full-corpus benchmarks, backup/restore validations, and Windows-native integrations. Blocked from default adoption due to Linux compilation blockers.
* **TQ-3**: `TURBOVEC_PACKAGING_READY_PENDING_CI_ARTIFACTS`
  * All CI/CD scripts, build matrices, and tests are implemented. Awaiting the first minted CI wheel artifacts to officially close the packaging gate.

## 3. What is Production-Safe Today
* **Core Memory Appliance (Qdrant)**: Remains the immutable default, production-safe deployment path.
* **Governance Native (pgvector)**: Remains the standard governance-oriented profile.
* **Backup/Restore Logic**: The SHA-256 and `PRAGMA integrity_check` tooling for the Portable Memory Appliance is robust, fail-safe, and atomic.

## 4. What Remains Experimental
* **The Portable Memory Appliance Profile**: While architecturally sound, it is officially classified as an opt-in experimental profile exclusively targeting validated local/offline hosts where operators accept the manual build constraints.

## 5. How to Enable the Profile
To activate the Portable Memory Appliance, operators must explicitly opt-in via environment variables:
```env
MNEMOS_PROFILE=portable_memory_appliance
MNEMOS_RETRIEVAL_BACKEND=turbovec
MNEMOS_TURBOVEC_ENABLED=true
```

## 6. Benchmark Evidence Table (TQ-2C Full Corpus)
| Metric | Result |
|---|---|
| **Corpus Scale** | 140 PDFs (ToLearn + SIGINT) |
| **Total Chunks** | 10,718 |
| **Ingestion Time** | 35.72s |
| **Dense p50 Latency** | 9.62 ms |
| **FTS p50 Latency** | 1.48 ms |
| **Hybrid p50 Latency** | 17.48 ms |
| **Traceability Score** | 1.000 (100% preservation) |

## 7. Backup/Restore Evidence
| Operation | Result |
|---|---|
| **Index Footprint** | 4.05 MB (`index.tvim`) |
| **Metadata Footprint** | 60.46 MB (`metadata.sqlite`) |
| **Backup Archive Size** | 20.18 MB (Zipped) |
| **Backup Duration** | 2.00s |
| **Restore Duration** | 0.94s |
| **Post-Restore Parity** | PASS (Identical retrieval quality) |

## 8. Cross-Platform Warnings
* **Windows Native**: Supported. Installable via source using Nightly Rust.
* **Linux / WSL2**: Blocked (unless manually compiled). Standard glibc containers lack Nightly Rust.
* **ARM64 / Apple Silicon**: Hard blocker. The codebase explicitly invokes unstable `core::arch::x86_64` intrinsics and will not compile on ARM.

## 9. Packaging Status
* **Implemented**: The `maturin` matrix is written, tested natively, and codified in `.github/workflows/turbovec_wheel_build.yml`.
* **Pending**: We are awaiting GitHub Actions to officially produce the `win_amd64` and `manylinux2014` wheel artifacts.

## 10. Remaining Blockers Before Broader Promotion
Broad promotion of the Portable Memory Appliance is strictly blocked until:
1. **CI Artifact Verification**: Minted CI wheel artifacts must pass pip-install smoke tests.
2. **Installer Integration**: The MNEMOS installer must be able to pull and resolve the `.whl` artifacts gracefully across Windows and Linux.

**Final Boundary**: Qdrant remains the default Core Memory Appliance. Turbovec + SQLite is an opt-in profile. Broad promotion is blocked until CI wheel artifacts are minted and verified.
