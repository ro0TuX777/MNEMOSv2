# TQ-2E Operator-Ready Experimental Profile Gate

## Final Technical Decision
**`TURBOVEC_OPERATOR_EXPERIMENTAL_READY_WITH_PLATFORM_WARNINGS`**

### Rationale
The Portable Memory Appliance has successfully passed rigorous functional evaluations:
* **TQ-1**: Established as a `TURBOVEC_PORTABLE_PROFILE_CANDIDATE` against the canonical 154-chunk subset.
* **TQ-2A**: Runtime profile routing logic was implemented and proven reliable (Fail-closed fallback to Qdrant).
* **TQ-2B**: Atomic Backup and Restore tooling implemented with strict SHA-256 and `PRAGMA integrity_check` validation.
* **TQ-2C**: Cleared the **Full-Corpus Benchmark** (140 PDFs, 10,718 chunks) successfully, validating ingestion speed, footprint stability, and Post-Restore Hybrid retrieval parity.
* **TQ-2D**: Executed Cross-Platform Verification. Flagged as **`TURBOVEC_CROSS_PLATFORM_WARN`**. Windows Native operates flawlessly, but Linux/WSL deployment is severely blocked due to missing `manylinux` or `musllinux` wheels, demanding a mandatory Nightly Rust toolchain compilation step.

Therefore, the `turbovec` profile is suitable for explicit operator-selected experimental use on validated hosts (e.g. Windows Native or Linux hosts properly equipped with Rust), but **is not suitable as a default or frictionless cross-platform deployment profile.**

### Operational Boundary
* **Default Profile**: Qdrant remains the immutable default Core Memory Appliance.
* **Role**: Turbovec acts exclusively as an opt-in Portable Memory Appliance profile.
* **Target Audience**: Local-first/offline deployments where the operator explicitly accepts and resolves the manual build constraints on Linux.
* **Promotion Status**: Blocked from broad production promotion until the Linux wheel/build friction is permanently resolved via CI pipelines (Phase TQ-3).
