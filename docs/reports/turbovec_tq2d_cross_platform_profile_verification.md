# TQ-2D Cross-Platform Profile Verification

## Decision Gate
**TURBOVEC_CROSS_PLATFORM_WARN**

*Status:* Windows Native passes flawlessly. Linux/WSL integration requires manual compilation via Nightly Rust due to the absence of precompiled wheels.

---

## 1. Windows Native Platform
- **Platform**: Windows Native
- **OS Version**: Windows 11 (Host)
- **Python Version**: Python 3.12+ (Anaconda)
- **Rust Toolchain**: Nightly (Required for `core::arch::x86_64` AVX-512 intrinsics)
- **Turbovec Install Method**: Source via `maturin develop --release`
- **Wheel Filename**: None / Source Build
- **Import Status**: `[PASS] import turbovec`
- **Test Result**: `[PASS]` 50/50 Quality queries completed.
- **Backup/Restore Result**: `[PASS]` Clean load, integrity check passed, backup archive created.
- **Cross-Platform Restore Result**: `[PENDING]` (Awaiting Linux test)
- **Known Blockers**: None for Windows.

## 2. Linux / WSL2 Platform
- **Platform**: Linux (WSL2)
- **OS Version**: Ubuntu (via WSL2)
- **Python Version**: Python 3.12.3
- **Rust Toolchain**: Missing / Requires Manual Setup
- **Turbovec Install Method**: Failed (No wheels, no local Rust toolchain)
- **Wheel Filename**: N/A
- **Import Status**: `[FAIL] ModuleNotFoundError`
- **Test Result**: `[BLOCKED]`
- **Backup/Restore Result**: `[BLOCKED]`
- **Cross-Platform Restore Result**: `[BLOCKED]`
- **Known Blockers**: 
  - `turbovec` does not yet publish `manylinux` or `musllinux` precompiled wheels to PyPI or a local repository.
  - The WSL2 environment lacks `rustc` by default.
  - Due to the hardware-specific AVX-512 instruction requirements in the Turbovec Rust code, the package strictly requires the `nightly` Rust channel. This is a severe deployment blocker for standard Linux Docker environments.

---

## Critical Operator Notes
**AVX-512 & Nightly Rust Constraint**
The earlier Windows build requirement for `nightly` Rust remains an active constraint across **all** operating systems. Because Turbovec directly utilizes unstable AVX-512 SIMD intrinsics, standard `stable` Rust toolchains will refuse to compile the package. 

Until Turbovec is refactored to use stable fallback SIMD paths or pre-built `.whl` binaries are provided via CI/CD, operators attempting to deploy the Portable Memory Appliance on Linux **MUST** execute:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
pip install maturin
maturin build --release
```

## Summary & Next Steps
**Cross-Platform Profile Verification** is technically sound on the code level—the profile JSON manifests and SQLite sidecars are entirely OS-agnostic—but strictly constrained at the infrastructure level due to the missing Linux wheel pipeline. 

Proceed to **TQ-2E (Operator-Ready Experimental Profile Gate)**, but the `portable_memory_appliance` documentation MUST explicitly declare Qdrant as the only frictionless deployment path for Linux until a `turbovec` manylinux wheel is finalized.
