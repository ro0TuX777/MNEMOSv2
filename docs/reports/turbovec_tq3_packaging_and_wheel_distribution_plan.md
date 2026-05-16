# TQ-3 Packaging and Wheel Distribution Plan

## Objective
Resolve the deployment friction for the **Portable Memory Appliance** (Turbovec + SQLite) by establishing a reproducible wheel packaging and distribution workflow. Because Turbovec relies on unstable AVX-512 intrinsics, compilation requires a Nightly Rust toolchain. Pre-building and distributing `.whl` files via CI/CD removes this heavy compilation burden from the operator environment, effectively lowering the deployment barrier for Linux/WSL.

## Current State & Blockers
- **Windows**: Works via manual `maturin develop` but forces operators to install Nightly Rust.
- **Linux/WSL**: Completely blocked in standard Python deployments because no `manylinux` wheels are available.

## Delivery Scope
1. **Build Scripts**: Establish `scripts/build_turbovec_wheel_windows.ps1` and `scripts/build_turbovec_wheel_linux.sh` to codify exactly how wheels are compiled natively or inside containerized workflows.
2. **CI/CD Pipeline**: Architect `.github/workflows/turbovec_wheel_build.yml` to automatically trigger maturin matrix builds across `ubuntu-latest` and `windows-latest` runners, pushing artifacts directly to releases.
3. **Smoke Testing**: Establish a lightweight smoke test (`tests/test_turbovec_wheel_install_smoke.py`) that strictly runs the adapter contract tests to verify that built wheels function correctly when pip-installed.
4. **Documentation**: Distribute a runbook detailing how operators can fetch these wheels or build them manually if operating inside isolated air-gapped domains.

## Target Profiles & Matrices
* **Python**: 3.11, 3.12
* **Architectures**: x86_64 (AVX-512 target specific)
* **OS**: `win_amd64`, `manylinux2014_x86_64`
* **Rust**: `nightly` (Pinned to ensure AVX-512 features do not arbitrarily break).

## Decision Gate
* **`TURBOVEC_PACKAGING_PASS`**: Wheels successfully build across all matrix targets, install cleanly, and pass the real adapter contract.
* **`TURBOVEC_PACKAGING_WARN`**: Wheels build but require extremely specific environment variables, or one OS fails while another passes.
* **`TURBOVEC_PACKAGING_FAIL`**: Wheel compilation natively fails across target OS environments or installed wheels fail the adapter test.
