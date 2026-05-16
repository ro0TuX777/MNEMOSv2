# Turbovec Wheel Build Matrix Runbook

This runbook outlines the required platform targets, Rust toolchains, and build steps to manually or programmatically compile distributable `.whl` files for the `turbovec` Python extension.

## 1. Supported Matrix
| Target OS | Target Arch | Python Target | Maturin Platform Tag | Rust Toolchain | SIMD Requirement |
|---|---|---|---|---|---|
| Windows | x86_64 | 3.11, 3.12 | `win_amd64` | `nightly` | AVX-512 |
| Linux | x86_64 | 3.11, 3.12 | `manylinux2014` | `nightly` | AVX-512 |

> **Note**: Due to strict AVX-512 intrinsic usage, arm64 (e.g., Apple Silicon / AWS Graviton) is NOT supported and will immediately fail compilation.

## 2. Nightly Rust Pinning
To ensure that standard library layout updates do not break Turbovec's unstable imports, we mandate pinning the Nightly toolchain.
* **Command**: `rustup default nightly`

## 3. Building on Windows
Windows requires Visual Studio C++ Build Tools installed along with Rust.
```powershell
# 1. Install rust nightly
rustup default nightly

# 2. Prepare Python build env
python -m pip install --upgrade pip maturin

# 3. Build Wheels (from turbovec project root)
cd turbovec/turbovec-python
maturin build --release --out target/wheels
```

## 4. Building on Linux (manylinux)
To ensure broad glibc compatibility across different Linux distributions, wheels must be built inside the official `manylinux` docker containers. However, since we require `nightly` rust, we cannot use standard pre-packaged maturin containers seamlessly without installing nightly.

```bash
# Example invocation via docker
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release -m /io/turbovec/turbovec-python/Cargo.toml --out /io/turbovec/turbovec-python/target/wheels --manylinux 2014
```
*(Ensure the container has nightly rust equipped!)*

## 5. Artifact Validation
Once built, operators must find the `.whl` output inside `target/wheels/`. 
To smoke test:
```bash
python -m pip install target/wheels/turbovec-<version>-cp312-cp312-win_amd64.whl
python -m pytest tests/test_turbovec_wheel_install_smoke.py
```
