# Turbovec Installation & Operator Runbook

## Overview
Turbovec is the SIMD-accelerated, quantized Rust vector index utilized by the **Portable Memory Appliance** profile. Because it leverages unstable AVX-512 intrinsics, it carries specific build prerequisites, most notably the requirement for the Nightly Rust toolchain when installing from source.

## Profile Boundaries
- **Opt-In Only**: This profile must be explicitly enabled.
- **Qdrant Default**: Qdrant remains the recommended backend for frictionless cross-platform deployments.
- **Linux Friction**: Linux deployments currently lack a `manylinux` pre-built wheel, demanding manual compilation.

---

## Windows Installation (Supported Experimental Path)

Windows deployments are well-supported when installing locally.

1. **Install Prerequisites**:
   - Python 3.12+ (Anaconda recommended)
   - Rust toolchain (Nightly required)
2. **Setup Rust**:
   ```powershell
   rustup default nightly
   ```
3. **Compile Turbovec**:
   ```powershell
   cd turbovec/turbovec-python
   python -m pip install --upgrade pip maturin
   maturin develop --release
   ```
4. **Validation**:
   ```powershell
   python -c "import turbovec; print('Turbovec installed successfully!')"
   python -m pytest tests/test_turbovec_real_adapter_contract.py -q
   ```

---

## Linux / WSL2 Installation (Blocked / Warn)

Linux/WSL installations strictly require the Nightly Rust build path. **Do not attempt to use Stable Rust.**

1. **Install Prerequisites**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```
2. **Setup Rust**:
   ```bash
   rustup default nightly
   ```
3. **Compile Turbovec via Maturin**:
   ```bash
   cd turbovec/turbovec-python
   python3 -m pip install --upgrade pip maturin
   maturin build --release
   ```
4. **Install the Wheel**:
   ```bash
   python3 -m pip install target/wheels/*.whl
   ```
5. **Validation**:
   ```bash
   python3 -c "import turbovec; print('Turbovec installed successfully!')"
   python3 -m pytest tests/test_turbovec_real_adapter_contract.py -q
   ```

---

## Fallback Behavior
If Turbovec installation fails or if a Nightly Rust toolchain is prohibited by your environment's governance standards, you **must** fall back to the default `core_memory_appliance` profile utilizing Qdrant.

To reset to defaults, ensure the following environment variables are unset or configured correctly:
```env
MNEMOS_PROFILE=core_memory_appliance
MNEMOS_RETRIEVAL_BACKEND=qdrant
MNEMOS_TURBOVEC_ENABLED=false
```
