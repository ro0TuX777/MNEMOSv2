#!/usr/bin/env python3
"""
MNEMOS Portable Memory Appliance — Offline Installation Script
===============================================================

Run this script on an AIR-GAPPED or OFFLINE target machine to install
the Portable Memory Appliance from a pre-built offline bundle.

This script expects to be run from inside the bundle directory created
by `prepare_offline_bundle.py`.

Prerequisites on the target:
  - Python 3.11+
  - Rust Nightly toolchain (for compiling Turbovec from source)

Usage:
    python install_offline_bundle.py
    python install_offline_bundle.py --skip-turbovec   # Skip Rust compilation
    python install_offline_bundle.py --venv ./my_venv   # Custom venv path
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import platform


def log(msg: str):
    print(f"[install] {msg}")


def error(msg: str):
    print(f"[install] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check_python_version():
    """Ensure Python 3.11+."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        error(f"Python 3.11+ required. Found: {major}.{minor}")
    log(f"Python version: {major}.{minor} ✓")


def check_rust_nightly():
    """Check if Rust nightly toolchain is available."""
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip()
        if "nightly" in version.lower():
            log(f"Rust toolchain: {version} ✓")
            return True
        else:
            log(f"Rust toolchain: {version} (WARNING: nightly required for Turbovec)")
            return False
    except FileNotFoundError:
        log("Rust toolchain: NOT FOUND")
        return False
    except Exception as e:
        log(f"Rust toolchain check failed: {e}")
        return False


def load_manifest(bundle_dir: str) -> dict:
    """Load and validate the bundle manifest."""
    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    if not os.path.exists(manifest_path):
        error(f"Bundle manifest not found at {manifest_path}. Is this a valid bundle?")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    log(f"Bundle version : {manifest.get('bundle_version', 'unknown')}")
    log(f"Created at     : {manifest.get('created_at', 'unknown')}")
    log(f"Target platform: {manifest.get('target_platform', 'unknown')}")
    return manifest


def create_venv(venv_path: str):
    """Create a Python virtual environment."""
    if os.path.exists(venv_path):
        log(f"Virtual environment already exists at {venv_path}")
        return

    log(f"Creating virtual environment at {venv_path}")
    subprocess.run(
        [sys.executable, "-m", "venv", venv_path],
        check=True,
    )
    log("  OK: Virtual environment created.")


def get_venv_python(venv_path: str) -> str:
    """Get the path to the Python executable inside the venv."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    return os.path.join(venv_path, "bin", "python3")


def install_wheels(venv_python: str, bundle_dir: str):
    """Install all pre-downloaded wheels from the bundle."""
    wheels_dir = os.path.join(bundle_dir, "wheels")
    if not os.path.exists(wheels_dir):
        error(f"Wheels directory not found: {wheels_dir}")

    wheel_files = [
        os.path.join(wheels_dir, f)
        for f in os.listdir(wheels_dir)
        if f.endswith((".whl", ".tar.gz", ".zip"))
    ]

    if not wheel_files:
        error(f"No wheel/package files found in {wheels_dir}")

    log(f"Installing {len(wheel_files)} packages from offline bundle...")

    cmd = [
        venv_python, "-m", "pip", "install",
        "--no-index",
        "--find-links", wheels_dir,
        "--no-cache-dir",
    ]

    # Install all wheels at once using find-links
    for wf in wheel_files:
        basename = os.path.basename(wf)
        # Extract package name from wheel filename
        pkg_name = basename.split("-")[0].replace("_", "-")
        cmd.append(pkg_name)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  Batch install had issues, falling back to individual installs...")
        # Fallback: install each wheel individually
        for wf in wheel_files:
            individual_cmd = [
                venv_python, "-m", "pip", "install",
                "--no-index",
                "--find-links", wheels_dir,
                "--no-cache-dir",
                wf,
            ]
            r = subprocess.run(individual_cmd, capture_output=True, text=True)
            basename = os.path.basename(wf)
            if r.returncode == 0:
                log(f"  OK: {basename}")
            else:
                log(f"  WARN: Failed to install {basename}: {r.stderr.strip()[:120]}")
    else:
        log(f"  OK: All {len(wheel_files)} packages installed.")


def stage_embedding_model(bundle_dir: str):
    """Stage the embedding model so sentence-transformers can find it offline."""
    model_source = os.path.join(bundle_dir, "models", "BAAI_bge-base-en-v1.5")

    if not os.path.exists(model_source):
        log("WARNING: Pre-downloaded embedding model not found in bundle.")
        log("  The Portable Memory Appliance will attempt to download it on first use.")
        return None

    log(f"Embedding model staged at: {model_source}")
    log("  Set MNEMOS_EMBEDDING_MODEL_PATH to this path in your .env file.")
    return model_source


def compile_turbovec(venv_python: str, bundle_dir: str):
    """Compile Turbovec from the bundled source using Rust nightly."""
    turbovec_dir = os.path.join(bundle_dir, "mnemos", "turbovec", "turbovec-python")

    if not os.path.exists(turbovec_dir):
        log("WARNING: Turbovec source not found in bundle. Skipping compilation.")
        log("  The Portable Memory Appliance will use the Mock adapter (CPU fallback).")
        return False

    log("Compiling Turbovec (Rust SIMD vector index)...")
    log("  This may take 2-5 minutes on the first build.")

    # Ensure maturin is available
    subprocess.run(
        [venv_python, "-m", "pip", "install", "maturin"],
        capture_output=True,
    )

    result = subprocess.run(
        [venv_python, "-m", "maturin", "develop", "--release"],
        cwd=turbovec_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"  WARN: Turbovec compilation failed:")
        log(f"  {result.stderr.strip()[:300]}")
        log("  The system will fall back to the mock dense index adapter.")
        return False

    # Verify import
    verify = subprocess.run(
        [venv_python, "-c", "import turbovec; print('Turbovec OK')"],
        capture_output=True, text=True,
    )
    if verify.returncode == 0:
        log("  OK: Turbovec compiled and verified. ✓")
        return True
    else:
        log("  WARN: Turbovec compiled but import verification failed.")
        return False


def install_mnemos_source(venv_python: str, bundle_dir: str):
    """Install the MNEMOS source as an editable package."""
    mnemos_dir = os.path.join(bundle_dir, "mnemos")
    if not os.path.exists(mnemos_dir):
        error(f"MNEMOS source not found in bundle: {mnemos_dir}")

    setup_py = os.path.join(mnemos_dir, "setup.py")
    pyproject = os.path.join(mnemos_dir, "pyproject.toml")

    if os.path.exists(setup_py) or os.path.exists(pyproject):
        log("Installing MNEMOS as editable package...")
        result = subprocess.run(
            [venv_python, "-m", "pip", "install", "-e", mnemos_dir, "--no-deps"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log("  OK: MNEMOS installed.")
        else:
            log(f"  Note: editable install skipped ({result.stderr.strip()[:100]})")
            log("  You can add the mnemos directory to PYTHONPATH instead.")
    else:
        log("No setup.py/pyproject.toml found. Add MNEMOS to PYTHONPATH manually:")
        log(f"  export PYTHONPATH={mnemos_dir}:$PYTHONPATH")


def write_env_file(bundle_dir: str, model_path: str, venv_path: str):
    """Write a .env.mnemos file for the portable profile."""
    env_path = os.path.join(bundle_dir, ".env.mnemos")
    env_lines = [
        "# MNEMOS Portable Memory Appliance — Environment Configuration",
        f"# Generated by offline installer on {platform.node()}",
        "",
        "# Profile selection (required — all three must be set)",
        "MNEMOS_PROFILE=portable_memory_appliance",
        "MNEMOS_RETRIEVAL_BACKEND=turbovec",
        "MNEMOS_TURBOVEC_ENABLED=true",
        "",
        "# Embedding model (offline path)",
    ]

    if model_path:
        env_lines.append(f"MNEMOS_EMBEDDING_MODEL_PATH={model_path}")
    else:
        env_lines.append("# MNEMOS_EMBEDDING_MODEL_PATH=  # Set to your local model path")

    env_lines.extend([
        "",
        "# Storage paths",
        f"MNEMOS_TURBOVEC_STORAGE_PATH={os.path.join(bundle_dir, 'runtime', 'turbovec_storage')}",
        "",
        "# EchoFrame (optional — enable for ~97% LLM context compression)",
        "MNEMOS_ECHOFRAME_DEFAULT_ON_ELIGIBLE=true",
        "MNEMOS_ECHOFRAME_KILL_SWITCH=false",
    ])

    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")

    log(f"  OK: Environment file written to {env_path}")
    return env_path


def run_smoke_test(venv_python: str, bundle_dir: str):
    """Run a minimal smoke test to verify the installation."""
    log("Running installation smoke test...")

    test_script = """
import sys
results = []

# 1. Core imports
try:
    import numpy as np
    results.append(("numpy", "OK"))
except ImportError as e:
    results.append(("numpy", f"FAIL: {e}"))

# 2. Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    results.append(("sentence-transformers", "OK"))
except ImportError as e:
    results.append(("sentence-transformers", f"FAIL: {e}"))

# 3. MNEMOS core
try:
    from mnemos.retrieval.turbovec_config import TurbovecConfig
    from mnemos.retrieval.turbovec_tier import TurbovecTier
    from mnemos.retrieval.turbovec_fusion import TurbovecFusion
    results.append(("mnemos.retrieval", "OK"))
except ImportError as e:
    results.append(("mnemos.retrieval", f"FAIL: {e}"))

# 4. Turbovec native
try:
    import turbovec
    results.append(("turbovec (native)", "OK"))
except ImportError:
    results.append(("turbovec (native)", "WARN: Not compiled (mock adapter will be used)"))

# 5. Backup/restore tools
try:
    from mnemos.tools.turbovec_backup import create_backup
    from mnemos.tools.turbovec_restore import run_restore
    results.append(("backup/restore tools", "OK"))
except ImportError as e:
    results.append(("backup/restore tools", f"FAIL: {e}"))

print()
print("=" * 50)
print("  MNEMOS Portable Memory Appliance Smoke Test")
print("=" * 50)
for name, status in results:
    icon = "✓" if status == "OK" else "⚠" if "WARN" in status else "✗"
    print(f"  {icon} {name}: {status}")
print("=" * 50)

failures = [r for r in results if "FAIL" in r[1]]
if failures:
    print(f"  {len(failures)} FAILURE(s) detected.")
    sys.exit(1)
else:
    print("  All checks passed.")
    sys.exit(0)
"""

    result = subprocess.run(
        [venv_python, "-c", test_script],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.join(bundle_dir, "mnemos")},
    )
    print(result.stdout)
    if result.stderr:
        # Only show stderr if there were actual errors (filter torch warnings)
        important_errors = [
            line for line in result.stderr.splitlines()
            if "error" in line.lower() or "fail" in line.lower()
        ]
        if important_errors:
            for line in important_errors:
                log(f"  {line}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Install MNEMOS Portable Memory Appliance from an offline bundle."
    )
    parser.add_argument(
        "--venv",
        default="mnemos_venv",
        help="Path to create the Python virtual environment (default: ./mnemos_venv)",
    )
    parser.add_argument(
        "--skip-turbovec",
        action="store_true",
        help="Skip Turbovec Rust compilation (uses mock adapter fallback).",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip venv creation, install into the current Python environment.",
    )
    args = parser.parse_args()

    bundle_dir = os.path.dirname(os.path.abspath(__file__))

    log("=" * 60)
    log("MNEMOS Portable Memory Appliance — Offline Installer")
    log("=" * 60)

    # Step 0: Preflight
    check_python_version()
    manifest = load_manifest(bundle_dir)
    has_rust = check_rust_nightly()
    log("")

    # Step 1: Virtual environment
    if args.skip_venv:
        venv_python = sys.executable
        log("Using current Python environment (--skip-venv).")
    else:
        venv_path = os.path.abspath(args.venv)
        create_venv(venv_path)
        venv_python = get_venv_python(venv_path)

    # Step 2: Install wheels
    install_wheels(venv_python, bundle_dir)
    log("")

    # Step 3: Stage embedding model
    model_path = stage_embedding_model(bundle_dir)
    log("")

    # Step 4: Compile Turbovec
    turbovec_ok = False
    if args.skip_turbovec:
        log("Skipping Turbovec compilation (--skip-turbovec).")
        log("  The mock dense index adapter will be used.")
    elif not has_rust:
        log("Rust nightly not found. Skipping Turbovec compilation.")
        log("  Install Rust nightly and re-run, or use --skip-turbovec.")
    else:
        turbovec_ok = compile_turbovec(venv_python, bundle_dir)
    log("")

    # Step 5: Install MNEMOS source
    install_mnemos_source(venv_python, bundle_dir)
    log("")

    # Step 6: Write .env file
    env_path = write_env_file(bundle_dir, model_path, args.venv)
    log("")

    # Step 7: Smoke test
    smoke_pass = run_smoke_test(venv_python, bundle_dir)
    log("")

    # Summary
    log("=" * 60)
    log("INSTALLATION COMPLETE")
    log("=" * 60)
    log(f"  Bundle directory : {bundle_dir}")
    if not args.skip_venv:
        log(f"  Virtual environment: {os.path.abspath(args.venv)}")
        if platform.system() == "Windows":
            log(f"  Activate with    : {os.path.abspath(args.venv)}\\Scripts\\activate")
        else:
            log(f"  Activate with    : source {os.path.abspath(args.venv)}/bin/activate")
    log(f"  Environment file : {env_path}")
    log(f"  Turbovec native  : {'✓ Compiled' if turbovec_ok else '⚠ Mock adapter (fallback)'}")
    log(f"  Embedding model  : {'✓ Staged' if model_path else '⚠ Not staged (will need download)'}")
    log(f"  Smoke test       : {'✓ Passed' if smoke_pass else '⚠ Issues detected'}")
    log("")

    if not turbovec_ok:
        log("NOTE: Without native Turbovec, the system uses a pure-Python mock")
        log("  adapter. For production-grade SIMD performance, install Rust nightly")
        log("  and re-run this script without --skip-turbovec.")

    log("")
    log("To start using MNEMOS Portable Memory Appliance:")
    log(f"  1. Source the environment: set -a; source {env_path}; set +a")
    log("  2. Run your application with the MNEMOS SDK or import directly.")
    log("=" * 60)


if __name__ == "__main__":
    main()
