#!/usr/bin/env python3
"""
MNEMOS Portable Memory Appliance — Offline Bundle Preparation Script
=====================================================================

Run this script on an INTERNET-CONNECTED machine to download and package
everything needed to install the Portable Memory Appliance on an air-gapped
or offline target.

Output: A self-contained directory (default: ./mnemos_offline_bundle/) containing:
  - All Python wheel dependencies
  - The BAAI/bge-base-en-v1.5 embedding model snapshot
  - A copy of the MNEMOS source tree
  - The offline install script

Usage:
    python scripts/prepare_offline_bundle.py
    python scripts/prepare_offline_bundle.py --output-dir /path/to/usb/mnemos_bundle
    python scripts/prepare_offline_bundle.py --platform linux   # default: auto-detect
"""

import os
import sys
import shutil
import subprocess
import argparse
import platform
import json
import datetime

PORTABLE_REQUIREMENTS = [
    "sentence-transformers>=2.2.2",
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "python-dotenv>=1.0",
    "requests>=2.31",
    "PyPDF2>=3.0",
    "maturin>=1.0",
]

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def log(msg: str):
    print(f"[prepare] {msg}")


def detect_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "win_amd64"
    elif system == "linux":
        if "aarch64" in machine or "arm" in machine:
            return "linux_aarch64"
        return "manylinux2014_x86_64"
    elif system == "darwin":
        return "macosx_arm64" if "arm" in machine else "macosx_x86_64"
    return "unknown"


def download_wheels(output_dir: str, target_platform: str):
    """Download all portable profile dependencies as wheels."""
    wheels_dir = os.path.join(output_dir, "wheels")
    os.makedirs(wheels_dir, exist_ok=True)

    log(f"Downloading wheels for platform: {target_platform}")

    for req in PORTABLE_REQUIREMENTS:
        log(f"  Downloading: {req}")
        cmd = [
            sys.executable, "-m", "pip", "download",
            req,
            "--dest", wheels_dir,
            "--no-cache-dir",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  WARNING: Failed to download {req}: {result.stderr.strip()}")
        else:
            log(f"  OK: {req}")


def download_embedding_model(output_dir: str):
    """Download the BGE embedding model for offline use."""
    model_dir = os.path.join(output_dir, "models", EMBEDDING_MODEL.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    log(f"Downloading embedding model: {EMBEDDING_MODEL}")
    log("  (This may take several minutes for the first download...)")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        model.save(model_dir)
        log(f"  OK: Model saved to {model_dir}")

        # Verify the model can be loaded from the saved path
        test_model = SentenceTransformer(model_dir)
        test_emb = test_model.encode(["offline test"], normalize_embeddings=True)
        assert test_emb.shape[1] == 768, f"Expected 768-dim, got {test_emb.shape[1]}"
        log(f"  VERIFIED: Model loads from saved path, produces 768-dim embeddings.")
    except ImportError:
        log("  ERROR: sentence-transformers not installed on this machine.")
        log("  Install it first: pip install sentence-transformers torch")
        sys.exit(1)
    except Exception as e:
        log(f"  ERROR: Failed to download model: {e}")
        sys.exit(1)


def copy_mnemos_source(output_dir: str):
    """Copy the MNEMOS source tree (excluding runtime data and .git)."""
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(output_dir, "mnemos")

    EXCLUDE_DIRS = {
        ".git", ".github", "__pycache__", "runtime", ".venv", "venv",
        "node_modules", ".pytest_cache", "data", "benchmarks/outputs",
        ".eggs", "dist", "build",
    }
    EXCLUDE_FILES = {".env", "smoke.py", "list_models.py"}

    log(f"Copying MNEMOS source from {src_root}")

    def ignore_patterns(directory, contents):
        rel = os.path.relpath(directory, src_root)
        ignored = set()
        for item in contents:
            full_rel = os.path.join(rel, item) if rel != "." else item
            if item in EXCLUDE_DIRS or full_rel in EXCLUDE_DIRS:
                ignored.add(item)
            if item in EXCLUDE_FILES:
                ignored.add(item)
            if item.endswith((".pyc", ".pyo", ".db")):
                ignored.add(item)
        return ignored

    if os.path.exists(dest):
        shutil.rmtree(dest)

    shutil.copytree(src_root, dest, ignore=ignore_patterns)
    log(f"  OK: Source tree copied to {dest}")


def copy_install_script(output_dir: str):
    """Copy the offline install script into the bundle."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "install_offline_bundle.py"
    )
    dest = os.path.join(output_dir, "install_offline_bundle.py")

    if os.path.exists(src):
        shutil.copy2(src, dest)
        log(f"  OK: Install script copied to bundle.")
    else:
        log(f"  WARNING: {src} not found. Generate it separately.")


def write_manifest(output_dir: str, target_platform: str):
    """Write a manifest describing the bundle contents."""
    manifest = {
        "bundle_type": "mnemos_portable_offline_bundle",
        "bundle_version": "1.0",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "created_by": platform.node(),
        "target_platform": target_platform,
        "python_version": platform.python_version(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": 768,
        "portable_requirements": PORTABLE_REQUIREMENTS,
        "contents": {
            "wheels/": "Pre-downloaded pip wheel packages",
            "models/": "Pre-downloaded embedding model snapshot",
            "mnemos/": "MNEMOS source tree",
            "install_offline_bundle.py": "Offline installation script",
        },
    }

    manifest_path = os.path.join(output_dir, "bundle_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"  OK: Manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an offline installation bundle for the MNEMOS Portable Memory Appliance."
    )
    parser.add_argument(
        "--output-dir",
        default="mnemos_offline_bundle",
        help="Directory to create the bundle in (default: ./mnemos_offline_bundle/)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Target platform (default: auto-detect). Options: win_amd64, manylinux2014_x86_64",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip downloading the embedding model (if you already have it staged).",
    )
    args = parser.parse_args()

    target_platform = args.platform or detect_platform()
    output_dir = os.path.abspath(args.output_dir)

    log("=" * 60)
    log("MNEMOS Portable Memory Appliance — Offline Bundle Prep")
    log("=" * 60)
    log(f"Target platform : {target_platform}")
    log(f"Output directory : {output_dir}")
    log(f"Python version  : {platform.python_version()}")
    log("")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download wheels
    download_wheels(output_dir, target_platform)

    # Step 2: Download embedding model
    if not args.skip_model:
        download_embedding_model(output_dir)
    else:
        log("Skipping model download (--skip-model).")

    # Step 3: Copy source
    copy_mnemos_source(output_dir)

    # Step 4: Copy install script
    copy_install_script(output_dir)

    # Step 5: Write manifest
    write_manifest(output_dir, target_platform)

    # Summary
    log("")
    log("=" * 60)
    log("BUNDLE COMPLETE")
    log("=" * 60)
    log(f"Bundle location: {output_dir}")
    log("")
    log("To install on an air-gapped target:")
    log("  1. Copy the entire bundle directory to the target machine.")
    log("  2. On the target, run:")
    log(f"       python {os.path.join(output_dir, 'install_offline_bundle.py')}")
    log("")
    log("The target machine still requires:")
    log("  - Python 3.11+")
    log("  - Rust Nightly toolchain (for Turbovec compilation)")
    log("    Install Rust beforehand or include rustup-init in the bundle.")
    log("=" * 60)


if __name__ == "__main__":
    main()
