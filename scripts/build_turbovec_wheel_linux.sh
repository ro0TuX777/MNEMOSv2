#!/usr/bin/env bash
set -e

echo "Ensures Nightly Rust Toolchain..."
rustup default nightly

echo "Installing/Upgrading Maturin..."
python3 -m pip install --upgrade pip maturin

echo "Building Turbovec Wheel..."
cd turbovec/turbovec-python
maturin build --release --out target/wheels

echo "Build complete! Wheels located in: turbovec/turbovec-python/target/wheels"
