<#
.SYNOPSIS
Builds the turbovec Python wheel for Windows natively.
.DESCRIPTION
Ensures that the nightly rust toolchain and maturin are installed, then compiles the wheel.
#>

$ErrorActionPreference = "Stop"

Write-Host "Ensures Nightly Rust Toolchain..."
rustup default nightly

Write-Host "Installing/Upgrading Maturin..."
python -m pip install --upgrade pip maturin

Write-Host "Building Turbovec Wheel..."
cd turbovec/turbovec-python
maturin build --release --out target/wheels

Write-Host "Build complete! Wheels located in: turbovec/turbovec-python/target/wheels"
