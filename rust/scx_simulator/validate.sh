#!/bin/bash
# validate.sh - Local validation script for scx_simulator workspace
# Run this before committing to ensure code quality.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Running cargo fmt --check ==="
cargo fmt --all -- --check

echo ""
echo "=== Running cargo clippy ==="
cargo clippy --all -- -D warnings

echo ""
echo "=== Running cargo clippy (frida feature) ==="
cargo clippy -p scx_simulator --features frida -- -D warnings

echo ""
echo "=== Running cargo test ==="
cargo test --all

echo ""
echo "=== Running cargo test (frida feature) ==="
# Frida Stalker requires JIT/mmap permissions that may be blocked in sandboxed
# environments. Set SCX_SIM_NO_FRIDA_STALKER=1 to skip Stalker-runtime tests
# while still running cooperative-mode determinism tests. In CI, Stalker tests
# run without this variable.
SCX_SIM_NO_FRIDA_STALKER="${SCX_SIM_NO_FRIDA_STALKER:-1}" \
    cargo test -p scx_simulator --features frida

echo ""
echo "=== All checks passed ==="
