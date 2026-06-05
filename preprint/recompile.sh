#!/usr/bin/env bash
#
# Recompile the manuscript — PDF + HTML — and verify both outputs.
# Run from the preprint/ directory:
#   ./recompile.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Rendering HTML ==="
quarto render main.qmd --to html
echo ""

echo "=== Rendering PDF ==="
quarto render main.qmd --to pdf
echo ""

# ── Verify outputs ────────────────────────────────────────────────
FAIL=0
for fmt in pdf html; do
    out="main.${fmt}"
    if [[ ! -f "$out" ]]; then
        echo "ERROR: $out was not created."
        FAIL=1
    elif [[ ! -s "$out" ]]; then
        echo "ERROR: $out is empty."
        FAIL=1
    else
        size=$(stat --printf="%s" "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null)
        echo "OK: $out (${size} bytes)"
    fi
done

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "=== Manuscript compiled successfully ==="
else
    echo "=== ERRORS detected above ==="
    exit 1
fi
