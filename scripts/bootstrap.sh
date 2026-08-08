#!/usr/bin/env bash
# Create the local development environment (Linux/macOS).
#
#   bash scripts/bootstrap.sh
#
# Run this once per machine.  The .venv it creates is deliberately gitignored:
# a venv records the absolute path of the interpreter that built it, so copying
# one between machines produces an environment that half-works in confusing
# ways.  Ship the repo; recreate the environment.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

# 3.11 is the pinned target (architecture v1.1 §0.4) and is what Kaggle runs.
python=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        case "$version" in
            3.11|3.12) python="$candidate"; break ;;
        esac
    fi
done

if [ -z "$python" ]; then
    echo "Python 3.11 or 3.12 not found. Install 3.11 and re-run." >&2
    exit 1
fi
echo "Using $($python -c 'import sys; print(sys.executable)')"

if [ -d .venv ]; then
    echo ".venv already exists; reusing it. Delete it to rebuild from scratch."
else
    "$python" -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e .

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
echo "Then check the install with:"
echo "    pytest -q"
