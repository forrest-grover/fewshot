#!/usr/bin/env bash
# Run every check that CI runs. Usage: scripts/check.sh  (from the repo root, with the project env active
# or ./.env present).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .env/bin/python ]]; then export PATH="$PWD/.env/bin:$PATH"; fi

step() { printf '\n== %s\n' "$1"; }
step ruff;    ruff check .
step mypy;    mypy
step pylint;  pylint --fail-under=10 fewshot tests
step pytest;  pytest -q --cov --cov-report=term-missing
cd web
step tsc;     npm run -s typecheck
step lint;    npm run -s lint
step build;   npm run -s build
printf '\nAll checks passed.\n'
