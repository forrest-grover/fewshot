#!/usr/bin/env bash
# Regenerate src/api-schema.ts (TypeScript types for the server's OpenAPI schema) from the backend in
# this checkout. Usage, from web/: npm run gen:api   (or npm run check:api to fail if it is out of date).
# Needs Python with fewshot importable: ../.env/bin/python if present, else $PYTHON, else python.
set -euo pipefail
cd "$(dirname "$0")/.."
py=${PYTHON:-python}
if [[ -x ../.env/bin/python ]]; then py="$PWD/../.env/bin/python"; fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# Importing fewshot.api doesn't load any model; a scratch FEWSHOT_HOME keeps it from touching real data.
(cd .. && FEWSHOT_HOME="$tmp/home" "$py" -c \
  'import json; from fewshot.api import app; print(json.dumps(app.openapi(), indent=1, sort_keys=True))') > "$tmp/openapi.json"
npx --no-install openapi-typescript "$tmp/openapi.json" -o "$tmp/api-schema.ts" >/dev/null
{
  echo "// Generated from the server's OpenAPI schema by scripts/gen-api-types.sh (npm run gen:api). Don't edit."
  echo "/* eslint-disable */"
  cat "$tmp/api-schema.ts"
} > "$tmp/out.ts"
if [[ ${1:-} == --check ]]; then
  if ! diff -q "$tmp/out.ts" src/api-schema.ts >/dev/null; then
    echo "src/api-schema.ts is out of date with the server's API; run: npm run gen:api" >&2
    diff -u src/api-schema.ts "$tmp/out.ts" | head -40 >&2 || true
    exit 1
  fi
else
  mv "$tmp/out.ts" src/api-schema.ts
fi
