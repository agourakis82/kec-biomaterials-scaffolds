#!/usr/bin/env bash
set -euo pipefail
URL="${RUN_URL:-PREENCHER_URL}"
KEY="${DARWIN_API_KEY:-PREENCHER_API_KEY}"

curl -s -XPOST "${URL}/rag-plus/search" \
 -H "Content-Type: application/json" \
 -H "X-API-KEY: ${KEY}" \
 -d '{"q":"Explain percolation diameter in porous tissue scaffolds","k":4}' | jq .
