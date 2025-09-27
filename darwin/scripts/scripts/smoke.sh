#!/usr/bin/env bash
set -euo pipefail

DARWIN_URL=${DARWIN_URL:-"http://localhost:8000"}
DARWIN_API_KEY=${DARWIN_API_KEY:-""}

hdr=("-H" "Content-Type: application/json")
if [[ -n "$DARWIN_API_KEY" ]]; then
  hdr+=("-H" "X-API-KEY: $DARWIN_API_KEY")
fi

echo "[1] /rag-plus/search"
curl -sS -X POST "${DARWIN_URL}/rag-plus/search" "${hdr[@]}" \
  -d '{"query":"What is RAG++?"}' | head -c 400 || true
echo -e "\n"

echo "[2] /rag-plus/iterative"
curl -sS -X POST "${DARWIN_URL}/rag-plus/iterative" "${hdr[@]}" \
  -d '{"query":"Summarize iterative reasoning"}' | head -c 400 || true
echo -e "\n"

echo "[3] /tree-search/puct"
curl -sS -X POST "${DARWIN_URL}/tree-search/puct" "${hdr[@]}" \
  -d '{"root":"start","budget":20,"c_puct":1.2}' | head -c 400 || true
echo -e "\n"

echo "[4] /discovery/run"
curl -sS -X POST "${DARWIN_URL}/discovery/run" "${hdr[@]}" \
  -d '{"run_once":true}' | head -c 400 || true
echo -e "\nDone."

