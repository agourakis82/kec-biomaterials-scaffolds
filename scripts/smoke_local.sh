#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_BASE_URL:-http://localhost:8090}"
WEB_URL="${FRONTEND_BASE_URL:-http://localhost:3000}"
JUPYTER_URL="http://localhost:${JUPYTER_PORT:-8888}"
CHROMA_URL="http://localhost:${CHROMA_PORT:-8000}"
OLLAMA_URL="http://localhost:${OLLAMA_PORT:-11434}"

retry() {
  local tries="${1:-20}"; shift || true
  local delay="${1:-2}"; shift || true
  local cmd=("$@")
  for ((i=1;i<=tries;i++)); do
    if "${cmd[@]}" ; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

echo "[smoke] Checking Redis (ping)"
retry 30 1 bash -lc 'docker compose exec -T redis redis-cli ping | grep -q PONG' && echo "OK: Redis"

echo "[smoke] API /health"
retry 30 2 curl -fsS "${API_URL}/health" && echo && echo "OK: API /health"

echo "[smoke] API /healthz"
retry 15 2 curl -fsS "${API_URL}/healthz" && echo && echo "OK: API /healthz"

echo "[smoke] Web /api/health"
retry 30 2 curl -fsS "${WEB_URL}/api/health" && echo && echo "OK: WEB /api/health"

echo "[smoke] Jupyter (optional)"
if curl -fsS "${JUPYTER_URL}/api" >/dev/null 2>&1; then
  echo "OK: Jupyter"
else
  echo "WARN: Jupyter not responding yet (optional)"
fi

echo "[smoke] ChromaDB heartbeat"
retry 30 2 curl -fsS "${CHROMA_URL}/api/v1/heartbeat" && echo && echo "OK: ChromaDB"

echo "[smoke] Ollama tags"
if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "OK: Ollama"
else
  echo "WARN: Ollama not responding yet (optional, requires GPU/runtime)"
fi

echo "[smoke] All checks attempted."
