#!/usr/bin/env bash
set -euo pipefail

BACKEND_DOMAIN="${BACKEND_DOMAIN:-api.agourakis.med.br}"
FRONTEND_DOMAIN="${FRONTEND_DOMAIN:-darwin.agourakis.med.br}"

echo "[public] Backend https://${BACKEND_DOMAIN}/health"
curl -fsSL "https://${BACKEND_DOMAIN}/health" | jq . || curl -fsSL "https://${BACKEND_DOMAIN}/health" || true
echo

echo "[public] Backend https://${BACKEND_DOMAIN}/healthz"
curl -fsSL "https://${BACKEND_DOMAIN}/healthz" | jq . || curl -fsSL "https://${BACKEND_DOMAIN}/healthz" || true
echo

echo "[public] Frontend https://${FRONTEND_DOMAIN}/api/health"
curl -fsSL "https://${FRONTEND_DOMAIN}/api/health" | jq . || curl -fsSL "https://${FRONTEND_DOMAIN}/api/health" || true
echo

echo "[public] Done."