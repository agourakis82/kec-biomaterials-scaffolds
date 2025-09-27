#!/usr/bin/env bash
set -euo pipefail

# DARWIN Cloudflare Tunnel Quickstart (local -> remote)
# Usage:
#   ./cloudflared_quickstart.sh           # quick tunnels (random URLs)
#   ./cloudflared_quickstart.sh quick     # same as above
#   ./cloudflared_quickstart.sh dns       # requires named tunnel + DNS (see notes)

# Load env defaults
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

BACKEND_PORT="${BACKEND_PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_DOMAIN="${BACKEND_DOMAIN:-api.agourakis.med.br}"
FRONTEND_DOMAIN="${FRONTEND_DOMAIN:-darwin.agourakis.med.br}"
MODE="${1:-quick}"  # quick | dns

# Prereq
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
  exit 1
fi

if [ "$MODE" = "quick" ]; then
  echo "Starting Cloudflare Quick Tunnels..."
  echo "  Backend: http://localhost:${BACKEND_PORT}"
  echo "  Frontend: http://localhost:${FRONTEND_PORT}"

  TMP_DIR="$(mktemp -d)"
  # Start tunnels in background and capture logs
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${BACKEND_PORT}" >"${TMP_DIR}/backend.log" 2>&1 &
  B_PID=$!
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${FRONTEND_PORT}" >"${TMP_DIR}/frontend.log" 2>&1 &
  F_PID=$!
 
  # Give cloudflared a moment to print public URLs
  sleep 6
  BACKEND_PUBLIC="$(awk '/trycloudflare.com/ {print $NF; exit}' "${TMP_DIR}/backend.log" || true)"
  FRONTEND_PUBLIC="$(awk '/trycloudflare.com/ {print $NF; exit}' "${TMP_DIR}/frontend.log" || true)"

  echo "Public URLs:"
  echo "  Backend → ${BACKEND_PUBLIC:-pending...}"
  echo "  Frontend → ${FRONTEND_PUBLIC:-pending...}"
  echo "Tunnels running. Press Ctrl+C to stop."

  # Stream logs
  tail -n +1 -f "${TMP_DIR}/backend.log" "${TMP_DIR}/frontend.log" &
  T_PID=$!
  # Wait for either tunnel to exit
  wait -n $B_PID $F_PID || true
  kill $T_PID 2>/dev/null || true
  exit 0
fi

if [ "$MODE" = "dns" ]; then
  echo "DNS mode requested for:"
  echo "  ${BACKEND_DOMAIN} -> http://localhost:${BACKEND_PORT}"
  echo "  ${FRONTEND_DOMAIN} -> http://localhost:${FRONTEND_PORT}"
  echo ""
  echo "Requirements:"
  echo "  1) cloudflared login"
  echo "  2) cloudflared tunnel create darwin"
  echo "  3) cloudflared tunnel route dns darwin ${BACKEND_DOMAIN}"
  echo "  4) cloudflared tunnel route dns darwin ${FRONTEND_DOMAIN}"
  echo "  5) cloudflared tunnel run darwin --metrics 127.0.0.1:0"
  echo ""
  echo "Tip: place this run command under a systemd service for persistence."
  echo "For now, falling back to quick tunnels. Configure the named tunnel later."
  exec "$0" quick
fi

echo "Unknown mode: ${MODE}. Use: quick | dns"
exit 2