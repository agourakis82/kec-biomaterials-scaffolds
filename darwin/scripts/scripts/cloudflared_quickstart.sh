#!/usr/bin/env bash
set -euo pipefail

# Cloudflare Tunnel quickstart for DARWIN local hybrid
# Exposes backend (8080) and frontend (3000) via trycloudflare.com ephemeral URLs

command -v cloudflared >/dev/null 2>&1 || { echo "cloudflared not found. Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"; exit 1; }

LOG_DIR="./.cloudflared"
mkdir -p "$LOG_DIR"

start_tunnel() {
  local name="$1"
  local port="$2"
  local log="$LOG_DIR/${name}.log"
  # Kill previous
  if pgrep -f "cloudflared tunnel --url http://localhost:${port}" >/dev/null 2>&1; then
    pkill -f "cloudflared tunnel --url http://localhost:${port}" || true
    sleep 1
  fi
  nohup cloudflared tunnel --no-autoupdate --url "http://localhost:${port}" > "$log" 2>&1 &
  local url=""
  for i in {1..30}; do
    url=$(grep -Eo 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$log" | head -n1 || true)
    if [[ -n "$url" ]]; then
      echo "${name}_url=${url}"
      break
    fi
    sleep 1
  done
  if [[ -z "$url" ]]; then
    echo "failed_to_get_url_for_${name}. Check $log"
  fi
}

echo "Starting Cloudflare quick tunnels (backend:8080, frontend:3000)..."
start_tunnel "backend" 8080
start_tunnel "frontend" 3000

echo "Logs in $LOG_DIR (tail -f .cloudflared/backend.log)"
echo "Press Ctrl+C to stop; or kill processes: pkill -f 'cloudflared tunnel --url'"