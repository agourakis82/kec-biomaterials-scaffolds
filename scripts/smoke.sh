#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-all}"

API_LOCAL_URL="${API_LOCAL_URL:-http://localhost:8090/healthz}"
WEB_LOCAL_URL="${WEB_LOCAL_URL:-http://localhost:3000/api/health}"
Q1_LOCAL_URL="${Q1_LOCAL_URL:-http://localhost:8090/q1-scholar/health}"

CF_API_HOST="${CF_API_HOST:-api-local.agourakis.med.br}"
CF_WEB_HOST="${CF_WEB_HOST:-darwin-local.agourakis.med.br}"
API_PUBLIC_URL="https://${CF_API_HOST}/healthz"
WEB_PUBLIC_URL="https://${CF_WEB_HOST}/api/health"
Q1_PUBLIC_URL="https://${CF_API_HOST}/q1-scholar/health"

PASS=0
FAIL=0

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

check_url() {
  local name="$1"
  local url="$2"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 15 "$url" || echo "000")"
  if [[ "$code" == "200" ]]; then
    echo "$(ts) PASS $name $url ($code)"
    PASS=$((PASS+1))
    return 0
  else
    echo "$(ts) FAIL $name $url ($code)"
    FAIL=$((FAIL+1))
    return 1
  fi
}

dns_info() {
  local host="$1"
  echo -n "$(ts) DNS  $host -> "
  if getent ahosts "$host" >/dev/null 2>&1; then
    getent ahosts "$host" | awk '{print $1}' | sort -u | xargs
  else
    echo "unresolved"
  fi
}

if [[ "$MODE" == "local" || "$MODE" == "all" ]]; then
  echo "== Local smoke =="
  check_url "api-local" "$API_LOCAL_URL"
  check_url "web-local" "$WEB_LOCAL_URL"
  check_url "q1-local" "$Q1_LOCAL_URL"
fi

if [[ "$MODE" == "public" || "$MODE" == "all" ]]; then
  echo "== Public smoke (Cloudflare Tunnel) =="
  dns_info "$CF_API_HOST"
  dns_info "$CF_WEB_HOST"
  check_url "api-public" "$API_PUBLIC_URL"
  check_url "web-public" "$WEB_PUBLIC_URL"
  check_url "q1-public" "$Q1_PUBLIC_URL"
fi

echo "== Summary =="
echo "PASS=$PASS FAIL=$FAIL"

# Exit non-zero if any failure
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi

exit 0