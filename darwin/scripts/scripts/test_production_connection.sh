#!/usr/bin/env bash
set -Eeuo pipefail

echo "DARWIN production validation - starting..."

PROJECT_ID="${PROJECT_ID:-pcs-helio}"
REGION="${REGION:-us-central1}"
API_DOMAIN="${API_DOMAIN:-api.agourakis.med.br}"
FRONTEND_DOMAIN="${FRONTEND_DOMAIN:-darwin.agourakis.med.br}"
TIMEOUT="${TIMEOUT:-15}"
STRICT="${STRICT:-1}"

pass() { echo -e "✅ $*"; }
warn() { echo -e "⚠️  $*"; }
fail() { echo -e "❌ $*"; exit 1; }

header() { echo; echo "=== $* ==="; }

check_dns() { local d="$1"; header "DNS: $d"; if command -v dig >/dev/null 2>&1; then ips=$(dig +short A "$d" || true); else ips=$(getent ahosts "$d" | awk '{print $1}' | sort -u || true); fi; if [ -z "${ips:-}" ]; then fail "DNS lookup failed for $d"; else echo "$d ->"; echo "$ips" | sed 's/^/  - /'; pass "DNS OK"; fi; }

check_tls() {
  local d="$1"
  header "TLS: $d"
  if command -v openssl >/dev/null 2>&1; then
    # Try to fetch and parse cert using OpenSSL (OpenSSL 3 friendly)
    local cert
    cert=$(</dev/null openssl s_client -servername "$d" -connect "$d:443" -showcerts 2>/dev/null | openssl x509 -noout -issuer -subject -dates 2>/dev/null || true)
    if [ -n "${cert:-}" ]; then
      echo "$cert"
      pass "TLS chain retrieved"
      return 0
    fi
  fi
  # Fallback: verify HTTPS handshake via curl
  if command -v curl >/dev/null 2>&1; then
    if curl -sS -I --max-time "$TIMEOUT" "https://$d" >/dev/null 2>&1; then
      pass "HTTPS reachable (fallback via curl)"
      return 0
    fi
  fi
  if [ "$STRICT" = "1" ]; then
    fail "Could not verify TLS for $d"
  else
    warn "Could not verify TLS for $d"
  fi
}

http_code() { local url="$1"; curl -sS -o /dev/null -m "$TIMEOUT" -w "%{http_code}" "$url"; }

check_http_ok() {
  local url="$1"
  local expect_min="${2:-200}"
  local expect_max="${3:-299}"
  header "HTTP: $url"
  code=$(http_code "$url" || echo "000")
  echo "HTTP $code"
  if [[ "$code" =~ ^[0-9]{3}$ ]] && [ "$code" -ge "$expect_min" ] && [ "$code" -le "$expect_max" ]; then
    pass "HTTP OK for $url"
  else
    if [ "$STRICT" = "1" ]; then fail "Unexpected HTTP code for $url: $code"; else warn "Unexpected HTTP code for $url: $code"; fi
  fi
}

check_header() {
  local url="$1"
  local header_name="$2"
  header "Header: $header_name @ $url"
  if curl -sSI -m "$TIMEOUT" "$url" | grep -qi "^$header_name:"; then
    pass "$header_name present"
  else
    if [ "$STRICT" = "1" ]; then fail "$header_name missing at $url"; else warn "$header_name missing at $url"; fi
  fi
}

check_openapi() {
  local url="https://${API_DOMAIN}/openapi.json"
  header "OpenAPI: $url"
  body=$(curl -fsSL -m "$TIMEOUT" "$url" 2>/dev/null || true)
  if echo "$body" | grep -q '"openapi"'; then
    pass "OpenAPI served"
  else
    if [ "$STRICT" = "1" ]; then fail "OpenAPI not available at $url"; else warn "OpenAPI not available"; fi
  fi
}

check_db_psql() {
  header "Database connectivity"
  if command -v psql >/dev/null 2>&1; then
    local url="${DATABASE_URL:-}"
    if [ -z "$url" ] && command -v gcloud >/dev/null 2>&1; then
      url=$(gcloud secrets versions access latest --secret="darwin-production-database-url" --project "$PROJECT_ID" 2>/dev/null || true)
    fi
    if [ -n "${url:-}" ]; then
      PGCONNECT_TIMEOUT="${PGCONNECT_TIMEOUT:-10}" psql "$url" -c "select 1;" >/dev/null 2>&1 && pass "PostgreSQL reachable (select 1)" || {
        if [ "$STRICT" = "1" ]; then fail "PostgreSQL check failed via psql"; else warn "PostgreSQL check failed via psql"; fi
      }
    else
      warn "DATABASE_URL not provided and secret not accessible; skipping DB check"
    fi
  else
    warn "psql not installed; skipping DB check"
  fi
}

# Run checks
header "Environment"
echo "PROJECT_ID=$PROJECT_ID"
echo "REGION=$REGION"
echo "API_DOMAIN=$API_DOMAIN"
echo "FRONTEND_DOMAIN=$FRONTEND_DOMAIN"
echo "STRICT=$STRICT"

check_dns "$API_DOMAIN"
check_tls "$API_DOMAIN"
check_http_ok "https://${API_DOMAIN}/health" 200 299
check_openapi

check_dns "$FRONTEND_DOMAIN"
check_tls "$FRONTEND_DOMAIN"
# accept 200-399 for frontend (may redirect to /app)
check_http_ok "https://${FRONTEND_DOMAIN}/" 200 399
check_header "https://${FRONTEND_DOMAIN}/" "Strict-Transport-Security"

check_db_psql

pass "All production connectivity checks completed"
exit 0