#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ID="${PROJECT_ID:-pcs-helio}"
REGION="${REGION:-us-central1}"
FRONTEND_DOMAIN="${FRONTEND_DOMAIN:-darwin.agourakis.med.br}"
BACKEND_DOMAIN="${BACKEND_DOMAIN:-api.agourakis.med.br}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-}"

print_status() { echo -e "${BLUE}[*]${NC} $*"; }
print_success() { echo -e "${GREEN}[ok]${NC} $*"; }
print_warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
print_error() { echo -e "${RED}[err]${NC} $*"; }

header() {
  echo -e "${CYAN}${BOLD}Fix Frontend Domain Mapping (Cloud Run)${NC}"
  echo "Project: ${PROJECT_ID} | Region: ${REGION}"
  echo "Frontend domain: ${FRONTEND_DOMAIN}"
  echo
}

require_tools() {
  for t in gcloud curl dig; do
    if ! command -v "$t" >/dev/null 2>&1; then
      print_error "Missing tool: $t"
      exit 1
    fi
  done
}

check_auth() {
  if [ -z "$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" ]; then
    print_error "Not authenticated in gcloud."
    echo "Run: gcloud auth login && gcloud auth application-default login"
    exit 1
  fi
}

ensure_project() {
  print_status "Setting gcloud project to ${PROJECT_ID}"
  gcloud config set project "${PROJECT_ID}" >/dev/null
}

choose_frontend_service() {
  if [ -n "${FRONTEND_SERVICE}" ]; then
    print_status "Using provided FRONTEND_SERVICE=${FRONTEND_SERVICE}"
    return 0
  fi
  print_status "Detecting frontend Cloud Run service..."
  local services
  services=$(gcloud run services list --region "${REGION}" --format='value(metadata.name)' || true)
  if echo "${services}" | grep -qx 'darwin-frontend-web'; then
    FRONTEND_SERVICE='darwin-frontend-web'
  elif echo "${services}" | grep -qx 'app-agourakis-med-br'; then
    FRONTEND_SERVICE='app-agourakis-med-br'
  else
    # Fallback: pick first service containing 'front'
    local guess
    guess=$(echo "${services}" | grep -i 'front' | head -n1 || true)
    if [ -n "${guess}" ]; then
      FRONTEND_SERVICE="${guess}"
    else
      print_error "Could not auto-detect a frontend service. Services found:"
      echo "${services:-<none>}"
      echo "Set env FRONTEND_SERVICE=your-service and re-run."
      exit 1
    fi
  fi
  print_success "Selected frontend service: ${FRONTEND_SERVICE}"
}

show_service_urls() {
  local f_url b_url
  f_url=$(gcloud run services describe "${FRONTEND_SERVICE}" --region "${REGION}" --format='value(status.url)' 2>/dev/null || true)
  b_url=$(gcloud run services describe 'darwin-backend-api' --region "${REGION}" --format='value(status.url)' 2>/dev/null || true)
  print_status "Cloud Run URLs:"
  echo "  Frontend: ${f_url:-<not found>}"
  echo "  Backend:  ${b_url:-<not found>}"
}

ensure_domain_mapping() {
  print_status "Creating/updating domain mapping for ${FRONTEND_DOMAIN} -> ${FRONTEND_SERVICE}"
  if gcloud run domain-mappings describe "${FRONTEND_DOMAIN}" --region "${REGION}" >/dev/null 2&>1; then
    print_status "Domain mapping exists. Updating target service..."
    gcloud run domain-mappings update \
      --service="${FRONTEND_SERVICE}" \
      --domain="${FRONTEND_DOMAIN}" \
      --region="${REGION}" \
      --quiet
  else
    print_status "Creating domain mapping..."
    gcloud run domain-mappings create \
      --service="${FRONTEND_SERVICE}" \
      --domain="${FRONTEND_DOMAIN}" \
      --region="${REGION}" \
      --quiet
  fi
  print_success "Domain mapping command completed"
}

show_dns_records() {
  print_status "Required DNS records for ${FRONTEND_DOMAIN}:"
  gcloud run domain-mappings describe "${FRONTEND_DOMAIN}" --region "${REGION}" \
    --format="table(status.resourceRecords[].name,status.resourceRecords[].rrdata,status.resourceRecords[].type)" || true
  echo
  print_status "Current public DNS resolution:"
  echo -n "CNAME: "; dig +short CNAME "${FRONTEND_DOMAIN}" || true
  echo -n "A:     "; dig +short A "${FRONTEND_DOMAIN}" || true
  echo -n "AAAA:  "; dig +short AAAA "${FRONTEND_DOMAIN}" || true
  echo
  print_warn "If you use Cloudflare, set CNAME to ghs.googlehosted.com with Proxy disabled (DNS only - grey cloud)."
}

wait_for_certificate() {
  print_status "Waiting for certificate provisioning (up to ~10 min)..."
  local i=0
  while [ $i -lt 30 ]; do
    local ready msg
    ready=$(gcloud run domain-mappings describe "${FRONTEND_DOMAIN}" --region "${REGION}" --format="value(status.conditions[?type=Ready].status)" 2>/dev/null || echo "")
    msg=$(gcloud run domain-mappings describe "${FRONTEND_DOMAIN}" --region "${REGION}" --format="value(status.conditions[?type=Ready].message)" 2>/dev/null || echo "")
    if [ "${ready}" = "True" ]; then
      print_success "Certificate Ready"
      return 0
    fi
    echo "  status=${ready:-<unknown>} message=${msg:-<none>}"
    sleep 20
    i=$((i+1))
  done
  print_warn "Timeout waiting for certificate. It may still be provisioning; try again later."
}

quick_tests() {
  print_status "Quick HTTPS check on ${FRONTEND_DOMAIN}"
  if curl -fsSI --max-time 20 "https://${FRONTEND_DOMAIN}" >/dev/null 2>&1; then
    print_success "Frontend domain responds over HTTPS"
  else
    print_warn "HTTPS check failed (may still be provisioning)."
  fi
}

main() {
  header
  require_tools
  check_auth
  ensure_project
  choose_frontend_service
  show_service_urls
  ensure_domain_mapping
  show_dns_records
  wait_for_certificate
  quick_tests
  echo
  print_status "Next steps:"
  echo "  - Re-test: curl -I https://${FRONTEND_DOMAIN}"
  echo "  - Cloud Console > Run > Domain mappings: verify status Ready"
  echo "  - If stuck: verify DNS CNAME to ghs.googlehosted.com and Cloudflare proxy OFF"
}

main "$@"