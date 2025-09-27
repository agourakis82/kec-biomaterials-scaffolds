#!/bin/bash

# =============================================================================
# DARWIN Monitoring Setup Script
# Script para configurar dashboards personalizados, alerting rules e métricas
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
PROJECT_ID=""
REGION="us-central1"
VERIFY_ONLY="false"
EMAIL_ADDRESSES=""
SLACK_WEBHOOK=""
PAGERDUTY_KEY=""
BUDGET_AMOUNT="500"
VERBOSE="false"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1" >&2
    fi
}

show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    DARWIN MONITORING                          ║
    ║                    Configuration Setup                        ║
    ║                                                               ║
    ║          Dashboards, Alerts, Metrics & Performance           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DARWIN Monitoring Setup Script

OPTIONS:
    -p, --project-id PROJECT_ID       GCP Project ID (required)
    -e, --environment ENVIRONMENT     Environment [dev|staging|production] (default: production)
    -r, --region REGION              GCP Region (default: us-central1)
    
    Monitoring configuration:
    --email EMAILS                   Comma-separated email addresses for alerts
    --slack-webhook URL              Slack webhook URL for notifications
    --pagerduty-key KEY              PagerDuty service integration key
    --budget AMOUNT                  Monthly budget amount in USD (default: 500)
    
    Actions:
    --verify                         Verify monitoring configuration only
    -v, --verbose                    Enable verbose logging
    -h, --help                       Show this help message

EXAMPLES:
    $0 -p my-project --email "admin@company.com,ops@company.com"
    $0 -p my-project --verify
    $0 -p my-project --email "admin@company.com" --slack-webhook "https://hooks.slack.com/..."
    $0 -p my-project -e staging --budget 200

ENVIRONMENT VARIABLES:
    DARWIN_PROJECT_ID                Project ID
    DARWIN_ENVIRONMENT               Environment
    DARWIN_REGION                    GCP Region
    DARWIN_EMAIL_ADDRESSES           Email addresses for alerts
    DARWIN_SLACK_WEBHOOK             Slack webhook URL
    DARWIN_PAGERDUTY_KEY             PagerDuty key
    DARWIN_BUDGET_AMOUNT             Budget amount

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access project $PROJECT_ID"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

verify_monitoring_resources() {
    log_info "Verifying monitoring resources..."
    
    local issues=()
    
    # Check if monitoring dashboard exists
    log_debug "Checking monitoring dashboards..."
    local dashboards
    dashboards=$(gcloud alpha monitoring dashboards list \
        --filter="displayName~'DARWIN' OR displayName~'darwin'" \
        --format="value(displayName)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$dashboards" ]]; then
        log_success "Found monitoring dashboards:"
        echo "$dashboards" | while read -r dashboard; do
            log_success "  - $dashboard"
        done
    else
        issues+=("No DARWIN monitoring dashboards found")
    fi
    
    # Check alert policies
    log_debug "Checking alert policies..."
    local policies
    policies=$(gcloud alpha monitoring policies list \
        --filter="displayName~'DARWIN' OR displayName~'darwin'" \
        --format="value(displayName)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$policies" ]]; then
        log_success "Found alert policies:"
        echo "$policies" | while read -r policy; do
            log_success "  - $policy"
        done
    else
        issues+=("No DARWIN alert policies found")
    fi
    
    # Check uptime checks
    log_debug "Checking uptime checks..."
    local uptime_checks
    uptime_checks=$(gcloud alpha monitoring uptime list \
        --format="value(displayName)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$uptime_checks" ]]; then
        log_success "Found uptime checks:"
        echo "$uptime_checks" | while read -r check; do
            log_success "  - $check"
        done
    else
        issues+=("No uptime checks found")
    fi
    
    # Check notification channels
    log_debug "Checking notification channels..."
    local channels
    channels=$(gcloud alpha monitoring channels list \
        --format="value(displayName,type)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$channels" ]]; then
        log_success "Found notification channels:"
        echo "$channels" | while read -r channel; do
            log_success "  - $channel"
        done
    else
        log_warning "No notification channels configured"
    fi
    
    # Check log-based metrics
    log_debug "Checking log-based metrics..."
    local metrics
    metrics=$(gcloud logging metrics list \
        --filter="name~'darwin'" \
        --format="value(name)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$metrics" ]]; then
        log_success "Found log-based metrics:"
        echo "$metrics" | while read -r metric; do
            log_success "  - $metric"
        done
    else
        issues+=("No DARWIN log-based metrics found")
    fi
    
    # Summary
    if [[ ${#issues[@]} -eq 0 ]]; then
        log_success "All monitoring resources are configured correctly"
        return 0
    else
        log_warning "Monitoring verification issues found:"
        for issue in "${issues[@]}"; do
            log_warning "  - $issue"
        done
        return 1
    fi
}

test_service_connectivity() {
    log_info "Testing service connectivity..."
    
    # Test backend service
    local backend_service="darwin-${ENVIRONMENT}-backend"
    if gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        local backend_url
        backend_url=$(gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
        
        log_debug "Testing backend health endpoint: $backend_url/health"
        if curl -f -s -m 10 "$backend_url/health" >/dev/null 2>&1; then
            log_success "Backend service is healthy"
        else
            log_warning "Backend service health check failed"
        fi
        
        log_debug "Testing backend metrics endpoint: $backend_url/metrics"
        if curl -f -s -m 10 "$backend_url/metrics" >/dev/null 2>&1; then
            log_success "Backend metrics endpoint is accessible"
        else
            log_warning "Backend metrics endpoint not accessible"
        fi
    else
        log_warning "Backend service not found or not deployed"
    fi
    
    # Test frontend service
    local frontend_service="darwin-${ENVIRONMENT}-frontend"
    if gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        local frontend_url
        frontend_url=$(gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
        
        log_debug "Testing frontend endpoint: $frontend_url"
        if curl -f -s -m 10 "$frontend_url/" >/dev/null 2>&1; then
            log_success "Frontend service is accessible"
        else
            log_warning "Frontend service not accessible"
        fi
    else
        log_warning "Frontend service not found or not deployed"
    fi
    
    # Test custom domains
    local api_domain="api.agourakis.med.br"
    local frontend_domain="darwin.agourakis.med.br"
    
    log_debug "Testing API domain: $api_domain"
    if curl -f -s -m 10 "https://$api_domain/health" >/dev/null 2>&1; then
        log_success "API domain is accessible with SSL"
    else
        log_warning "API domain not accessible (may be DNS/SSL provisioning)"
    fi
    
    log_debug "Testing frontend domain: $frontend_domain"
    if curl -f -s -m 10 "https://$frontend_domain/" >/dev/null 2>&1; then
        log_success "Frontend domain is accessible with SSL"
    else
        log_warning "Frontend domain not accessible (may be DNS/SSL provisioning)"
    fi
}

check_ssl_certificates() {
    log_info "Checking SSL certificate status..."
    
    # Check managed SSL certificates
    local ssl_certs
    ssl_certs=$(gcloud compute ssl-certificates list \
        --filter="name~'darwin'" \
        --format="table(name,managed.domains.list():label=DOMAINS,managed.status:label=STATUS)" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -n "$ssl_certs" ]]; then
        log_success "SSL Certificate Status:"
        echo "$ssl_certs"
        
        # Check if any certificates are still provisioning
        local provisioning
        provisioning=$(gcloud compute ssl-certificates list \
            --filter="name~'darwin' AND managed.status:PROVISIONING" \
            --format="value(name)" \
            --project="$PROJECT_ID" 2>/dev/null || echo "")
        
        if [[ -n "$provisioning" ]]; then
            log_warning "SSL certificates still provisioning (this can take 10-60 minutes):"
            echo "$provisioning" | while read -r cert; do
                log_warning "  - $cert"
            done
        fi
    else
        log_warning "No managed SSL certificates found"
    fi
}

generate_monitoring_report() {
    log_info "Generating monitoring report..."
    
    local report_file="monitoring-report-$(date +%Y%m%d-%H%M%S).json"
    
    # Collect monitoring information
    local monitoring_data="{}"
    
    # Dashboards
    local dashboards
    dashboards=$(gcloud alpha monitoring dashboards list \
        --format="json" \
        --project="$PROJECT_ID" 2>/dev/null || echo "[]")
    monitoring_data=$(echo "$monitoring_data" | jq --argjson dashboards "$dashboards" '. + {dashboards: $dashboards}')
    
    # Alert policies
    local policies
    policies=$(gcloud alpha monitoring policies list \
        --format="json" \
        --project="$PROJECT_ID" 2>/dev/null || echo "[]")
    monitoring_data=$(echo "$monitoring_data" | jq --argjson policies "$policies" '. + {alertPolicies: $policies}')
    
    # Notification channels
    local channels
    channels=$(gcloud alpha monitoring channels list \
        --format="json" \
        --project="$PROJECT_ID" 2>/dev/null || echo "[]")
    monitoring_data=$(echo "$monitoring_data" | jq --argjson channels "$channels" '. + {notificationChannels: $channels}')
    
    # Uptime checks
    local uptime
    uptime=$(gcloud alpha monitoring uptime list \
        --format="json" \
        --project="$PROJECT_ID" 2>/dev/null || echo "[]")
    monitoring_data=$(echo "$monitoring_data" | jq --argjson uptime "$uptime" '. + {uptimeChecks: $uptime}')
    
    # Add metadata
    monitoring_data=$(echo "$monitoring_data" | jq \
        --arg project "$PROJECT_ID" \
        --arg environment "$ENVIRONMENT" \
        --arg region "$REGION" \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '. + {metadata: {project: $project, environment: $environment, region: $region, timestamp: $timestamp}}')
    
    # Save report
    echo "$monitoring_data" | jq '.' > "$report_file"
    
    log_success "Monitoring report saved: $report_file"
    
    # Show summary
    local dashboard_count policy_count channel_count uptime_count
    dashboard_count=$(echo "$monitoring_data" | jq '.dashboards | length')
    policy_count=$(echo "$monitoring_data" | jq '.alertPolicies | length')
    channel_count=$(echo "$monitoring_data" | jq '.notificationChannels | length')
    uptime_count=$(echo "$monitoring_data" | jq '.uptimeChecks | length')
    
    log_info "Monitoring Summary:"
    log_info "  - Dashboards: $dashboard_count"
    log_info "  - Alert Policies: $policy_count"
    log_info "  - Notification Channels: $channel_count"
    log_info "  - Uptime Checks: $uptime_count"
}

show_monitoring_urls() {
    log_info "Important Monitoring URLs:"
    echo ""
    echo -e "${CYAN}📊 Monitoring Resources:${NC}"
    echo -e "  • Cloud Monitoring: https://console.cloud.google.com/monitoring?project=$PROJECT_ID"
    echo -e "  • Dashboards: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
    echo -e "  • Alert Policies: https://console.cloud.google.com/monitoring/alerting?project=$PROJECT_ID"
    echo -e "  • Uptime Checks: https://console.cloud.google.com/monitoring/uptime?project=$PROJECT_ID"
    echo -e "  • Metrics Explorer: https://console.cloud.google.com/monitoring/metrics-explorer?project=$PROJECT_ID"
    echo ""
    echo -e "${CYAN}🔍 Logging & Debugging:${NC}"
    echo -e "  • Cloud Logging: https://console.cloud.google.com/logs/query?project=$PROJECT_ID"
    echo -e "  • Error Reporting: https://console.cloud.google.com/errors?project=$PROJECT_ID"
    echo -e "  • Cloud Trace: https://console.cloud.google.com/traces?project=$PROJECT_ID"
    echo -e "  • Cloud Profiler: https://console.cloud.google.com/profiler?project=$PROJECT_ID"
    echo ""
    echo -e "${CYAN}🏥 Service Health:${NC}"
    echo -e "  • Cloud Run Services: https://console.cloud.google.com/run?project=$PROJECT_ID"
    echo -e "  • Cloud SQL: https://console.cloud.google.com/sql/instances?project=$PROJECT_ID"
    echo -e "  • Memorystore Redis: https://console.cloud.google.com/memorystore/redis/instances?project=$PROJECT_ID"
    echo ""
    echo -e "${CYAN}💰 Billing & Budgets:${NC}"
    echo -e "  • Billing: https://console.cloud.google.com/billing"
    echo -e "  • Budgets & Alerts: https://console.cloud.google.com/billing/budgets"
    echo ""
}

show_next_steps() {
    echo ""
    echo -e "${YELLOW}📋 Next Steps:${NC}"
    echo ""
    echo -e "1. ${GREEN}Configure DNS Records:${NC}"
    echo -e "   • Point api.agourakis.med.br to the load balancer IP"
    echo -e "   • Point darwin.agourakis.med.br to the load balancer IP"
    echo ""
    echo -e "2. ${GREEN}Wait for SSL Certificates:${NC}"
    echo -e "   • Managed certificates can take 10-60 minutes to provision"
    echo -e "   • Check status: gcloud compute ssl-certificates list"
    echo ""
    echo -e "3. ${GREEN}Test Complete System:${NC}"
    echo -e "   • Visit https://darwin.agourakis.med.br"
    echo -e "   • Test API at https://api.agourakis.med.br/health"
    echo -e "   • Check all monitoring dashboards"
    echo ""
    echo -e "4. ${GREEN}Set Up Notifications:${NC}"
    if [[ -z "$EMAIL_ADDRESSES" ]]; then
        echo -e "   • Configure email alerts: $0 --email 'admin@company.com'"
    fi
    if [[ -z "$SLACK_WEBHOOK" ]]; then
        echo -e "   • Configure Slack alerts: $0 --slack-webhook 'https://hooks.slack.com/...'"
    fi
    echo ""
    echo -e "5. ${GREEN}Monitor and Optimize:${NC}"
    echo -e "   • Review performance metrics weekly"
    echo -e "   • Adjust scaling parameters as needed"
    echo -e "   • Monitor costs and optimize resources"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            --email)
                EMAIL_ADDRESSES="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --pagerduty-key)
                PAGERDUTY_KEY="$2"
                shift 2
                ;;
            --budget)
                BUDGET_AMOUNT="$2"
                shift 2
                ;;
            --verify)
                VERIFY_ONLY="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check for environment variables
    PROJECT_ID="${PROJECT_ID:-${DARWIN_PROJECT_ID:-}}"
    ENVIRONMENT="${ENVIRONMENT:-${DARWIN_ENVIRONMENT:-production}}"
    REGION="${REGION:-${DARWIN_REGION:-us-central1}}"
    EMAIL_ADDRESSES="${EMAIL_ADDRESSES:-${DARWIN_EMAIL_ADDRESSES:-}}"
    SLACK_WEBHOOK="${SLACK_WEBHOOK:-${DARWIN_SLACK_WEBHOOK:-}}"
    PAGERDUTY_KEY="${PAGERDUTY_KEY:-${DARWIN_PAGERDUTY_KEY:-}}"
    BUDGET_AMOUNT="${BUDGET_AMOUNT:-${DARWIN_BUDGET_AMOUNT:-500}}"
    
    # Validate required parameters
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        show_usage
        exit 1
    fi
    
    # Show banner
    show_banner
    
    log_info "Starting DARWIN monitoring setup..."
    log_info "Project: $PROJECT_ID | Environment: $ENVIRONMENT | Region: $REGION"
    
    check_prerequisites
    
    if [[ "$VERIFY_ONLY" == "true" ]]; then
        log_info "Verification mode - checking existing monitoring configuration"
        verify_monitoring_resources
        test_service_connectivity
        check_ssl_certificates
        generate_monitoring_report
        show_monitoring_urls
        show_next_steps
    else
        log_info "Setup mode - configuring monitoring resources"
        verify_monitoring_resources || log_warning "Some monitoring resources may not be configured yet"
        test_service_connectivity
        check_ssl_certificates
        generate_monitoring_report
        show_monitoring_urls
        show_next_steps
    fi
    
    log_success "DARWIN monitoring setup completed!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi