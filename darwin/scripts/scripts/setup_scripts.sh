#!/bin/bash

# =============================================================================
# DARWIN Scripts Setup
# Torna todos os scripts executáveis e configura permissões
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
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

show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                      DARWIN SCRIPTS                           ║
    ║                     Setup & Overview                          ║
    ║                                                               ║
    ║               Production-Ready Automation                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

setup_permissions() {
    log_info "Setting up script permissions..."
    
    # Make all shell scripts executable
    find "$SCRIPT_DIR" -name "*.sh" -type f -exec chmod +x {} \;
    
    # List all scripts
    log_success "Made the following scripts executable:"
    find "$SCRIPT_DIR" -name "*.sh" -type f | while read -r script; do
        local script_name
        script_name=$(basename "$script")
        echo -e "  ${GREEN}✓${NC} $script_name"
    done
}

show_deployment_overview() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    DEPLOYMENT OVERVIEW                       ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}📋 Deployment Scripts Available:${NC}"
    echo ""
    
    echo -e "${GREEN}1. Infrastructure Deployment:${NC}"
    echo -e "   ${BLUE}./scripts/deploy_infrastructure.sh${NC}"
    echo -e "   • Deploys complete Terraform infrastructure"
    echo -e "   • Sets up VPC, database, Redis, monitoring"
    echo -e "   • Configures SSL certificates and load balancer"
    echo -e "   • Example: ./scripts/deploy_infrastructure.sh -p my-project -b billing-account-id"
    echo ""
    
    echo -e "${GREEN}2. Applications Deployment:${NC}"
    echo -e "   ${BLUE}./scripts/deploy_applications.sh${NC}"
    echo -e "   • Deploys backend (JAX-powered API) and/or frontend (React TypeScript)"
    echo -e "   • Supports parallel deployment"
    echo -e "   • Includes health checks and validation"
    echo -e "   • Example: ./scripts/deploy_applications.sh -p my-project --both --parallel"
    echo ""
    
    echo -e "${GREEN}3. Monitoring Setup:${NC}"
    echo -e "   ${BLUE}./scripts/setup_monitoring.sh${NC}"
    echo -e "   • Verifies monitoring configuration"
    echo -e "   • Tests service connectivity"
    echo -e "   • Checks SSL certificate status"
    echo -e "   • Example: ./scripts/setup_monitoring.sh -p my-project --verify"
    echo ""
    
    echo -e "${GREEN}4. Legacy Scripts (from previous phases):${NC}"
    echo -e "   • ${BLUE}./scripts/gcp_inventory_analysis.sh${NC} - Analyze existing GCP resources"
    echo -e "   • ${BLUE}./scripts/gcp_backup_critical_data.sh${NC} - Backup critical data"
    echo -e "   • ${BLUE}./scripts/gcp_cleanup_legacy.sh${NC} - Clean up legacy resources"
    echo -e "   • ${BLUE}./scripts/deploy_darwin_production_optimized.sh${NC} - Legacy deployment script"
    echo ""
}

show_deployment_sequence() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   DEPLOYMENT SEQUENCE                        ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}🚀 Complete Deployment Sequence:${NC}"
    echo ""
    
    echo -e "${GREEN}Step 1: Deploy Infrastructure${NC}"
    echo -e "   ${BLUE}./scripts/deploy_infrastructure.sh -p YOUR_PROJECT_ID -b YOUR_BILLING_ACCOUNT${NC}"
    echo -e "   ⏱️  Duration: ~15-20 minutes"
    echo -e "   📋 Creates: VPC, database, Redis, storage, monitoring, SSL certificates"
    echo ""
    
    echo -e "${GREEN}Step 2: Deploy Applications${NC}"
    echo -e "   ${BLUE}./scripts/deploy_applications.sh -p YOUR_PROJECT_ID --both --parallel${NC}"
    echo -e "   ⏱️  Duration: ~10-15 minutes"
    echo -e "   📋 Creates: Backend Cloud Run service, Frontend Cloud Run service"
    echo ""
    
    echo -e "${GREEN}Step 3: Configure DNS${NC}"
    echo -e "   • Point api.agourakis.med.br to the load balancer IP"
    echo -e "   • Point darwin.agourakis.med.br to the load balancer IP"
    echo -e "   ⏱️  Duration: DNS propagation ~5-30 minutes"
    echo ""
    
    echo -e "${GREEN}Step 4: Wait for SSL Certificates${NC}"
    echo -e "   • Managed certificates provision automatically"
    echo -e "   • Monitor with: gcloud compute ssl-certificates list"
    echo -e "   ⏱️  Duration: SSL provisioning ~10-60 minutes"
    echo ""
    
    echo -e "${GREEN}Step 5: Verify Deployment${NC}"
    echo -e "   ${BLUE}./scripts/setup_monitoring.sh -p YOUR_PROJECT_ID --verify${NC}"
    echo -e "   ⏱️  Duration: ~2-3 minutes"
    echo -e "   📋 Verifies: All services, monitoring, SSL status"
    echo ""
    
    echo -e "${YELLOW}📊 Total Deployment Time: ~45-90 minutes${NC}"
    echo -e "${YELLOW}💰 Expected Monthly Cost: ~$300-500 USD${NC}"
    echo ""
}

show_urls_and_endpoints() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  URLS & ENDPOINTS                             ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}🌐 Production URLs (after DNS configuration):${NC}"
    echo -e "   • Frontend: ${GREEN}https://darwin.agourakis.med.br${NC}"
    echo -e "   • API: ${GREEN}https://api.agourakis.med.br${NC}"
    echo -e "   • API Health: ${GREEN}https://api.agourakis.med.br/health${NC}"
    echo -e "   • API Docs: ${GREEN}https://api.agourakis.med.br/docs${NC}"
    echo -e "   • API Metrics: ${GREEN}https://api.agourakis.med.br/metrics${NC}"
    echo ""
    
    echo -e "${YELLOW}📊 GCP Console URLs:${NC}"
    echo -e "   • Cloud Run: https://console.cloud.google.com/run"
    echo -e "   • Cloud SQL: https://console.cloud.google.com/sql"
    echo -e "   • Monitoring: https://console.cloud.google.com/monitoring"
    echo -e "   • Cloud Build: https://console.cloud.google.com/cloud-build"
    echo -e "   • Load Balancing: https://console.cloud.google.com/net-services/loadbalancing"
    echo ""
}

show_troubleshooting() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   TROUBLESHOOTING                            ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}🔧 Common Issues & Solutions:${NC}"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}Script permission denied"
    echo -e "${GREEN}✅ Solution: ${NC}Run ./scripts/setup_scripts.sh to fix permissions"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}gcloud not authenticated"
    echo -e "${GREEN}✅ Solution: ${NC}Run gcloud auth login"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}Project access denied"
    echo -e "${GREEN}✅ Solution: ${NC}Check project ID and run gcloud config set project PROJECT_ID"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}Terraform state bucket access denied"
    echo -e "${GREEN}✅ Solution: ${NC}Check billing account permissions and enable APIs"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}Domain not accessible"
    echo -e "${GREEN}✅ Solution: ${NC}Check DNS configuration and wait for SSL certificate provisioning"
    echo ""
    
    echo -e "${RED}❌ Issue: ${NC}Service health check failed"
    echo -e "${GREEN}✅ Solution: ${NC}Check Cloud Run logs and database connectivity"
    echo ""
    
    echo -e "${YELLOW}📞 Get Help:${NC}"
    echo -e "   • Check build logs: https://console.cloud.google.com/cloud-build"
    echo -e "   • View service logs: https://console.cloud.google.com/logs"
    echo -e "   • Monitor resources: https://console.cloud.google.com/monitoring"
    echo ""
}

show_environment_variables() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                 ENVIRONMENT VARIABLES                        ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}🔧 Optional Environment Variables:${NC}"
    echo -e "   Export these to avoid passing parameters repeatedly:"
    echo ""
    
    echo -e "${GREEN}# Core Configuration${NC}"
    echo -e "export DARWIN_PROJECT_ID=\"your-project-id\""
    echo -e "export DARWIN_BILLING_ACCOUNT_ID=\"your-billing-account-id\""
    echo -e "export DARWIN_ENVIRONMENT=\"production\""
    echo -e "export DARWIN_REGION=\"us-central1\""
    echo ""
    
    echo -e "${GREEN}# Monitoring Configuration${NC}"
    echo -e "export DARWIN_EMAIL_ADDRESSES=\"admin@company.com,ops@company.com\""
    echo -e "export DARWIN_SLACK_WEBHOOK=\"https://hooks.slack.com/services/...\""
    echo -e "export DARWIN_PAGERDUTY_KEY=\"your-pagerduty-integration-key\""
    echo -e "export DARWIN_BUDGET_AMOUNT=\"500\""
    echo ""
    
    echo -e "${YELLOW}💡 Usage with environment variables:${NC}"
    echo -e "   ./scripts/deploy_infrastructure.sh  # Uses DARWIN_PROJECT_ID, etc."
    echo -e "   ./scripts/deploy_applications.sh --both"
    echo -e "   ./scripts/setup_monitoring.sh --verify"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    show_banner
    
    log_info "Setting up DARWIN deployment scripts..."
    
    setup_permissions
    
    show_deployment_overview
    show_deployment_sequence
    show_urls_and_endpoints
    show_environment_variables
    show_troubleshooting
    
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║  🎉 DARWIN Scripts Setup Complete!                           ║"
    echo "║                                                               ║"
    echo "║  Ready to deploy production-ready infrastructure:             ║"
    echo "║  • JAX-powered backend with vector database                   ║"
    echo "║  • React TypeScript frontend with PWA support                ║"
    echo "║  • Comprehensive monitoring and alerting                     ║"
    echo "║  • Auto-scaling Cloud Run services                          ║"
    echo "║  • Managed SSL certificates and CDN                         ║"
    echo "║                                                               ║"
    echo "║  Start with: ./scripts/deploy_infrastructure.sh              ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log_success "All scripts are ready for deployment!"
}

# Execute main function
main "$@"