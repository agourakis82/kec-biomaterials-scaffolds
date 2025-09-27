#!/bin/bash

# =============================================================================
# DARWIN Infrastructure Deployment Script
# Script master para orquestração completa do deployment da infraestrutura
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"
CLOUDBUILD_DIR="$PROJECT_ROOT/infrastructure/cloudbuild"

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
BILLING_ACCOUNT_ID=""
DRY_RUN="false"
SKIP_VALIDATION="false"
AUTO_APPROVE="false"
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
    ║                    DARWIN INFRASTRUCTURE                      ║
    ║                    Deployment Orchestrator                    ║
    ║                                                               ║
    ║            Production-Ready GCP Infrastructure                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DARWIN Infrastructure Deployment Script

OPTIONS:
    -p, --project-id PROJECT_ID       GCP Project ID (required)
    -b, --billing-account BILLING_ID  Billing Account ID (required)
    -e, --environment ENVIRONMENT     Environment [dev|staging|production] (default: production)
    -r, --region REGION              GCP Region (default: us-central1)
    -d, --dry-run                    Perform dry run without applying changes
    -s, --skip-validation            Skip pre-deployment validation
    -y, --auto-approve               Auto-approve Terraform changes
    -v, --verbose                    Enable verbose logging
    -h, --help                       Show this help message

EXAMPLES:
    $0 -p my-project -b 123456-789012-345678
    $0 -p my-project -b 123456-789012-345678 -e staging -r us-east1
    $0 -p my-project -b 123456-789012-345678 --dry-run --verbose

ENVIRONMENT VARIABLES:
    DARWIN_PROJECT_ID                Project ID
    DARWIN_BILLING_ACCOUNT_ID        Billing Account ID
    DARWIN_ENVIRONMENT               Environment
    DARWIN_REGION                    GCP Region

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
    command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access project $PROJECT_ID. Please check project ID and permissions."
        exit 1
    fi
    
    # Check billing account
    if [[ -n "$BILLING_ACCOUNT_ID" ]]; then
        if ! gcloud billing accounts describe "$BILLING_ACCOUNT_ID" >/dev/null 2>&1; then
            log_error "Cannot access billing account $BILLING_ACCOUNT_ID"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

validate_configuration() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_warning "Skipping configuration validation"
        return 0
    fi
    
    log_info "Validating configuration..."
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi
    
    # Validate region
    if ! gcloud compute regions list --format="value(name)" | grep -q "^$REGION$"; then
        log_error "Invalid region: $REGION"
        exit 1
    fi
    
    # Check if Terraform files exist
    if [[ ! -f "$TERRAFORM_DIR/main.tf" ]]; then
        log_error "Terraform configuration not found at $TERRAFORM_DIR"
        exit 1
    fi
    
    # Validate Terraform configuration
    log_info "Validating Terraform configuration..."
    cd "$TERRAFORM_DIR"
    terraform fmt -check -recursive || {
        log_warning "Terraform files need formatting. Run: terraform fmt -recursive"
    }
    
    terraform init -backend=false >/dev/null
    terraform validate || {
        log_error "Terraform configuration is invalid"
        exit 1
    }
    
    log_success "Configuration validation completed"
}

enable_required_apis() {
    log_info "Enabling required GCP APIs..."
    
    local apis=(
        "compute.googleapis.com"
        "run.googleapis.com"
        "sql-component.googleapis.com"
        "sqladmin.googleapis.com"
        "redis.googleapis.com"
        "storage.googleapis.com"
        "secretmanager.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudbuild.googleapis.com"
        "containerregistry.googleapis.com"
        "artifactregistry.googleapis.com"
        "certificatemanager.googleapis.com"
        "dns.googleapis.com"
        "containeranalysis.googleapis.com"
        "cloudkms.googleapis.com"
        "secretmanager.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_debug "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" 2>/dev/null || {
            log_warning "Failed to enable $api"
        }
    done
    
    log_success "APIs enabled"
}

setup_terraform_backend() {
    log_info "Setting up Terraform backend..."
    
    local state_bucket="darwin-terraform-state-bucket"
    
    # Create state bucket if it doesn't exist
    if ! gsutil ls -b "gs://$state_bucket" >/dev/null 2>&1; then
        log_info "Creating Terraform state bucket..."
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$state_bucket"
        gsutil versioning set on "gs://$state_bucket"
        
        # Set lifecycle policy
        cat > /tmp/lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 90, "isLive": false}
    }
  ]
}
EOF
        gsutil lifecycle set /tmp/lifecycle.json "gs://$state_bucket"
        rm /tmp/lifecycle.json
    fi
    
    log_success "Terraform backend configured"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure using Cloud Build..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy infrastructure"
        return 0
    fi
    
    # Create build logs bucket if it doesn't exist
    local logs_bucket="${PROJECT_ID}-build-logs"
    if ! gsutil ls -b "gs://$logs_bucket" >/dev/null 2>&1; then
        log_info "Creating build logs bucket..."
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$logs_bucket"
    fi
    
    # Create build artifacts bucket if it doesn't exist
    local artifacts_bucket="${PROJECT_ID}-build-artifacts"
    if ! gsutil ls -b "gs://$artifacts_bucket" >/dev/null 2>&1; then
        log_info "Creating build artifacts bucket..."
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$artifacts_bucket"
    fi
    
    # Submit Cloud Build job
    local build_config="$CLOUDBUILD_DIR/infrastructure-deploy.yaml"
    if [[ ! -f "$build_config" ]]; then
        log_error "Cloud Build configuration not found: $build_config"
        exit 1
    fi
    
    log_info "Submitting Cloud Build job..."
    local substitutions="_PROJECT_ID=$PROJECT_ID,_REGION=$REGION,_ENVIRONMENT=$ENVIRONMENT,_BILLING_ACCOUNT_ID=$BILLING_ACCOUNT_ID"
    
    if [[ "$VERBOSE" == "true" ]]; then
        gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            .
    else
        gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            . >/dev/null &
        local build_pid=$!
        
        # Show progress
        while kill -0 $build_pid 2>/dev/null; do
            echo -n "."
            sleep 5
        done
        echo ""
        wait $build_pid
    fi
    
    log_success "Infrastructure deployment completed"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if main resources exist
    log_debug "Checking VPC network..."
    local vpc_name="darwin-${ENVIRONMENT}-vpc"
    if gcloud compute networks describe "$vpc_name" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "VPC network exists: $vpc_name"
    else
        log_warning "VPC network not found: $vpc_name"
    fi
    
    log_debug "Checking Cloud Run services..."
    local backend_service="darwin-${ENVIRONMENT}-backend"
    local frontend_service="darwin-${ENVIRONMENT}-frontend"
    
    if gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "Backend service exists: $backend_service"
    else
        log_info "Backend service not deployed yet: $backend_service"
    fi
    
    if gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "Frontend service exists: $frontend_service"
    else
        log_info "Frontend service not deployed yet: $frontend_service"
    fi
    
    log_debug "Checking database..."
    local db_instance="darwin-${ENVIRONMENT}-db"
    if gcloud sql instances describe "$db_instance" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "Database instance exists: $db_instance"
    else
        log_warning "Database instance not found: $db_instance"
    fi
    
    log_debug "Checking Redis..."
    local redis_instance="darwin-${ENVIRONMENT}-redis"
    if gcloud redis instances describe "$redis_instance" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "Redis instance exists: $redis_instance"
    else
        log_warning "Redis instance not found: $redis_instance"
    fi
    
    log_success "Deployment verification completed"
}

show_deployment_summary() {
    log_info "Deployment Summary"
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    DEPLOYMENT SUMMARY                        ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Project ID:       $PROJECT_ID"
    echo -e "${CYAN}║${NC} Environment:      $ENVIRONMENT"
    echo -e "${CYAN}║${NC} Region:           $REGION"
    echo -e "${CYAN}║${NC} Billing Account:  $BILLING_ACCOUNT_ID"
    echo -e "${CYAN}║${NC} Terraform Dir:    $TERRAFORM_DIR"
    echo -e "${CYAN}║${NC} Timestamp:        $(date)"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Infrastructure deployed successfully!"
    echo -e "${CYAN}║${NC} "
    echo -e "${CYAN}║${NC} Next steps:"
    echo -e "${CYAN}║${NC} 1. Deploy backend:  ./scripts/deploy_applications.sh --backend"
    echo -e "${CYAN}║${NC} 2. Deploy frontend: ./scripts/deploy_applications.sh --frontend"
    echo -e "${CYAN}║${NC} 3. Configure DNS:   See deployment report for IP addresses"
    echo -e "${CYAN}║${NC} 4. Monitor services: ./scripts/setup_monitoring.sh --verify"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

cleanup_on_error() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Check build logs for details:"
        log_info "https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -b|--billing-account)
                BILLING_ACCOUNT_ID="$2"
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
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -s|--skip-validation)
                SKIP_VALIDATION="true"
                shift
                ;;
            -y|--auto-approve)
                AUTO_APPROVE="true"
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
    
    # Check for environment variables if not provided via CLI
    PROJECT_ID="${PROJECT_ID:-${DARWIN_PROJECT_ID:-}}"
    BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-${DARWIN_BILLING_ACCOUNT_ID:-}}"
    ENVIRONMENT="${ENVIRONMENT:-${DARWIN_ENVIRONMENT:-production}}"
    REGION="${REGION:-${DARWIN_REGION:-us-central1}}"
    
    # Validate required parameters
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        show_usage
        exit 1
    fi
    
    if [[ -z "$BILLING_ACCOUNT_ID" ]]; then
        log_error "Billing Account ID is required"
        show_usage
        exit 1
    fi
    
    # Show banner
    show_banner
    
    # Execute deployment steps
    log_info "Starting DARWIN infrastructure deployment..."
    log_info "Project: $PROJECT_ID | Environment: $ENVIRONMENT | Region: $REGION"
    
    check_prerequisites
    validate_configuration
    enable_required_apis
    setup_terraform_backend
    deploy_infrastructure
    verify_deployment
    show_deployment_summary
    
    log_success "DARWIN infrastructure deployment completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi