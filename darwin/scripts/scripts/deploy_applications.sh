#!/bin/bash

# =============================================================================
# DARWIN Applications Deployment Script
# Script para deployment do backend JAX-powered e frontend React TypeScript
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
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
DEPLOY_BACKEND="false"
DEPLOY_FRONTEND="false"
DEPLOY_BOTH="false"
DRY_RUN="false"
SKIP_TESTS="false"
PARALLEL="false"
VERBOSE="false"
FORCE="false"

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
    ║                    DARWIN APPLICATIONS                        ║
    ║                    Deployment Orchestrator                    ║
    ║                                                               ║
    ║              Backend (JAX) + Frontend (React)                 ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DARWIN Applications Deployment Script

OPTIONS:
    -p, --project-id PROJECT_ID       GCP Project ID (required)
    -e, --environment ENVIRONMENT     Environment [dev|staging|production] (default: production)
    -r, --region REGION              GCP Region (default: us-central1)
    
    Deployment targets:
    --backend                        Deploy backend only
    --frontend                       Deploy frontend only
    --both                          Deploy both backend and frontend
    
    Deployment options:
    -d, --dry-run                    Perform dry run without deploying
    -s, --skip-tests                 Skip tests during deployment
    -P, --parallel                   Deploy backend and frontend in parallel
    -f, --force                      Force deployment even if validation fails
    -v, --verbose                    Enable verbose logging
    -h, --help                       Show this help message

EXAMPLES:
    $0 -p my-project --backend
    $0 -p my-project --frontend
    $0 -p my-project --both --parallel
    $0 -p my-project --both -e staging --verbose
    $0 -p my-project --backend --dry-run

ENVIRONMENT VARIABLES:
    DARWIN_PROJECT_ID                Project ID
    DARWIN_ENVIRONMENT               Environment
    DARWIN_REGION                    GCP Region

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
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
    
    # Check if infrastructure exists
    log_debug "Checking if infrastructure exists..."
    local vpc_name="darwin-${ENVIRONMENT}-vpc"
    if ! gcloud compute networks describe "$vpc_name" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Infrastructure not found. Please deploy infrastructure first:"
        log_error "  ./scripts/deploy_infrastructure.sh -p $PROJECT_ID"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

validate_environment() {
    log_info "Validating environment configuration..."
    
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
    
    # Check if Cloud Build configs exist
    local backend_config="$CLOUDBUILD_DIR/backend-deploy.yaml"
    local frontend_config="$CLOUDBUILD_DIR/frontend-deploy.yaml"
    
    if [[ "$DEPLOY_BACKEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        if [[ ! -f "$backend_config" ]]; then
            log_error "Backend deployment config not found: $backend_config"
            exit 1
        fi
    fi
    
    if [[ "$DEPLOY_FRONTEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        if [[ ! -f "$frontend_config" ]]; then
            log_error "Frontend deployment config not found: $frontend_config"
            exit 1
        fi
    fi
    
    log_success "Environment validation completed"
}

check_infrastructure_health() {
    log_info "Checking infrastructure health..."
    
    local issues=()
    
    # Check database
    local db_instance="darwin-${ENVIRONMENT}-db"
    local db_status
    db_status=$(gcloud sql instances describe "$db_instance" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$db_status" != "RUNNABLE" ]]; then
        issues+=("Database instance $db_instance is not ready (status: $db_status)")
    fi
    
    # Check Redis
    local redis_instance="darwin-${ENVIRONMENT}-redis"
    local redis_status
    redis_status=$(gcloud redis instances describe "$redis_instance" --region="$REGION" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$redis_status" != "READY" ]]; then
        issues+=("Redis instance $redis_instance is not ready (status: $redis_status)")
    fi
    
    # Check VPC connector
    local vpc_connector="darwin-${ENVIRONMENT}-connector"
    local connector_status
    connector_status=$(gcloud compute networks vpc-access connectors describe "$vpc_connector" --region="$REGION" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$connector_status" != "READY" ]]; then
        issues+=("VPC connector $vpc_connector is not ready (status: $connector_status)")
    fi
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        if [[ "$FORCE" == "true" ]]; then
            log_warning "Infrastructure issues detected but continuing due to --force flag:"
            for issue in "${issues[@]}"; do
                log_warning "  - $issue"
            done
        else
            log_error "Infrastructure health check failed:"
            for issue in "${issues[@]}"; do
                log_error "  - $issue"
            done
            log_error "Use --force to deploy anyway or fix infrastructure issues first"
            exit 1
        fi
    else
        log_success "Infrastructure health check passed"
    fi
}

deploy_backend() {
    log_info "Deploying backend application..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy backend"
        return 0
    fi
    
    local build_config="$CLOUDBUILD_DIR/backend-deploy.yaml"
    local substitutions="_PROJECT_ID=$PROJECT_ID,_REGION=$REGION,_ENVIRONMENT=$ENVIRONMENT"
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        substitutions+=",_SKIP_TESTS=true"
    fi
    
    log_info "Submitting backend deployment job..."
    log_debug "Config: $build_config"
    log_debug "Substitutions: $substitutions"
    
    if [[ "$VERBOSE" == "true" ]]; then
        gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            .
    else
        local build_id
        build_id=$(gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            --format="value(id)" \
            . 2>/dev/null)
        
        log_info "Backend build started: $build_id"
        log_info "Monitor at: https://console.cloud.google.com/cloud-build/builds/$build_id?project=$PROJECT_ID"
        
        # Monitor build progress
        local status="WORKING"
        while [[ "$status" == "WORKING" || "$status" == "QUEUED" ]]; do
            sleep 30
            status=$(gcloud builds describe "$build_id" --project="$PROJECT_ID" --format="value(status)" 2>/dev/null || echo "UNKNOWN")
            echo -n "."
        done
        echo ""
        
        if [[ "$status" == "SUCCESS" ]]; then
            log_success "Backend deployment completed successfully"
        else
            log_error "Backend deployment failed with status: $status"
            exit 1
        fi
    fi
}

deploy_frontend() {
    log_info "Deploying frontend application..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy frontend"
        return 0
    fi
    
    local build_config="$CLOUDBUILD_DIR/frontend-deploy.yaml"
    local substitutions="_PROJECT_ID=$PROJECT_ID,_REGION=$REGION,_ENVIRONMENT=$ENVIRONMENT"
    
    # Get API URL for frontend configuration
    local api_url="https://api.agourakis.med.br"
    local frontend_url="https://darwin.agourakis.med.br"
    
    substitutions+=",_API_URL=$api_url,_FRONTEND_URL=$frontend_url"
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        substitutions+=",_SKIP_TESTS=true"
    fi
    
    log_info "Submitting frontend deployment job..."
    log_debug "Config: $build_config"
    log_debug "Substitutions: $substitutions"
    
    if [[ "$VERBOSE" == "true" ]]; then
        gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            .
    else
        local build_id
        build_id=$(gcloud builds submit \
            --config="$build_config" \
            --substitutions="$substitutions" \
            --project="$PROJECT_ID" \
            --format="value(id)" \
            . 2>/dev/null)
        
        log_info "Frontend build started: $build_id"
        log_info "Monitor at: https://console.cloud.google.com/cloud-build/builds/$build_id?project=$PROJECT_ID"
        
        # Monitor build progress
        local status="WORKING"
        while [[ "$status" == "WORKING" || "$status" == "QUEUED" ]]; do
            sleep 30
            status=$(gcloud builds describe "$build_id" --project="$PROJECT_ID" --format="value(status)" 2>/dev/null || echo "UNKNOWN")
            echo -n "."
        done
        echo ""
        
        if [[ "$status" == "SUCCESS" ]]; then
            log_success "Frontend deployment completed successfully"
        else
            log_error "Frontend deployment failed with status: $status"
            exit 1
        fi
    fi
}

deploy_parallel() {
    log_info "Deploying backend and frontend in parallel..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy backend and frontend in parallel"
        return 0
    fi
    
    # Start backend deployment in background
    (
        log_info "[BACKEND] Starting backend deployment..."
        deploy_backend
        echo "BACKEND_DONE" > /tmp/backend_status
    ) &
    local backend_pid=$!
    
    # Start frontend deployment in background
    (
        log_info "[FRONTEND] Starting frontend deployment..."
        deploy_frontend
        echo "FRONTEND_DONE" > /tmp/frontend_status
    ) &
    local frontend_pid=$!
    
    # Wait for both to complete
    log_info "Waiting for deployments to complete..."
    
    local backend_success=false
    local frontend_success=false
    
    # Wait for backend
    if wait $backend_pid; then
        backend_success=true
        log_success "[BACKEND] Deployment completed"
    else
        log_error "[BACKEND] Deployment failed"
    fi
    
    # Wait for frontend
    if wait $frontend_pid; then
        frontend_success=true
        log_success "[FRONTEND] Deployment completed"
    else
        log_error "[FRONTEND] Deployment failed"
    fi
    
    # Clean up temporary files
    rm -f /tmp/backend_status /tmp/frontend_status
    
    if [[ "$backend_success" == "true" && "$frontend_success" == "true" ]]; then
        log_success "Both deployments completed successfully"
    else
        log_error "One or more deployments failed"
        exit 1
    fi
}

verify_deployments() {
    log_info "Verifying deployments..."
    
    local issues=()
    
    if [[ "$DEPLOY_BACKEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        log_debug "Verifying backend deployment..."
        local backend_service="darwin-${ENVIRONMENT}-backend"
        
        if gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
            local backend_url
            backend_url=$(gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
            
            log_debug "Testing backend health endpoint..."
            if curl -f -s -m 30 "$backend_url/health" >/dev/null; then
                log_success "Backend is healthy: $backend_url"
            else
                issues+=("Backend health check failed")
            fi
        else
            issues+=("Backend service not found")
        fi
    fi
    
    if [[ "$DEPLOY_FRONTEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        log_debug "Verifying frontend deployment..."
        local frontend_service="darwin-${ENVIRONMENT}-frontend"
        
        if gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
            local frontend_url
            frontend_url=$(gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
            
            log_debug "Testing frontend endpoint..."
            if curl -f -s -m 30 "$frontend_url/" >/dev/null; then
                log_success "Frontend is healthy: $frontend_url"
            else
                issues+=("Frontend health check failed")
            fi
        else
            issues+=("Frontend service not found")
        fi
    fi
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_warning "Verification issues detected:"
        for issue in "${issues[@]}"; do
            log_warning "  - $issue"
        done
        log_warning "Services may still be starting up. Check again in a few minutes."
    else
        log_success "All deployments verified successfully"
    fi
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
    echo -e "${CYAN}║${NC} Backend:          $([ "$DEPLOY_BACKEND" == "true" ] || [ "$DEPLOY_BOTH" == "true" ] && echo "✅ Deployed" || echo "⏭️ Skipped")"
    echo -e "${CYAN}║${NC} Frontend:         $([ "$DEPLOY_FRONTEND" == "true" ] || [ "$DEPLOY_BOTH" == "true" ] && echo "✅ Deployed" || echo "⏭️ Skipped")"
    echo -e "${CYAN}║${NC} Parallel:         $([ "$PARALLEL" == "true" ] && echo "Yes" || echo "No")"
    echo -e "${CYAN}║${NC} Timestamp:        $(date)"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    
    # Show service URLs if deployed
    if [[ "$DEPLOY_BACKEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        local backend_service="darwin-${ENVIRONMENT}-backend"
        if gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
            local backend_url
            backend_url=$(gcloud run services describe "$backend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
            echo -e "${CYAN}║${NC} Backend URL:      $backend_url"
        fi
    fi
    
    if [[ "$DEPLOY_FRONTEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
        local frontend_service="darwin-${ENVIRONMENT}-frontend"
        if gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
            local frontend_url
            frontend_url=$(gcloud run services describe "$frontend_service" --region="$REGION" --project="$PROJECT_ID" --format="value(status.url)")
            echo -e "${CYAN}║${NC} Frontend URL:     $frontend_url"
        fi
    fi
    
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Applications deployed successfully!"
    echo -e "${CYAN}║${NC} "
    echo -e "${CYAN}║${NC} Next steps:"
    echo -e "${CYAN}║${NC} 1. Configure DNS: Point domains to load balancer IP"
    echo -e "${CYAN}║${NC} 2. Wait for SSL:  Certificates may take 10-60 minutes"
    echo -e "${CYAN}║${NC} 3. Test services: Visit the URLs above"
    echo -e "${CYAN}║${NC} 4. Monitor:       ./scripts/setup_monitoring.sh --verify"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

cleanup_on_error() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Check build logs for details:"
        log_info "https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
        
        # Clean up any temporary files
        rm -f /tmp/backend_status /tmp/frontend_status
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
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            --backend)
                DEPLOY_BACKEND="true"
                shift
                ;;
            --frontend)
                DEPLOY_FRONTEND="true"
                shift
                ;;
            --both)
                DEPLOY_BOTH="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -P|--parallel)
                PARALLEL="true"
                shift
                ;;
            -f|--force)
                FORCE="true"
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
    ENVIRONMENT="${ENVIRONMENT:-${DARWIN_ENVIRONMENT:-production}}"
    REGION="${REGION:-${DARWIN_REGION:-us-central1}}"
    
    # Validate required parameters
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        show_usage
        exit 1
    fi
    
    # Validate deployment targets
    if [[ "$DEPLOY_BACKEND" == "false" && "$DEPLOY_FRONTEND" == "false" && "$DEPLOY_BOTH" == "false" ]]; then
        log_error "Must specify deployment target: --backend, --frontend, or --both"
        show_usage
        exit 1
    fi
    
    # Show banner
    show_banner
    
    # Execute deployment steps
    log_info "Starting DARWIN applications deployment..."
    log_info "Project: $PROJECT_ID | Environment: $ENVIRONMENT | Region: $REGION"
    
    check_prerequisites
    validate_environment
    check_infrastructure_health
    
    # Execute deployments based on options
    if [[ "$DEPLOY_BOTH" == "true" && "$PARALLEL" == "true" ]]; then
        deploy_parallel
    else
        if [[ "$DEPLOY_BACKEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
            deploy_backend
        fi
        
        if [[ "$DEPLOY_FRONTEND" == "true" || "$DEPLOY_BOTH" == "true" ]]; then
            deploy_frontend
        fi
    fi
    
    verify_deployments
    show_deployment_summary
    
    log_success "DARWIN applications deployment completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi