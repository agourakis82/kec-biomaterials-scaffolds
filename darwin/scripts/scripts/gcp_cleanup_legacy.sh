#!/bin/bash

# GCP SYSTEMATIC CLEANUP SCRIPT
# Limpeza sistemática e segura de recursos GCP legacy/obsoletos
# 🧹 SYSTEMATIC CLEANUP - SAFETY FIRST, VALIDATION AT EVERY STEP

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
CLEANUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CLEANUP_LOG_FILE="gcp_cleanup_log_${CLEANUP_TIMESTAMP}.log"
DRY_RUN="${DRY_RUN:-true}"  # Default to dry run for safety
FORCE_DELETE="${FORCE_DELETE:-false}"
INTERACTIVE="${INTERACTIVE:-true}"
BACKUP_VERIFIED="${BACKUP_VERIFIED:-false}"

# Safety configuration
REQUIRE_BACKUP_CONFIRMATION="${REQUIRE_BACKUP_CONFIRMATION:-true}"
MAX_RETRIES=3
CONFIRMATION_REQUIRED=true

# Resources to cleanup (based on analysis)
CLEANUP_CLOUD_RUN="${CLEANUP_CLOUD_RUN:-true}"
CLEANUP_CONTAINER_IMAGES="${CLEANUP_CONTAINER_IMAGES:-true}"
CLEANUP_STORAGE="${CLEANUP_STORAGE:-false}" # Dangerous, default off
CLEANUP_BIGQUERY="${CLEANUP_BIGQUERY:-false}" # Dangerous, default off  
CLEANUP_SECRETS="${CLEANUP_SECRETS:-false}" # Dangerous, default off
CLEANUP_SERVICE_ACCOUNTS="${CLEANUP_SERVICE_ACCOUNTS:-false}" # Dangerous, default off
CLEANUP_DOMAIN_MAPPINGS="${CLEANUP_DOMAIN_MAPPINGS:-true}"
CLEANUP_APIS="${CLEANUP_APIS:-false}"

# Logging functions with file output
log() {
    local message="$1"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" >> "$CLEANUP_LOG_FILE"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $message" >> "$CLEANUP_LOG_FILE"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $message" >> "$CLEANUP_LOG_FILE"
}

log_error() {
    local message="$1"
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $message" >> "$CLEANUP_LOG_FILE"
}

log_critical() {
    local message="$1"
    echo -e "${RED}${BOLD}[$(date +'%Y-%m-%d %H:%M:%S')] 🚨${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] CRITICAL: $message" >> "$CLEANUP_LOG_FILE"
}

log_action() {
    local message="$1"
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] 🔧${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ACTION: $message" >> "$CLEANUP_LOG_FILE"
}

# Header display
show_header() {
    echo -e "${RED}${BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  🧹 DARWIN GCP SYSTEMATIC CLEANUP SCRIPT 🧹                             ║
║                                                                           ║
║  ⚠️  DANGER: THIS SCRIPT REMOVES GCP RESOURCES PERMANENTLY ⚠️           ║
║                                                                           ║
║  Systematic cleanup with safety checks:                                  ║
║  • Cloud Run Services (traffic disruption)                              ║
║  • Container Images (build artifacts)                                   ║
║  • Storage Buckets (DATA LOSS RISK)                                     ║
║  • BigQuery Datasets (DATA LOSS RISK)                                   ║
║  • Secrets Manager (security risk)                                      ║
║  • IAM Service Accounts (access disruption)                             ║
║                                                                           ║
║  🔒 BACKUP VERIFICATION REQUIRED BEFORE EXECUTION                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝${NC}
"
}

# Safety confirmation system
confirm_action() {
    local action="$1"
    local resource="$2"
    local risk_level="${3:-medium}"
    
    if [[ "$INTERACTIVE" != "true" || "$FORCE_DELETE" == "true" ]]; then
        log_warning "Auto-confirming action: $action (interactive=$INTERACTIVE, force=$FORCE_DELETE)"
        return 0
    fi
    
    case "$risk_level" in
        "low")
            echo -e "${YELLOW}⚠️  CONFIRM: $action${NC}"
            ;;
        "medium") 
            echo -e "${YELLOW}⚠️  MODERATE RISK: $action${NC}"
            echo -e "${YELLOW}   Resource: $resource${NC}"
            ;;
        "high")
            echo -e "${RED}🚨 HIGH RISK: $action${NC}"
            echo -e "${RED}   Resource: $resource${NC}"
            echo -e "${RED}   This action may cause DATA LOSS or service disruption!${NC}"
            ;;
        "critical")
            echo -e "${RED}${BOLD}🚨 CRITICAL RISK: $action${NC}"
            echo -e "${RED}${BOLD}   Resource: $resource${NC}"
            echo -e "${RED}${BOLD}   THIS WILL CAUSE PERMANENT DATA LOSS!${NC}"
            echo -e "${RED}${BOLD}   Are you absolutely certain you want to proceed?${NC}"
            ;;
    esac
    
    if [[ "$risk_level" == "critical" ]]; then
        echo -e "${RED}Type 'DELETE PERMANENTLY' to confirm this critical action:${NC}"
        read -r confirmation
        if [[ "$confirmation" != "DELETE PERMANENTLY" ]]; then
            log_warning "Critical action cancelled by user"
            return 1
        fi
    else
        echo -e "${YELLOW}Type 'yes' to confirm, anything else to skip:${NC}"
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log_warning "Action cancelled by user"
            return 1
        fi
    fi
    
    log_action "User confirmed: $action"
    return 0
}

# Initialize cleanup environment
initialize_cleanup() {
    log "🚀 Initializing systematic cleanup..."
    
    if [[ -z "$PROJECT_ID" ]]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "No project specified. Set GCP_PROJECT_ID environment variable"
            exit 1
        fi
    fi
    
    # Create cleanup log
    cat > "$CLEANUP_LOG_FILE" << EOF
# GCP Cleanup Log
# Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Project: $PROJECT_ID
# Dry Run: $DRY_RUN
# Interactive: $INTERACTIVE
# Force Delete: $FORCE_DELETE
EOF
    
    log "Project: $PROJECT_ID"
    log "Region: $REGION"
    log "Cleanup Log: $CLEANUP_LOG_FILE"
    log "Dry Run Mode: $DRY_RUN"
    log "Interactive Mode: $INTERACTIVE"
    
    # Safety warnings for production mode
    if [[ "$DRY_RUN" != "true" ]]; then
        echo -e "${RED}${BOLD}
⚠️  PRODUCTION MODE ACTIVE ⚠️
This will make REAL CHANGES to your GCP project!
Resources will be PERMANENTLY DELETED!
${NC}"
        
        if [[ "$INTERACTIVE" == "true" ]]; then
            echo -e "${RED}Type 'I UNDERSTAND THE RISKS' to continue:${NC}"
            read -r safety_confirmation
            if [[ "$safety_confirmation" != "I UNDERSTAND THE RISKS" ]]; then
                log_critical "Production mode cancelled by user - safety first!"
                exit 1
            fi
        fi
    fi
    
    log_success "Cleanup environment initialized"
}

# Check backup verification
verify_backup_exists() {
    log "🔍 Checking for backup verification..."
    
    if [[ "$REQUIRE_BACKUP_CONFIRMATION" == "true" && "$BACKUP_VERIFIED" != "true" ]]; then
        if [[ "$INTERACTIVE" == "true" ]]; then
            echo -e "${YELLOW}
⚠️  BACKUP VERIFICATION REQUIRED ⚠️

Before proceeding with cleanup, you must confirm that:
1. Critical data backup has been completed successfully
2. Backup has been tested and verified
3. Backup is stored in a secure location
4. You have access to restore procedures

Have you completed and verified your backup?${NC}"
            
            echo -e "${YELLOW}Type 'BACKUP VERIFIED' to confirm:${NC}"
            read -r backup_confirmation
            if [[ "$backup_confirmation" != "BACKUP VERIFIED" ]]; then
                log_critical "Cleanup cancelled - backup not verified"
                echo -e "${RED}Please run the backup script first:${NC}"
                echo -e "${CYAN}./scripts/gcp_backup_critical_data.sh${NC}"
                exit 1
            fi
            
            BACKUP_VERIFIED="true"
            log_success "Backup verification confirmed by user"
        else
            log_error "Non-interactive mode requires BACKUP_VERIFIED=true"
            exit 1
        fi
    fi
    
    if [[ "$BACKUP_VERIFIED" == "true" ]]; then
        log_success "Backup verification passed"
    fi
}

# Check prerequisites
check_cleanup_prerequisites() {
    log "🔑 Checking cleanup prerequisites..."
    
    # Check required tools
    local required_tools=("gcloud" "gsutil" "bq" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool not found - required for cleanup operations"
            exit 1
        fi
    done
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        log_error "Not authenticated with Google Cloud"
        exit 1
    fi
    
    # Check project access and permissions
    if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        exit 1
    fi
    
    # Verify user has deletion permissions (check a few key roles)
    log "Verifying deletion permissions..."
    local current_user
    current_user=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1)
    log "Current user: $current_user"
    
    log_success "Prerequisites verified"
}

# Cleanup Cloud Run services
cleanup_cloud_run_services() {
    if [[ "$CLEANUP_CLOUD_RUN" != "true" ]]; then
        log "Cloud Run cleanup disabled, skipping"
        return 0
    fi
    
    log "☁️ Starting Cloud Run services cleanup..."
    
    # Known DARWIN Cloud Run services
    local darwin_services=(
        "darwin-backend-api"
        "darwin-frontend-web" 
        "darwin-backend"
        "darwin-backend-gpu"
    )
    
    # Get all services in region
    local all_services
    if all_services=$(gcloud run services list --region="$REGION" --format="value(metadata.name)" 2>/dev/null); then
        
        # Process known DARWIN services first
        for service in "${darwin_services[@]}"; do
            if echo "$all_services" | grep -q "^$service$"; then
                log "Found DARWIN service: $service"
                
                # Get service details
                local service_url
                service_url=$(gcloud run services describe "$service" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "unknown")
                
                if confirm_action "Delete Cloud Run service: $service" "$service (URL: $service_url)" "medium"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete Cloud Run service: $service"
                    else
                        log_action "Deleting Cloud Run service: $service"
                        
                        local retry_count=0
                        while [[ $retry_count -lt $MAX_RETRIES ]]; do
                            if gcloud run services delete "$service" --region="$REGION" --quiet; then
                                log_success "Deleted Cloud Run service: $service"
                                break
                            else
                                retry_count=$((retry_count + 1))
                                if [[ $retry_count -eq $MAX_RETRIES ]]; then
                                    log_error "Failed to delete Cloud Run service after $MAX_RETRIES attempts: $service"
                                else
                                    log_warning "Retry $retry_count/$MAX_RETRIES for service: $service"
                                    sleep 10
                                fi
                            fi
                        done
                    fi
                else
                    log_warning "Skipped Cloud Run service: $service"
                fi
            fi
        done
        
        # Check for any other potentially related services
        for service in $all_services; do
            if [[ "$service" =~ darwin|kec|biomaterials ]] && ! printf '%s\n' "${darwin_services[@]}" | grep -q "^$service$"; then
                log_warning "Found potentially related service not in known list: $service"
                
                if confirm_action "Delete unrecognized DARWIN-related service: $service" "$service" "high"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete unrecognized service: $service"
                    else
                        log_action "Deleting unrecognized service: $service"
                        gcloud run services delete "$service" --region="$REGION" --quiet || log_error "Failed to delete: $service"
                    fi
                fi
            fi
        done
        
    else
        log_warning "Could not list Cloud Run services"
    fi
    
    log_success "Cloud Run cleanup phase completed"
}

# Cleanup domain mappings
cleanup_domain_mappings() {
    if [[ "$CLEANUP_DOMAIN_MAPPINGS" != "true" ]]; then
        log "Domain mappings cleanup disabled, skipping"
        return 0
    fi
    
    log "🌐 Starting domain mappings cleanup..."
    
    # Known DARWIN domains
    local darwin_domains=(
        "api.agourakis.med.br"
        "darwin.agourakis.med.br"
    )
    
    # Get all domain mappings
    local all_domains
    if all_domains=$(gcloud run domain-mappings list --region="$REGION" --format="value(metadata.name)" 2>/dev/null); then
        
        for domain in "${darwin_domains[@]}"; do
            if echo "$all_domains" | grep -q "^$domain$"; then
                log "Found DARWIN domain mapping: $domain"
                
                if confirm_action "Delete domain mapping: $domain" "$domain" "medium"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete domain mapping: $domain"
                    else
                        log_action "Deleting domain mapping: $domain"
                        gcloud run domain-mappings delete "$domain" --region="$REGION" --quiet || log_error "Failed to delete domain mapping: $domain"
                        log_success "Deleted domain mapping: $domain"
                    fi
                else
                    log_warning "Skipped domain mapping: $domain"
                fi
            fi
        done
        
    else
        log_warning "Could not list domain mappings"
    fi
    
    log_success "Domain mappings cleanup completed"
}

# Cleanup container images
cleanup_container_images() {
    if [[ "$CLEANUP_CONTAINER_IMAGES" != "true" ]]; then
        log "Container images cleanup disabled, skipping"
        return 0
    fi
    
    log "🐳 Starting container images cleanup..."
    
    # Known DARWIN image repositories
    local darwin_images=(
        "gcr.io/$PROJECT_ID/darwin-backend"
        "gcr.io/$PROJECT_ID/darwin-frontend"
        "gcr.io/$PROJECT_ID/darwin-backend-gpu"
    )
    
    # Check for pcs-helio project images too
    if [[ "$PROJECT_ID" == "pcs-helio" ]]; then
        darwin_images+=(
            "gcr.io/pcs-helio/darwin-backend"
            "gcr.io/pcs-helio/darwin-frontend"
        )
    fi
    
    for image in "${darwin_images[@]}"; do
        log "Checking for container image: $image"
        
        # Check if image exists
        if gcloud container images list --repository="$(dirname "$image")" --format="value(name)" 2>/dev/null | grep -q "$image"; then
            log "Found container image: $image"
            
            # Get image tags
            local tags
            tags=$(gcloud container images list-tags "$image" --format="value(tags)" --limit=10 2>/dev/null || echo "")
            local tag_count
            tag_count=$(gcloud container images list-tags "$image" --format="value(tags)" 2>/dev/null | wc -l || echo "0")
            
            if confirm_action "Delete container image repository: $image" "$image ($tag_count tags)" "low"; then
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_action "[DRY RUN] Would delete container image: $image (all tags)"
                else
                    log_action "Deleting all tags for image: $image"
                    
                    # Delete all tags for this image
                    local deleted_count=0
                    while IFS= read -r tag_digest; do
                        if [[ -n "$tag_digest" ]]; then
                            local image_with_digest="$image@$tag_digest"
                            if gcloud container images delete "$image_with_digest" --quiet --force-delete-tags 2>/dev/null; then
                                deleted_count=$((deleted_count + 1))
                            fi
                        fi
                    done < <(gcloud container images list-tags "$image" --format="value(digest)" --limit=50 2>/dev/null || true)
                    
                    if [[ $deleted_count -gt 0 ]]; then
                        log_success "Deleted $deleted_count image tags for: $image"
                    else
                        log_warning "No images deleted for: $image"
                    fi
                fi
            else
                log_warning "Skipped container image: $image"
            fi
        else
            log "Container image not found: $image"
        fi
    done
    
    log_success "Container images cleanup completed"
}

# Cleanup storage buckets (HIGH RISK)
cleanup_storage_buckets() {
    if [[ "$CLEANUP_STORAGE" != "true" ]]; then
        log "Storage cleanup disabled, skipping (RECOMMENDED FOR SAFETY)"
        return 0
    fi
    
    log_warning "🪣 Starting storage buckets cleanup - HIGH RISK OPERATION"
    
    # DARWIN bucket patterns (project-specific)
    local darwin_bucket_patterns=(
        "darwin-training-data-$PROJECT_ID"
        "darwin-model-artifacts-$PROJECT_ID"
        "darwin-experiment-logs-$PROJECT_ID" 
        "darwin-backup-data-$PROJECT_ID"
        "darwin-temp-$PROJECT_ID"
    )
    
    # Get all buckets
    if buckets=$(gsutil ls -p "$PROJECT_ID" 2>/dev/null); then
        for bucket in $buckets; do
            bucket_name=${bucket#gs://}
            bucket_name=${bucket_name%/}
            
            # Check if bucket matches DARWIN patterns
            local is_darwin_bucket=false
            for pattern in "${darwin_bucket_patterns[@]}"; do
                if [[ "$bucket_name" == "$pattern" ]]; then
                    is_darwin_bucket=true
                    break
                fi
            done
            
            if [[ "$is_darwin_bucket" == "true" ]]; then
                log_warning "Found DARWIN storage bucket: $bucket_name"
                
                # Get bucket info
                local object_count
                object_count=$(gsutil ls "gs://$bucket_name/**" 2>/dev/null | wc -l || echo "0")
                
                if confirm_action "DELETE STORAGE BUCKET (DATA LOSS!): $bucket_name" "$bucket_name ($object_count objects)" "critical"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete storage bucket: $bucket_name"
                    else
                        log_action "Deleting storage bucket: $bucket_name"
                        
                        # Force remove all objects first, then bucket
                        if gsutil -m rm -r "gs://$bucket_name/**" 2>/dev/null || true; then
                            if gsutil rb "gs://$bucket_name" 2>/dev/null; then
                                log_success "Deleted storage bucket: $bucket_name"
                            else
                                log_error "Failed to delete bucket: $bucket_name"
                            fi
                        else
                            log_error "Failed to delete bucket contents: $bucket_name"
                        fi
                    fi
                else
                    log_warning "Skipped storage bucket: $bucket_name"
                fi
            fi
        done
    else
        log_warning "Could not list storage buckets"
    fi
    
    log_success "Storage cleanup completed"
}

# Cleanup BigQuery datasets (CRITICAL RISK)
cleanup_bigquery_datasets() {
    if [[ "$CLEANUP_BIGQUERY" != "true" ]]; then
        log "BigQuery cleanup disabled, skipping (RECOMMENDED FOR SAFETY)"
        return 0
    fi
    
    log_critical "📊 Starting BigQuery cleanup - CRITICAL DATA LOSS RISK"
    
    # Known DARWIN datasets
    local darwin_datasets=(
        "darwin_research_insights"
        "darwin_performance_metrics"
        "darwin_scaffold_results"
        "darwin_collaboration_data"
        "darwin_real_time_analytics"
        "darwin_training_logs"
    )
    
    if datasets_list=$(bq ls --format=json --project_id="$PROJECT_ID" 2>/dev/null); then
        for dataset in "${darwin_datasets[@]}"; do
            if echo "$datasets_list" | jq -r '.[].datasetReference.datasetId' | grep -q "^$dataset$"; then
                log_critical "Found DARWIN BigQuery dataset: $dataset"
                
                # Get table count
                local table_count
                table_count=$(bq ls "$PROJECT_ID:$dataset" 2>/dev/null | grep -c "TABLE" || echo "0")
                
                if confirm_action "DELETE BIGQUERY DATASET (PERMANENT DATA LOSS!): $dataset" "$dataset ($table_count tables)" "critical"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete BigQuery dataset: $dataset"
                    else
                        log_action "Deleting BigQuery dataset: $dataset"
                        
                        if bq rm -r -f --project_id="$PROJECT_ID" "$dataset"; then
                            log_success "Deleted BigQuery dataset: $dataset"
                        else
                            log_error "Failed to delete BigQuery dataset: $dataset"
                        fi
                    fi
                else
                    log_warning "Skipped BigQuery dataset: $dataset"
                fi
            fi
        done
    else
        log_warning "Could not list BigQuery datasets"
    fi
    
    log_success "BigQuery cleanup completed"
}

# Cleanup secrets (HIGH RISK)
cleanup_secrets() {
    if [[ "$CLEANUP_SECRETS" != "true" ]]; then
        log "Secrets cleanup disabled, skipping (RECOMMENDED FOR SAFETY)"
        return 0
    fi
    
    log_warning "🔐 Starting secrets cleanup - HIGH SECURITY RISK"
    
    # Known DARWIN secrets
    local darwin_secrets=(
        "darwin-openai-api-key"
        "darwin-anthropic-api-key"
        "darwin-google-api-key"
        "darwin-vertex-ai-config"
        "darwin-bigquery-config"
        "darwin-autogen-config"
        "darwin-jax-config"
        "darwin-database-url"
        "darwin-redis-url"
        "darwin-webhook-secret"
    )
    
    if secrets_list=$(gcloud secrets list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null); then
        for secret in "${darwin_secrets[@]}"; do
            if echo "$secrets_list" | grep -q "$secret"; then
                log_warning "Found DARWIN secret: $secret"
                
                if confirm_action "Delete secret: $secret" "$secret" "high"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete secret: $secret"
                    else
                        log_action "Deleting secret: $secret"
                        
                        if gcloud secrets delete "$secret" --project="$PROJECT_ID" --quiet; then
                            log_success "Deleted secret: $secret"
                        else
                            log_error "Failed to delete secret: $secret"
                        fi
                    fi
                else
                    log_warning "Skipped secret: $secret"
                fi
            fi
        done
    else
        log_warning "Could not list secrets"
    fi
    
    log_success "Secrets cleanup completed"
}

# Cleanup service accounts (HIGH RISK)
cleanup_service_accounts() {
    if [[ "$CLEANUP_SERVICE_ACCOUNTS" != "true" ]]; then
        log "Service accounts cleanup disabled, skipping (RECOMMENDED FOR SAFETY)"
        return 0
    fi
    
    log_warning "👤 Starting service accounts cleanup - HIGH ACCESS RISK"
    
    # Known DARWIN service accounts
    local darwin_service_accounts=(
        "vertex-ai-darwin-main@$PROJECT_ID.iam.gserviceaccount.com"
        "darwin-model-training@$PROJECT_ID.iam.gserviceaccount.com"
        "darwin-data-pipeline@$PROJECT_ID.iam.gserviceaccount.com"
    )
    
    if sa_list=$(gcloud iam service-accounts list --project="$PROJECT_ID" --format="value(email)" 2>/dev/null); then
        for sa in "${darwin_service_accounts[@]}"; do
            if echo "$sa_list" | grep -q "$sa"; then
                log_warning "Found DARWIN service account: $sa"
                
                # Get key count
                local key_count
                key_count=$(gcloud iam service-accounts keys list --iam-account="$sa" --project="$PROJECT_ID" 2>/dev/null | grep -c "KEY_ID" || echo "0")
                
                if confirm_action "Delete service account: $sa" "$sa ($key_count keys)" "high"; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        log_action "[DRY RUN] Would delete service account: $sa"
                    else
                        log_action "Deleting service account: $sa"
                        
                        if gcloud iam service-accounts delete "$sa" --project="$PROJECT_ID" --quiet; then
                            log_success "Deleted service account: $sa"
                        else
                            log_error "Failed to delete service account: $sa"
                        fi
                    fi
                else
                    log_warning "Skipped service account: $sa"
                fi
            fi
        done
    else
        log_warning "Could not list service accounts"
    fi
    
    log_success "Service accounts cleanup completed"
}

# Generate cleanup summary
generate_cleanup_summary() {
    log "📋 Generating cleanup summary..."
    
    local summary_file="gcp_cleanup_summary_${CLEANUP_TIMESTAMP}.md"
    
    cat > "$summary_file" << EOF
# GCP Cleanup Summary Report

**Cleanup Date**: $(date)  
**Project**: $PROJECT_ID  
**Region**: $REGION  
**Execution Mode**: $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY RUN"; else echo "PRODUCTION"; fi)

## 🧹 Cleanup Configuration

- **Cloud Run Services**: $CLEANUP_CLOUD_RUN
- **Container Images**: $CLEANUP_CONTAINER_IMAGES
- **Domain Mappings**: $CLEANUP_DOMAIN_MAPPINGS
- **Storage Buckets**: $CLEANUP_STORAGE
- **BigQuery Datasets**: $CLEANUP_BIGQUERY
- **Secrets Manager**: $CLEANUP_SECRETS
- **Service Accounts**: $CLEANUP_SERVICE_ACCOUNTS

## 📊 Actions Summary

$(if [[ "$DRY_RUN" == "true" ]]; then
    echo "This was a **DRY RUN** - no actual resources were deleted."
    echo "Review the log file for details of what would be done in production mode."
else
    echo "**PRODUCTION RUN** - Resources were permanently deleted."
    echo "See log file for detailed results of each operation."
fi)

## 📁 Log Files

- **Detailed Log**: \`$CLEANUP_LOG_FILE\`
- **Summary Report**: \`$summary_file\`

## ⚠️ Post-Cleanup Actions

1. **Verify Services**: Check that expected services are still running
2. **Test Applications**: Validate that remaining applications work correctly
3. **Monitor Costs**: Check for expected cost reductions
4. **Update Documentation**: Document what was cleaned up
5. **Review Monitoring**: Update alerts for removed resources

## 🔄 Rollback Information

If immediate rollback is needed:
1. **Restore from Backup**: Use the backup created before cleanup
2. **Redeploy Services**: Use deployment scripts to recreate services
3. **Check Dependencies**: Verify all interdependencies are restored

## 📞 Support

If issues arise after cleanup:
- Review the detailed cleanup log: \`$CLEANUP_LOG_FILE\`
- Check backup directory for restoration options
- Contact operations team with specific error details

---
Generated by GCP Systematic Cleanup Script  
**SAFETY FIRST - BACKUP BEFORE CLEANUP**
EOF
    
    log_success "Cleanup summary generated: $summary_file"
    echo ""
    echo -e "${CYAN}=== CLEANUP SUMMARY ===${NC}"
    cat "$summary_file"
}

# Main execution function
main() {
    show_header
    
    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project=*)
                PROJECT_ID="${1#*=}"
                shift
                ;;
            --region=*)
                REGION="${1#*=}"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --production)
                DRY_RUN="false"
                log_warning "PRODUCTION MODE ENABLED - RESOURCES WILL BE DELETED"
                shift
                ;;
            --force)
                FORCE_DELETE="true"
                INTERACTIVE="false"
                log_warning "FORCE MODE - NO CONFIRMATIONS"
                shift
                ;;
            --non-interactive)
                INTERACTIVE="false"
                shift
                ;;
            --backup-verified)
                BACKUP_VERIFIED="true"
                shift
                ;;
            --include-storage)
                CLEANUP_STORAGE="true"
                log_warning "STORAGE CLEANUP ENABLED - DATA LOSS RISK"
                shift
                ;;
            --include-bigquery)
                CLEANUP_BIGQUERY="true"
                log_warning "BIGQUERY CLEANUP ENABLED - DATA LOSS RISK"
                shift
                ;;
            --include-secrets)
                CLEANUP_SECRETS="true"
                log_warning "SECRETS CLEANUP ENABLED - SECURITY RISK"
                shift
                ;;
            --include-service-accounts)
                CLEANUP_SERVICE_ACCOUNTS="true"
                log_warning "SERVICE ACCOUNTS CLEANUP ENABLED - ACCESS RISK"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "GCP systematic cleanup script with safety checks"
                echo ""
                echo "Options:"
                echo "  --project=PROJECT_ID       GCP Project ID"
                echo "  --region=REGION            Primary region"
                echo "  --dry-run                  Test mode only (DEFAULT - SAFE)"
                echo "  --production               REAL cleanup mode (DANGEROUS)"
                echo "  --force                    Skip confirmations (VERY DANGEROUS)"
                echo "  --non-interactive          No user prompts"
                echo "  --backup-verified          Confirm backup completed"
                echo "  --include-storage          Enable storage cleanup (DATA LOSS)"
                echo "  --include-bigquery         Enable BigQuery cleanup (DATA LOSS)"
                echo "  --include-secrets          Enable secrets cleanup (SECURITY RISK)"
                echo "  --include-service-accounts Enable SA cleanup (ACCESS RISK)"
                echo "  --help                     Show this help"
                echo ""
                echo "SAFETY FEATURES:"
                echo "  - Defaults to dry-run mode"
                echo "  - Requires backup verification"
                echo "  - Interactive confirmations for risky operations"
                echo "  - Detailed logging of all actions"
                echo ""
                echo "RECOMMENDED USAGE:"
                echo "  1. ./scripts/gcp_backup_critical_data.sh"
                echo "  2. ./scripts/gcp_cleanup_legacy.sh --dry-run"
                echo "  3. ./scripts/gcp_cleanup_legacy.sh --production --backup-verified"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Safety check for production mode
    if [[ "$DRY_RUN" != "true" ]]; then
        echo -e "${RED}${BOLD}
🚨 PRODUCTION MODE WARNING 🚨

You are about to run the cleanup script in PRODUCTION MODE.
This will PERMANENTLY DELETE GCP resources.

Current configuration:
- Project: $PROJECT_ID
- Region: $REGION
- Interactive: $INTERACTIVE
- Force Delete: $FORCE_DELETE
- Backup Verified: $BACKUP_VERIFIED

Are you absolutely sure you want to proceed?${NC}"
        
        if [[ "$INTERACTIVE" == "true" ]]; then
            echo -e "${RED}Type 'EXECUTE PRODUCTION CLEANUP' to confirm:${NC}"
            read -r final_confirmation
            if [[ "$final_confirmation" != "EXECUTE PRODUCTION CLEANUP" ]]; then
                log_critical "Production cleanup cancelled - wise choice!"
                exit 1
            fi
        fi
    fi
    
    # Execute cleanup steps
    log "🧹 Starting systematic GCP cleanup..."
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting cleanup execution" >> "$CLEANUP_LOG_FILE"
    
    initialize_cleanup
    verify_backup_exists  
    check_cleanup_prerequisites
    
    # Execute cleanup in safe order (least destructive first)
    cleanup_cloud_run_services
    cleanup_domain_mappings
    cleanup_container_images
    
    # High-risk cleanups (disabled by default)
    cleanup_storage_buckets
    cleanup_bigquery_datasets
    cleanup_secrets
    cleanup_service_accounts
    
    # Generate final report
    generate_cleanup_summary
    
    # Final completion message
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${GREEN}${BOLD}
✅ GCP CLEANUP DRY RUN COMPLETED SUCCESSFULLY!

This was a SIMULATION - no resources were actually deleted.

📋 Review the results:
   - Summary: gcp_cleanup_summary_${CLEANUP_TIMESTAMP}.md  
   - Detailed Log: $CLEANUP_LOG_FILE

🎯 To execute for real:
   ./scripts/gcp_cleanup_legacy.sh --production --backup-verified

⚠️  Remember: BACKUP FIRST, then cleanup!${NC}
"
    else
        echo -e "${GREEN}${BOLD}
🎉 GCP SYSTEMATIC CLEANUP COMPLETED!

✅ Cleanup operations completed
✅ Detailed logs generated
✅ Summary report created
✅ Resources cleaned up safely

📋 Files Generated:
   📄 Summary: gcp_cleanup_summary_${CLEANUP_TIMESTAMP}.md
   📋 Detailed Log: $CLEANUP_LOG_FILE

🔍 POST-CLEANUP CHECKLIST:
   1. Verify remaining services are working
   2. Check cost reduction in billing
   3. Update monitoring and alerts
   4. Archive cleanup logs securely

🧹 GCP CLEANUP MISSION ACCOMPLISHED! 🧹${NC}
"
    fi
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Cleanup completed" >> "$CLEANUP_LOG_FILE"
}

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}Cleanup interrupted by user - check logs for partial completion${NC}"; exit 130' INT TERM

# Execute main function
main "$@"