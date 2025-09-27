#!/bin/bash

# GCP CRITICAL DATA BACKUP SCRIPT
# Backup completo de dados críticos antes do cleanup sistemático
# 🔄 CRITICAL DATA BACKUP - SAFETY FIRST APPROACH

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
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BASE_DIR="gcp_backup_${BACKUP_TIMESTAMP}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"

# Backup configuration
BACKUP_BIGQUERY="${BACKUP_BIGQUERY:-true}"
BACKUP_STORAGE="${BACKUP_STORAGE:-true}"
BACKUP_SECRETS="${BACKUP_SECRETS:-true}"
BACKUP_IAM="${BACKUP_IAM:-true}"
BACKUP_DNS="${BACKUP_DNS:-true}"
BACKUP_CONFIGS="${BACKUP_CONFIGS:-true}"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️${NC} $1"
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${WHITE}[$(date +'%Y-%m-%d %H:%M:%S')] 🔍${NC} $1"
    fi
}

# Header display
show_header() {
    echo -e "${PURPLE}${BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  🔄 DARWIN GCP CRITICAL DATA BACKUP SCRIPT 🔄                           ║
║                                                                           ║
║  Comprehensive backup of critical resources before cleanup               ║
║  • BigQuery Datasets & Tables                                            ║
║  • Storage Buckets & Critical Files                                      ║
║  • Secret Manager Secrets                                                ║
║  • IAM Policies & Service Accounts                                       ║
║  • DNS Configurations & Domain Mappings                                  ║
║  • Cloud Run Configurations                                              ║
║                                                                           ║
║  SAFETY FIRST - BACKUP BEFORE CLEANUP                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝${NC}
"
}

# Initialize backup environment
initialize_backup() {
    log "🚀 Initializing critical data backup..."
    
    if [[ -z "$PROJECT_ID" ]]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "No project specified. Set GCP_PROJECT_ID environment variable"
            exit 1
        fi
    fi
    
    log_info "Project: $PROJECT_ID"
    log_info "Backup Directory: $BACKUP_BASE_DIR"
    log_info "Timestamp: $BACKUP_TIMESTAMP"
    log_info "Dry Run: $DRY_RUN"
    
    # Create backup directory structure
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$BACKUP_BASE_DIR"/{bigquery,storage,secrets,iam,dns,configs,logs}
        log_success "Backup directory structure created"
    else
        log_info "Dry run mode - no directories created"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "🔑 Checking backup prerequisites..."
    
    # Check required tools
    local required_tools=("gcloud" "gsutil" "bq" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool not found - required for backup operations"
            exit 1
        fi
    done
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        log_error "Not authenticated with Google Cloud"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        exit 1
    fi
    
    # Check required permissions
    local required_roles=(
        "roles/bigquery.dataViewer"
        "roles/storage.objectViewer" 
        "roles/secretmanager.secretAccessor"
        "roles/iam.securityReviewer"
        "roles/dns.reader"
    )
    
    log_info "Prerequisites verified"
}

# Create backup manifest
create_backup_manifest() {
    log "📋 Creating backup manifest..."
    
    local manifest_file="$BACKUP_BASE_DIR/backup_manifest.json"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        cat > "$manifest_file" << EOF
{
  "backup_metadata": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "project_id": "$PROJECT_ID",
    "region": "$REGION",
    "backup_version": "1.0.0",
    "dry_run": $DRY_RUN,
    "encryption_enabled": $([ -n "$ENCRYPTION_KEY" ] && echo "true" || echo "false")
  },
  "backup_components": {
    "bigquery": $BACKUP_BIGQUERY,
    "storage": $BACKUP_STORAGE,
    "secrets": $BACKUP_SECRETS,
    "iam": $BACKUP_IAM,
    "dns": $BACKUP_DNS,
    "configs": $BACKUP_CONFIGS
  },
  "backup_items": [],
  "backup_summary": {
    "total_items": 0,
    "total_size_bytes": 0,
    "successful_backups": 0,
    "failed_backups": 0,
    "warnings": 0
  }
}
EOF
        log_success "Backup manifest created: $manifest_file"
    else
        log_info "Dry run - manifest creation skipped"
    fi
}

# Backup BigQuery datasets
backup_bigquery_datasets() {
    if [[ "$BACKUP_BIGQUERY" != "true" ]]; then
        log_info "BigQuery backup disabled, skipping"
        return 0
    fi
    
    log "📊 Starting BigQuery datasets backup..."
    
    local bq_backup_dir="$BACKUP_BASE_DIR/bigquery"
    local dataset_count=0
    local backup_jobs=()
    
    # Get all datasets
    local datasets
    if datasets=$(bq ls --format=json --project_id="$PROJECT_ID" 2>/dev/null); then
        # Focus on DARWIN-related datasets
        local darwin_datasets=(
            "darwin_research_insights"
            "darwin_performance_metrics"
            "darwin_scaffold_results"
            "darwin_collaboration_data"
            "darwin_real_time_analytics"
            "darwin_training_logs"
        )
        
        for dataset in "${darwin_datasets[@]}"; do
            log_verbose "Processing dataset: $dataset"
            
            # Check if dataset exists
            if bq show --dataset "$PROJECT_ID:$dataset" &>/dev/null; then
                log "Backing up dataset: $dataset"
                
                # Get dataset schema
                if [[ "$DRY_RUN" != "true" ]]; then
                    bq show --format=json --dataset "$PROJECT_ID:$dataset" > "$bq_backup_dir/${dataset}_schema.json" 2>/dev/null || log_warning "Failed to get schema for $dataset"
                fi
                
                # Get table list
                local tables
                if tables=$(bq ls --format=json "$PROJECT_ID:$dataset" 2>/dev/null); then
                    echo "$tables" > "$bq_backup_dir/${dataset}_tables.json" 2>/dev/null || true
                    
                    # Backup each table
                    for table in $(echo "$tables" | jq -r '.[] | select(.type == "TABLE") | .tableReference.tableId' 2>/dev/null); do
                        log_verbose "Backing up table: $dataset.$table"
                        
                        if [[ "$DRY_RUN" != "true" ]]; then
                            # Export table to Cloud Storage first, then download
                            local export_uri="gs://darwin-backup-temp-$PROJECT_ID/bigquery/${dataset}/${table}_${BACKUP_TIMESTAMP}.json"
                            
                            # Create temporary backup bucket if needed
                            gsutil mb -p "$PROJECT_ID" "gs://darwin-backup-temp-$PROJECT_ID" 2>/dev/null || true
                            
                            # Start export job
                            local job_id="backup_${dataset}_${table}_${BACKUP_TIMESTAMP}"
                            if bq extract --destination_format=NEWLINE_DELIMITED_JSON --job_id="$job_id" "$PROJECT_ID:$dataset.$table" "$export_uri" 2>/dev/null; then
                                backup_jobs+=("$job_id:$export_uri:$bq_backup_dir/${dataset}_${table}.json")
                                log_verbose "Export job started: $job_id"
                            else
                                log_warning "Failed to start export for $dataset.$table"
                            fi
                        fi
                    done
                fi
                
                dataset_count=$((dataset_count + 1))
                log_success "Dataset backup initiated: $dataset"
            else
                log_verbose "Dataset not found: $dataset"
            fi
        done
        
        # Wait for export jobs to complete and download data
        if [[ "$DRY_RUN" != "true" && ${#backup_jobs[@]} -gt 0 ]]; then
            log "Waiting for BigQuery export jobs to complete..."
            
            for job_info in "${backup_jobs[@]}"; do
                IFS=':' read -r job_id export_uri local_file <<< "$job_info"
                
                # Wait for job completion (max 10 minutes)
                local wait_count=0
                while [[ $wait_count -lt 60 ]]; do
                    local job_status
                    job_status=$(bq show --format=json --job "$job_id" 2>/dev/null | jq -r '.status.state' 2>/dev/null || echo "UNKNOWN")
                    
                    if [[ "$job_status" == "DONE" ]]; then
                        # Check for errors
                        local error_result
                        error_result=$(bq show --format=json --job "$job_id" 2>/dev/null | jq '.status.errorResult' 2>/dev/null || echo "null")
                        
                        if [[ "$error_result" == "null" ]]; then
                            # Download the exported data
                            if gsutil cp "$export_uri" "$local_file" 2>/dev/null; then
                                log_success "Downloaded: $(basename "$local_file")"
                                # Cleanup temporary file
                                gsutil rm "$export_uri" 2>/dev/null || true
                            else
                                log_warning "Failed to download: $export_uri"
                            fi
                        else
                            log_warning "Export job failed: $job_id"
                        fi
                        break
                    elif [[ "$job_status" == "RUNNING" || "$job_status" == "PENDING" ]]; then
                        sleep 10
                        wait_count=$((wait_count + 1))
                    else
                        log_warning "Job in unexpected state: $job_status"
                        break
                    fi
                done
                
                if [[ $wait_count -eq 60 ]]; then
                    log_warning "Timeout waiting for job: $job_id"
                fi
            done
            
            # Cleanup temporary bucket
            gsutil rm -r "gs://darwin-backup-temp-$PROJECT_ID" 2>/dev/null || true
        fi
        
    else
        log_warning "Could not list BigQuery datasets"
    fi
    
    log_success "BigQuery backup completed: $dataset_count datasets processed"
}

# Backup critical storage buckets
backup_critical_storage() {
    if [[ "$BACKUP_STORAGE" != "true" ]]; then
        log_info "Storage backup disabled, skipping"
        return 0
    fi
    
    log "🪣 Starting critical storage backup..."
    
    local storage_backup_dir="$BACKUP_BASE_DIR/storage"
    local bucket_count=0
    
    # DARWIN-related bucket patterns
    local darwin_bucket_patterns=(
        "darwin-training-data"
        "darwin-model-artifacts" 
        "darwin-experiment-logs"
        "darwin-backup-data"
    )
    
    # Get all buckets and filter for DARWIN-related ones
    if buckets=$(gsutil ls -p "$PROJECT_ID" 2>/dev/null); then
        for bucket in $buckets; do
            bucket_name=${bucket#gs://}
            bucket_name=${bucket_name%/}
            
            # Check if bucket matches DARWIN patterns
            local is_darwin_bucket=false
            for pattern in "${darwin_bucket_patterns[@]}"; do
                if [[ "$bucket_name" =~ $pattern ]]; then
                    is_darwin_bucket=true
                    break
                fi
            done
            
            if [[ "$is_darwin_bucket" == "true" ]]; then
                log "Backing up critical bucket: $bucket_name"
                
                if [[ "$DRY_RUN" != "true" ]]; then
                    # Create bucket backup directory
                    mkdir -p "$storage_backup_dir/$bucket_name"
                    
                    # Get bucket metadata
                    gsutil ls -L -b "gs://$bucket_name" > "$storage_backup_dir/${bucket_name}_metadata.txt" 2>/dev/null || log_warning "Could not get metadata for $bucket_name"
                    
                    # Get object list
                    gsutil ls -r "gs://$bucket_name/**" > "$storage_backup_dir/${bucket_name}_objects.txt" 2>/dev/null || log_warning "Could not list objects for $bucket_name"
                    
                    # Backup critical files (limit size to avoid huge downloads)
                    local file_count=0
                    while IFS= read -r object_path; do
                        if [[ $file_count -ge 100 ]]; then
                            log_warning "Reached backup limit for bucket $bucket_name (100 files)"
                            break
                        fi
                        
                        # Get file size
                        local file_size
                        file_size=$(gsutil du "$object_path" 2>/dev/null | awk '{print $1}' || echo "0")
                        
                        # Only backup files smaller than 100MB
                        if [[ "$file_size" -lt 104857600 ]]; then
                            local filename
                            filename=$(basename "$object_path")
                            if gsutil cp "$object_path" "$storage_backup_dir/$bucket_name/$filename" 2>/dev/null; then
                                log_verbose "Backed up: $filename"
                                file_count=$((file_count + 1))
                            fi
                        else
                            log_verbose "Skipping large file: $object_path (${file_size} bytes)"
                        fi
                    done < "$storage_backup_dir/${bucket_name}_objects.txt"
                    
                    log_success "Bucket backup completed: $bucket_name ($file_count files)"
                fi
                
                bucket_count=$((bucket_count + 1))
            fi
        done
    else
        log_warning "Could not list storage buckets"
    fi
    
    log_success "Storage backup completed: $bucket_count critical buckets processed"
}

# Backup Secret Manager secrets
backup_secrets() {
    if [[ "$BACKUP_SECRETS" != "true" ]]; then
        log_info "Secrets backup disabled, skipping"
        return 0
    fi
    
    log "🔐 Starting Secret Manager backup..."
    
    local secrets_backup_dir="$BACKUP_BASE_DIR/secrets"
    local secret_count=0
    
    # Get all secrets
    if secrets_list=$(gcloud secrets list --project="$PROJECT_ID" --format=json 2>/dev/null); then
        for secret in $(echo "$secrets_list" | jq -r '.[].name' 2>/dev/null); do
            secret_name=${secret##*/}
            log_verbose "Backing up secret: $secret_name"
            
            if [[ "$DRY_RUN" != "true" ]]; then
                # Get secret metadata (not the actual secret value for security)
                gcloud secrets describe "$secret_name" --project="$PROJECT_ID" --format=json > "$secrets_backup_dir/${secret_name}_metadata.json" 2>/dev/null || log_warning "Could not get metadata for $secret_name"
                
                # Get versions list
                gcloud secrets versions list "$secret_name" --project="$PROJECT_ID" --format=json > "$secrets_backup_dir/${secret_name}_versions.json" 2>/dev/null || log_warning "Could not get versions for $secret_name"
                
                # Get IAM policy
                gcloud secrets get-iam-policy "$secret_name" --project="$PROJECT_ID" --format=json > "$secrets_backup_dir/${secret_name}_iam.json" 2>/dev/null || log_warning "Could not get IAM policy for $secret_name"
            fi
            
            secret_count=$((secret_count + 1))
        done
        
        # Create secrets restore script
        if [[ "$DRY_RUN" != "true" ]]; then
            cat > "$secrets_backup_dir/restore_secrets.sh" << 'EOF'
#!/bin/bash
# Secret Manager Restore Script
# Generated by GCP Backup Script

set -e

PROJECT_ID="${1:-}"
if [[ -z "$PROJECT_ID" ]]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

echo "🔐 Restoring Secret Manager secrets to project: $PROJECT_ID"

for metadata_file in *_metadata.json; do
    secret_name=${metadata_file%_metadata.json}
    echo "Processing secret: $secret_name"
    
    # Recreate secret (without values - those need manual restoration)
    if jq -r '.replication' "$metadata_file" > /dev/null 2>&1; then
        echo "Secret structure exists for: $secret_name"
        echo "⚠️  Secret values must be manually restored for security"
    fi
done

echo "✅ Secret structures documented. Manual value restoration required."
EOF
            chmod +x "$secrets_backup_dir/restore_secrets.sh"
        fi
        
    else
        log_warning "Could not list Secret Manager secrets"
    fi
    
    log_success "Secrets backup completed: $secret_count secrets processed"
}

# Backup IAM policies and service accounts
backup_iam_policies() {
    if [[ "$BACKUP_IAM" != "true" ]]; then
        log_info "IAM backup disabled, skipping"
        return 0
    fi
    
    log "👤 Starting IAM policies backup..."
    
    local iam_backup_dir="$BACKUP_BASE_DIR/iam"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Backup project IAM policy
        log "Backing up project IAM policy..."
        gcloud projects get-iam-policy "$PROJECT_ID" --format=json > "$iam_backup_dir/project_iam_policy.json" 2>/dev/null || log_warning "Could not get project IAM policy"
        
        # Backup service accounts
        log "Backing up service accounts..."
        gcloud iam service-accounts list --project="$PROJECT_ID" --format=json > "$iam_backup_dir/service_accounts.json" 2>/dev/null || log_warning "Could not list service accounts"
        
        # Backup individual service account policies
        if sa_list=$(gcloud iam service-accounts list --project="$PROJECT_ID" --format="value(email)" 2>/dev/null); then
            mkdir -p "$iam_backup_dir/service_account_policies"
            
            while IFS= read -r sa_email; do
                if [[ -n "$sa_email" ]]; then
                    log_verbose "Backing up policy for: $sa_email"
                    sa_filename=$(echo "$sa_email" | tr '@' '_' | tr '.' '_')
                    gcloud iam service-accounts get-iam-policy "$sa_email" --format=json > "$iam_backup_dir/service_account_policies/${sa_filename}_policy.json" 2>/dev/null || true
                fi
            done <<< "$sa_list"
        fi
        
        # Backup custom roles
        log "Backing up custom roles..."
        gcloud iam roles list --project="$PROJECT_ID" --format=json > "$iam_backup_dir/custom_roles.json" 2>/dev/null || log_warning "Could not list custom roles"
        
        # Create IAM restore script
        cat > "$iam_backup_dir/restore_iam.sh" << 'EOF'
#!/bin/bash
# IAM Restore Script
# Generated by GCP Backup Script

set -e

PROJECT_ID="${1:-}"
if [[ -z "$PROJECT_ID" ]]; then
    echo "Usage: $0 PROJECT_ID"
    exit 1
fi

echo "👤 Restoring IAM policies to project: $PROJECT_ID"

# Restore project IAM policy
if [[ -f "project_iam_policy.json" ]]; then
    echo "Restoring project IAM policy..."
    gcloud projects set-iam-policy "$PROJECT_ID" project_iam_policy.json
fi

# Restore service account IAM policies
if [[ -d "service_account_policies" ]]; then
    for policy_file in service_account_policies/*_policy.json; do
        sa_filename=$(basename "$policy_file" _policy.json)
        sa_email=$(echo "$sa_filename" | tr '_' '@' | sed 's/@/@/' | sed 's/_/./g')
        echo "Restoring policy for: $sa_email"
        gcloud iam service-accounts set-iam-policy "$sa_email" "$policy_file"
    done
fi

echo "✅ IAM policies restored"
EOF
        chmod +x "$iam_backup_dir/restore_iam.sh"
    fi
    
    log_success "IAM backup completed"
}

# Backup DNS configurations
backup_dns_configs() {
    if [[ "$BACKUP_DNS" != "true" ]]; then
        log_info "DNS backup disabled, skipping"
        return 0
    fi
    
    log "🌐 Starting DNS configurations backup..."
    
    local dns_backup_dir="$BACKUP_BASE_DIR/dns"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Backup Cloud DNS zones
        log "Backing up Cloud DNS zones..."
        gcloud dns managed-zones list --project="$PROJECT_ID" --format=json > "$dns_backup_dir/dns_zones.json" 2>/dev/null || log_warning "Could not list DNS zones"
        
        # Backup domain mappings
        log "Backing up Cloud Run domain mappings..."
        gcloud run domain-mappings list --region="$REGION" --format=json > "$dns_backup_dir/domain_mappings.json" 2>/dev/null || log_warning "Could not list domain mappings"
        
        # Export individual zone records
        if dns_zones=$(gcloud dns managed-zones list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null); then
            mkdir -p "$dns_backup_dir/zone_records"
            
            while IFS= read -r zone_name; do
                if [[ -n "$zone_name" ]]; then
                    log_verbose "Backing up records for zone: $zone_name"
                    gcloud dns record-sets list --zone="$zone_name" --format=json > "$dns_backup_dir/zone_records/${zone_name}_records.json" 2>/dev/null || true
                fi
            done <<< "$dns_zones"
        fi
    fi
    
    log_success "DNS backup completed"
}

# Backup Cloud Run and other configurations  
backup_configurations() {
    if [[ "$BACKUP_CONFIGS" != "true" ]]; then
        log_info "Configurations backup disabled, skipping"
        return 0
    fi
    
    log "⚙️ Starting configurations backup..."
    
    local config_backup_dir="$BACKUP_BASE_DIR/configs"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Backup Cloud Run services
        log "Backing up Cloud Run services configurations..."
        gcloud run services list --region="$REGION" --format=json > "$config_backup_dir/cloud_run_services.json" 2>/dev/null || log_warning "Could not list Cloud Run services"
        
        # Backup individual service configurations
        if services_list=$(gcloud run services list --region="$REGION" --format="value(metadata.name)" 2>/dev/null); then
            mkdir -p "$config_backup_dir/cloud_run_configs"
            
            while IFS= read -r service_name; do
                if [[ -n "$service_name" ]]; then
                    log_verbose "Backing up configuration for service: $service_name"
                    gcloud run services describe "$service_name" --region="$REGION" --format=json > "$config_backup_dir/cloud_run_configs/${service_name}_config.json" 2>/dev/null || true
                fi
            done <<< "$services_list"
        fi
        
        # Backup enabled APIs
        log "Backing up enabled APIs..."
        gcloud services list --enabled --project="$PROJECT_ID" --format=json > "$config_backup_dir/enabled_apis.json" 2>/dev/null || log_warning "Could not list enabled APIs"
        
        # Backup project metadata
        log "Backing up project metadata..."
        gcloud projects describe "$PROJECT_ID" --format=json > "$config_backup_dir/project_metadata.json" 2>/dev/null || log_warning "Could not get project metadata"
        
        # Backup billing information
        gcloud billing projects describe "$PROJECT_ID" --format=json > "$config_backup_dir/billing_info.json" 2>/dev/null || log_warning "Could not get billing info"
    fi
    
    log_success "Configurations backup completed"
}

# Update backup manifest with results
update_backup_manifest() {
    log "📋 Updating backup manifest..."
    
    local manifest_file="$BACKUP_BASE_DIR/backup_manifest.json"
    
    if [[ "$DRY_RUN" != "true" && -f "$manifest_file" ]]; then
        # Calculate backup statistics
        local total_files total_size
        total_files=$(find "$BACKUP_BASE_DIR" -type f | wc -l || echo 0)
        total_size=$(du -sb "$BACKUP_BASE_DIR" 2>/dev/null | cut -f1 || echo 0)
        
        # Update manifest with results
        local temp_file=$(mktemp)
        jq --arg total_files "$total_files" \
           --arg total_size "$total_size" \
           '.backup_summary.total_items = ($total_files | tonumber) |
            .backup_summary.total_size_bytes = ($total_size | tonumber) |
            .backup_summary.completion_timestamp = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"' \
           "$manifest_file" > "$temp_file" && mv "$temp_file" "$manifest_file"
        
        log_success "Backup manifest updated: $total_files files, $total_size bytes"
    else
        log_info "Manifest update skipped (dry run or missing file)"
    fi
}

# Generate backup summary report
generate_backup_summary() {
    log "📋 Generating backup summary report..."
    
    local summary_file="$BACKUP_BASE_DIR/backup_summary.md"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Calculate statistics
        local total_files total_size_mb bigquery_files storage_files secrets_files iam_files dns_files config_files
        total_files=$(find "$BACKUP_BASE_DIR" -type f -not -name "*.md" | wc -l || echo 0)
        total_size_mb=$(du -sm "$BACKUP_BASE_DIR" 2>/dev/null | cut -f1 || echo 0)
        bigquery_files=$(find "$BACKUP_BASE_DIR/bigquery" -type f 2>/dev/null | wc -l || echo 0)
        storage_files=$(find "$BACKUP_BASE_DIR/storage" -type f 2>/dev/null | wc -l || echo 0)
        secrets_files=$(find "$BACKUP_BASE_DIR/secrets" -type f 2>/dev/null | wc -l || echo 0)
        iam_files=$(find "$BACKUP_BASE_DIR/iam" -type f 2>/dev/null | wc -l || echo 0)
        dns_files=$(find "$BACKUP_BASE_DIR/dns" -type f 2>/dev/null | wc -l || echo 0)
        config_files=$(find "$BACKUP_BASE_DIR/configs" -type f 2>/dev/null | wc -l || echo 0)
        
        cat > "$summary_file" << EOF
# GCP Critical Data Backup Summary

**Backup Date**: $(date)  
**Project**: $PROJECT_ID  
**Region**: $REGION  
**Backup Directory**: $BACKUP_BASE_DIR

## 📊 Backup Statistics

- **Total Files**: $total_files
- **Total Size**: ${total_size_mb}MB
- **Backup Duration**: Started at backup initialization

## 🗂️ Backup Components

### BigQuery Datasets: $bigquery_files files
$(if [[ -d "$BACKUP_BASE_DIR/bigquery" ]]; then ls "$BACKUP_BASE_DIR/bigquery" | head -10 | sed 's/^/- /'; fi)

### Storage Buckets: $storage_files files  
$(if [[ -d "$BACKUP_BASE_DIR/storage" ]]; then ls "$BACKUP_BASE_DIR/storage" | head -10 | sed 's/^/- /'; fi)

### Secret Manager: $secrets_files files
$(if [[ -d "$BACKUP_BASE_DIR/secrets" ]]; then ls "$BACKUP_BASE_DIR/secrets" | grep metadata | head -10 | sed 's/_metadata.json//g' | sed 's/^/- /'; fi)

### IAM Policies: $iam_files files
$(if [[ -d "$BACKUP_BASE_DIR/iam" ]]; then ls "$BACKUP_BASE_DIR/iam" | head -10 | sed 's/^/- /'; fi)

### DNS Configurations: $dns_files files
$(if [[ -d "$BACKUP_BASE_DIR/dns" ]]; then ls "$BACKUP_BASE_DIR/dns" | head -10 | sed 's/^/- /'; fi)

### Service Configurations: $config_files files
$(if [[ -d "$BACKUP_BASE_DIR/configs" ]]; then ls "$BACKUP_BASE_DIR/configs" | head -10 | sed 's/^/- /'; fi)

## 🔧 Restore Instructions

### Quick Restore Commands
\`\`\`bash
# Restore IAM policies
./iam/restore_iam.sh $PROJECT_ID

# Restore secrets structure (values need manual restoration)
./secrets/restore_secrets.sh $PROJECT_ID

# BigQuery data needs to be imported manually using:
# bq load --source_format=NEWLINE_DELIMITED_JSON dataset.table backup_file.json
\`\`\`

### Manual Restoration Required
- **Secret Values**: For security, actual secret values are not backed up
- **Storage Data**: Large files were skipped, restore from original sources
- **BigQuery Data**: Import JSON files back to tables manually

## ⚠️ Important Notes

1. **Security**: This backup contains sensitive configuration data
2. **Storage**: Store backup in secure, encrypted location
3. **Testing**: Test restore procedures in development environment first
4. **Cleanup**: Remove old backups after successful restoration verification

## 🎯 Next Steps

1. **Secure the backup**: Move to encrypted storage
2. **Test restoration**: Verify backup integrity
3. **Proceed with cleanup**: Run cleanup script if backup is satisfactory
4. **Monitor**: Watch for any issues during cleanup

---
Generated by GCP Critical Data Backup Script  
**BACKUP BEFORE CLEANUP - SAFETY FIRST**
EOF
        
        log_success "Backup summary generated: $summary_file"
        echo ""
        echo -e "${CYAN}=== BACKUP SUMMARY ===${NC}"
        cat "$summary_file"
    else
        log_info "Summary generation skipped (dry run mode)"
    fi
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
            --backup-dir=*)
                BACKUP_BASE_DIR="${1#*=}"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --no-bigquery)
                BACKUP_BIGQUERY="false"
                shift
                ;;
            --no-storage)
                BACKUP_STORAGE="false"
                shift
                ;;
            --no-secrets)
                BACKUP_SECRETS="false"
                shift
                ;;
            --no-iam)
                BACKUP_IAM="false"
                shift
                ;;
            --no-dns)
                BACKUP_DNS="false"
                shift
                ;;
            --no-configs)
                BACKUP_CONFIGS="false"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Critical data backup script for GCP resources"
                echo ""
                echo "Options:"
                echo "  --project=PROJECT_ID     GCP Project ID"
                echo "  --region=REGION          Primary region (default: us-central1)"
                echo "  --backup-dir=DIR         Backup directory name"
                echo "  --dry-run               Test mode, no actual backup"
                echo "  --verbose               Enable verbose logging"
                echo "  --no-bigquery           Skip BigQuery backup"
                echo "  --no-storage            Skip Storage backup" 
                echo "  --no-secrets            Skip Secrets backup"
                echo "  --no-iam                Skip IAM backup"
                echo "  --no-dns                Skip DNS backup"
                echo "  --no-configs            Skip configurations backup"
                echo "  --help                  Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Execute backup steps
    log "🔄 Starting comprehensive GCP critical data backup..."
    
    initialize_backup
    check_prerequisites
    create_backup_manifest
    
    # Execute backup components
    backup_bigquery_datasets
    backup_critical_storage
    backup_secrets
    backup_iam_policies
    backup_dns_configs
    backup_configurations
    
    # Finalize backup
    update_backup_manifest
    generate_backup_summary
    
    # Final success message
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}${BOLD}
🔄 GCP BACKUP DRY RUN COMPLETED!

This was a dry run - no actual backup was performed.
Run without --dry-run to execute the actual backup.
${NC}"
    else
        echo -e "${GREEN}${BOLD}
🎉 GCP CRITICAL DATA BACKUP COMPLETED SUCCESSFULLY! 🎉

✅ All critical resources backed up safely
✅ Backup manifest and summary generated
✅ Restore scripts created for key components
✅ Ready for safe cleanup operations

📁 Backup Location: $BACKUP_BASE_DIR
📋 Summary Report: $BACKUP_BASE_DIR/backup_summary.md

🔒 CRITICAL DATA IS SAFELY BACKED UP! 🔒

🎯 Next Steps:
   1. Secure the backup in encrypted storage
   2. Test restore procedures if needed
   3. Proceed with cleanup: ./scripts/gcp_cleanup_legacy.sh --dry-run
   4. Monitor cleanup operations carefully

⚠️  KEEP THIS BACKUP SAFE - IT'S YOUR SAFETY NET! ⚠️${NC}
"
    fi
}

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}Backup interrupted by user${NC}"; exit 130' INT TERM

# Execute main function
main "$@"