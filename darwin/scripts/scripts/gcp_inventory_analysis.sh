#!/bin/bash

# GCP INVENTORY & ANALYSIS SCRIPT
# Análise completa do estado atual dos recursos GCP no projeto DARWIN
# 🔍 GCP RESOURCE INVENTORY - COMPLETE ANALYSIS & CLEANUP PREPARATION

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
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"
OUTPUT_FILE="gcp_inventory_$(date +%Y%m%d_%H%M%S).json"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Expected projects based on analysis
KNOWN_PROJECTS=(
    "darwin-biomaterials-scaffolds"
    "pcs-helio"
)

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
║  🔍 DARWIN GCP INVENTORY & ANALYSIS SCRIPT 🔍                           ║
║                                                                           ║
║  Comprehensive analysis of GCP resources for cleanup preparation         ║
║  • Cloud Run Services & Container Images                                 ║
║  • Storage Buckets & BigQuery Datasets                                   ║
║  • IAM Service Accounts & Secrets                                        ║
║  • APIs, Domains & Monitoring Resources                                  ║
║  • Legacy Resource Detection                                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝${NC}
"
}

# Initialize JSON output structure
initialize_inventory() {
    log "🚀 Initializing GCP inventory analysis..."
    
    if [[ -z "$PROJECT_ID" ]]; then
        # Try to detect current project
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "No project specified. Set GCP_PROJECT_ID environment variable or use gcloud config set project"
            exit 1
        fi
    fi
    
    log_info "Target Project: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Output File: $OUTPUT_FILE"
    log_info "Dry Run Mode: $DRY_RUN"
    
    # Initialize JSON structure
    cat > "$OUTPUT_FILE" << EOF
{
  "inventory_metadata": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "project_id": "$PROJECT_ID",
    "region": "$REGION",
    "analysis_version": "1.0.0",
    "dry_run_mode": $DRY_RUN
  },
  "project_info": {},
  "resources": {
    "cloud_run": [],
    "container_registry": [],
    "storage_buckets": [],
    "bigquery_datasets": [],
    "iam_service_accounts": [],
    "secrets_manager": [],
    "apis_enabled": [],
    "domain_mappings": [],
    "monitoring": [],
    "compute_instances": [],
    "networks": [],
    "other_resources": []
  },
  "analysis": {
    "total_resources": 0,
    "potentially_obsolete": [],
    "high_cost_resources": [],
    "security_concerns": [],
    "cleanup_recommendations": []
  }
}
EOF
    
    log_success "Inventory structure initialized"
}

# Check prerequisites and authentication
check_prerequisites() {
    log "🔑 Checking prerequisites and authentication..."
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        log_error "Not authenticated with Google Cloud"
        log "Please run: gcloud auth login"
        exit 1
    fi
    
    # Verify project access
    if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        log "Please check project ID and permissions"
        exit 1
    fi
    
    # Check required tools
    local tools=("jq" "curl")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_warning "$tool not found - some features may be limited"
        fi
    done
    
    log_success "Prerequisites verified"
}

# Update JSON with project information
analyze_project_info() {
    log "📋 Analyzing project information..."
    
    local project_info
    project_info=$(gcloud projects describe "$PROJECT_ID" --format=json 2>/dev/null || echo '{}')
    
    # Get billing info if available
    local billing_account
    billing_account=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "unknown")
    
    # Get quotas info
    local quotas_info='[]'
    if command -v jq &> /dev/null; then
        quotas_info=$(gcloud compute project-info describe --project="$PROJECT_ID" --format=json 2>/dev/null | jq '.quotas // []' || echo '[]')
    fi
    
    # Update JSON
    local temp_file=$(mktemp)
    jq --argjson project_info "$project_info" \
       --arg billing_account "$billing_account" \
       --argjson quotas "$quotas_info" \
       '.project_info = {
          project_details: $project_info,
          billing_account: $billing_account,
          quotas: $quotas
        }' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Project information analyzed"
}

# Analyze Cloud Run services
analyze_cloud_run() {
    log "☁️ Analyzing Cloud Run services..."
    
    local services_json='[]'
    local service_count=0
    
    # Get all Cloud Run services
    if services_json=$(gcloud run services list --region="$REGION" --format=json 2>/dev/null); then
        service_count=$(echo "$services_json" | jq 'length' 2>/dev/null || echo 0)
        log_info "Found $service_count Cloud Run services in region $REGION"
        
        # Get detailed info for each service
        local detailed_services='[]'
        if [[ $service_count -gt 0 ]]; then
            for service in $(echo "$services_json" | jq -r '.[].metadata.name' 2>/dev/null); do
                log_verbose "Analyzing service: $service"
                
                local service_detail
                service_detail=$(gcloud run services describe "$service" --region="$REGION" --format=json 2>/dev/null || echo '{}')
                
                # Get service URL and traffic info
                local service_url
                service_url=$(echo "$service_detail" | jq -r '.status.url // "none"' 2>/dev/null || echo "none")
                
                # Get image info
                local image
                image=$(echo "$service_detail" | jq -r '.spec.template.spec.template.spec.containers[0].image // "unknown"' 2>/dev/null || echo "unknown")
                
                # Check if service is actively used (simplified check)
                local last_deployed
                last_deployed=$(echo "$service_detail" | jq -r '.metadata.annotations."serving.knative.dev/lastModifier" // "unknown"' 2>/dev/null || echo "unknown")
                
                detailed_services=$(echo "$detailed_services" | jq --argjson service "$service_detail" \
                    --arg service_url "$service_url" \
                    --arg image "$image" \
                    --arg last_deployed "$last_deployed" \
                    '. + [{
                        name: $service.metadata.name,
                        url: $service_url,
                        image: $image,
                        last_deployed: $last_deployed,
                        region: $service.metadata.labels."cloud.googleapis.com/location",
                        details: $service
                    }]' 2>/dev/null || echo "$detailed_services")
            done
        fi
        
        # Update main JSON
        local temp_file=$(mktemp)
        jq --argjson cloud_run_services "$detailed_services" \
           '.resources.cloud_run = $cloud_run_services' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    else
        log_warning "Could not retrieve Cloud Run services"
    fi
    
    log_success "Cloud Run analysis completed: $service_count services found"
}

# Analyze Container Registry images
analyze_container_registry() {
    log "🐳 Analyzing Container Registry images..."
    
    local images_json='[]'
    local image_count=0
    
    # Check GCR images
    if gcloud container images list --repository="gcr.io/$PROJECT_ID" --format=json &>/dev/null; then
        local gcr_images
        gcr_images=$(gcloud container images list --repository="gcr.io/$PROJECT_ID" --format=json 2>/dev/null || echo '[]')
        
        # Get detailed info for each image
        for image in $(echo "$gcr_images" | jq -r '.[].name' 2>/dev/null); do
            log_verbose "Analyzing image: $image"
            
            # Get image tags
            local tags
            tags=$(gcloud container images list-tags "$image" --format=json --limit=10 2>/dev/null || echo '[]')
            
            images_json=$(echo "$images_json" | jq --arg image "$image" --argjson tags "$tags" \
                '. + [{
                    name: $image,
                    repository: "gcr.io",
                    tags: $tags,
                    tag_count: ($tags | length)
                }]' 2>/dev/null || echo "$images_json")
            
            image_count=$((image_count + 1))
        done
    fi
    
    # Check Artifact Registry
    local ar_repos
    if ar_repos=$(gcloud artifacts repositories list --location="$REGION" --format=json 2>/dev/null); then
        log_verbose "Found Artifact Registry repositories: $(echo "$ar_repos" | jq 'length')"
        # Add AR analysis if needed
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson container_images "$images_json" \
       '.resources.container_registry = $container_images' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Container registry analysis completed: $image_count images found"
}

# Analyze Storage Buckets
analyze_storage_buckets() {
    log "🪣 Analyzing Cloud Storage buckets..."
    
    local buckets_json='[]'
    local bucket_count=0
    
    # Get all buckets in project
    if buckets_list=$(gsutil ls -p "$PROJECT_ID" 2>/dev/null); then
        for bucket in $buckets_list; do
            bucket_name=${bucket#gs://}
            bucket_name=${bucket_name%/}
            log_verbose "Analyzing bucket: $bucket_name"
            
            # Get bucket metadata
            local bucket_info
            bucket_info=$(gsutil ls -L -b "gs://$bucket_name" 2>/dev/null | head -20 || echo "")
            
            # Get storage class and location
            local storage_class location
            storage_class=$(echo "$bucket_info" | grep "Storage class:" | awk '{print $3}' || echo "unknown")
            location=$(echo "$bucket_info" | grep "Location constraint:" | awk '{print $3}' || echo "unknown")
            
            # Get approximate size (simplified)
            local object_count size_bytes
            object_count=$(gsutil ls "gs://$bucket_name/**" 2>/dev/null | wc -l || echo "0")
            
            # Check if bucket name matches DARWIN patterns
            local is_darwin_related="false"
            if [[ "$bucket_name" =~ darwin|biomaterials|kec ]]; then
                is_darwin_related="true"
            fi
            
            buckets_json=$(echo "$buckets_json" | jq --arg name "$bucket_name" \
                --arg storage_class "$storage_class" \
                --arg location "$location" \
                --arg object_count "$object_count" \
                --arg is_darwin_related "$is_darwin_related" \
                '. + [{
                    name: $name,
                    storage_class: $storage_class,
                    location: $location,
                    estimated_object_count: ($object_count | tonumber),
                    is_darwin_related: ($is_darwin_related == "true")
                }]' 2>/dev/null || echo "$buckets_json")
            
            bucket_count=$((bucket_count + 1))
        done
    else
        log_warning "Could not list storage buckets (may not have permission)"
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson storage_buckets "$buckets_json" \
       '.resources.storage_buckets = $storage_buckets' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Storage analysis completed: $bucket_count buckets found"
}

# Analyze BigQuery datasets
analyze_bigquery() {
    log "📊 Analyzing BigQuery datasets..."
    
    local datasets_json='[]'
    local dataset_count=0
    
    # Get all datasets
    if datasets_list=$(bq ls --format=json --project_id="$PROJECT_ID" 2>/dev/null); then
        for dataset in $(echo "$datasets_list" | jq -r '.[].datasetReference.datasetId' 2>/dev/null); do
            log_verbose "Analyzing dataset: $dataset"
            
            # Get dataset details
            local dataset_info
            dataset_info=$(bq show --format=json "$PROJECT_ID:$dataset" 2>/dev/null || echo '{}')
            
            # Get table count
            local table_count
            table_count=$(bq ls "$PROJECT_ID:$dataset" 2>/dev/null | grep -c "TABLE" || echo "0")
            
            # Check if dataset matches DARWIN patterns
            local is_darwin_related="false"
            if [[ "$dataset" =~ darwin|kec|biomaterials ]]; then
                is_darwin_related="true"
            fi
            
            datasets_json=$(echo "$datasets_json" | jq --arg name "$dataset" \
                --argjson dataset_info "$dataset_info" \
                --arg table_count "$table_count" \
                --arg is_darwin_related "$is_darwin_related" \
                '. + [{
                    name: $name,
                    table_count: ($table_count | tonumber),
                    is_darwin_related: ($is_darwin_related == "true"),
                    details: $dataset_info
                }]' 2>/dev/null || echo "$datasets_json")
            
            dataset_count=$((dataset_count + 1))
        done
    else
        log_warning "Could not list BigQuery datasets (may not have permission)"
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson bigquery_datasets "$datasets_json" \
       '.resources.bigquery_datasets = $bigquery_datasets' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "BigQuery analysis completed: $dataset_count datasets found"
}

# Analyze IAM Service Accounts
analyze_service_accounts() {
    log "👤 Analyzing IAM Service Accounts..."
    
    local sa_json='[]'
    local sa_count=0
    
    # Get all service accounts
    if sa_list=$(gcloud iam service-accounts list --project="$PROJECT_ID" --format=json 2>/dev/null); then
        for sa in $(echo "$sa_list" | jq -r '.[].email' 2>/dev/null); do
            log_verbose "Analyzing service account: $sa"
            
            # Get service account details
            local sa_info
            sa_info=$(gcloud iam service-accounts describe "$sa" --project="$PROJECT_ID" --format=json 2>/dev/null || echo '{}')
            
            # Get keys count
            local key_count
            key_count=$(gcloud iam service-accounts keys list --iam-account="$sa" --project="$PROJECT_ID" 2>/dev/null | grep -c "KEY_ID" || echo "0")
            
            # Check if SA is DARWIN related
            local is_darwin_related="false"
            if [[ "$sa" =~ darwin|vertex|biomaterials ]]; then
                is_darwin_related="true"
            fi
            
            sa_json=$(echo "$sa_json" | jq --arg email "$sa" \
                --argjson sa_info "$sa_info" \
                --arg key_count "$key_count" \
                --arg is_darwin_related "$is_darwin_related" \
                '. + [{
                    email: $email,
                    key_count: ($key_count | tonumber),
                    is_darwin_related: ($is_darwin_related == "true"),
                    details: $sa_info
                }]' 2>/dev/null || echo "$sa_json")
            
            sa_count=$((sa_count + 1))
        done
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson service_accounts "$sa_json" \
       '.resources.iam_service_accounts = $service_accounts' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Service accounts analysis completed: $sa_count accounts found"
}

# Analyze Secret Manager secrets
analyze_secrets() {
    log "🔐 Analyzing Secret Manager secrets..."
    
    local secrets_json='[]'
    local secret_count=0
    
    # Get all secrets
    if secrets_list=$(gcloud secrets list --project="$PROJECT_ID" --format=json 2>/dev/null); then
        for secret in $(echo "$secrets_list" | jq -r '.[].name' 2>/dev/null); do
            secret_name=${secret##*/}
            log_verbose "Analyzing secret: $secret_name"
            
            # Get secret details
            local secret_info
            secret_info=$(gcloud secrets describe "$secret_name" --project="$PROJECT_ID" --format=json 2>/dev/null || echo '{}')
            
            # Count versions
            local version_count
            version_count=$(gcloud secrets versions list "$secret_name" --project="$PROJECT_ID" 2>/dev/null | grep -c "STATE" || echo "0")
            
            # Check if secret is DARWIN related
            local is_darwin_related="false"
            if [[ "$secret_name" =~ darwin ]]; then
                is_darwin_related="true"
            fi
            
            secrets_json=$(echo "$secrets_json" | jq --arg name "$secret_name" \
                --argjson secret_info "$secret_info" \
                --arg version_count "$version_count" \
                --arg is_darwin_related "$is_darwin_related" \
                '. + [{
                    name: $name,
                    version_count: ($version_count | tonumber),
                    is_darwin_related: ($is_darwin_related == "true"),
                    details: $secret_info
                }]' 2>/dev/null || echo "$secrets_json")
            
            secret_count=$((secret_count + 1))
        done
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson secrets "$secrets_json" \
       '.resources.secrets_manager = $secrets' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Secrets analysis completed: $secret_count secrets found"
}

# Analyze enabled APIs
analyze_apis() {
    log "🔌 Analyzing enabled APIs..."
    
    local apis_json='[]'
    local api_count=0
    
    # Get enabled APIs
    if apis_list=$(gcloud services list --enabled --project="$PROJECT_ID" --format=json 2>/dev/null); then
        # Known DARWIN APIs
        local darwin_apis=(
            "aiplatform.googleapis.com"
            "ml.googleapis.com"
            "run.googleapis.com"
            "cloudbuild.googleapis.com"
            "secretmanager.googleapis.com"
            "bigquery.googleapis.com"
            "storage.googleapis.com"
            "monitoring.googleapis.com"
            "logging.googleapis.com"
            "domains.googleapis.com"
            "dns.googleapis.com"
        )
        
        for api in $(echo "$apis_list" | jq -r '.[].config.name' 2>/dev/null); do
            local is_darwin_api="false"
            for darwin_api in "${darwin_apis[@]}"; do
                if [[ "$api" == "$darwin_api" ]]; then
                    is_darwin_api="true"
                    break
                fi
            done
            
            apis_json=$(echo "$apis_json" | jq --arg name "$api" \
                --arg is_darwin_api "$is_darwin_api" \
                '. + [{
                    name: $name,
                    is_darwin_related: ($is_darwin_api == "true")
                }]' 2>/dev/null || echo "$apis_json")
            
            api_count=$((api_count + 1))
        done
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson apis "$apis_json" \
       '.resources.apis_enabled = $apis' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "APIs analysis completed: $api_count enabled APIs found"
}

# Analyze domain mappings
analyze_domain_mappings() {
    log "🌐 Analyzing domain mappings..."
    
    local domains_json='[]'
    local domain_count=0
    
    # Get Cloud Run domain mappings
    if domains_list=$(gcloud run domain-mappings list --region="$REGION" --format=json 2>/dev/null); then
        for domain in $(echo "$domains_list" | jq -r '.[].metadata.name' 2>/dev/null); do
            log_verbose "Analyzing domain: $domain"
            
            local domain_info
            domain_info=$(gcloud run domain-mappings describe "$domain" --region="$REGION" --format=json 2>/dev/null || echo '{}')
            
            # Check if domain is DARWIN related
            local is_darwin_related="false"
            if [[ "$domain" =~ agourakis.med.br|darwin ]]; then
                is_darwin_related="true"
            fi
            
            domains_json=$(echo "$domains_json" | jq --arg name "$domain" \
                --argjson domain_info "$domain_info" \
                --arg is_darwin_related "$is_darwin_related" \
                '. + [{
                    name: $name,
                    is_darwin_related: ($is_darwin_related == "true"),
                    details: $domain_info
                }]' 2>/dev/null || echo "$domains_json")
            
            domain_count=$((domain_count + 1))
        done
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson domains "$domains_json" \
       '.resources.domain_mappings = $domains' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Domain mappings analysis completed: $domain_count domains found"
}

# Analyze compute instances (VMs)
analyze_compute_instances() {
    log "💻 Analyzing Compute Engine instances..."
    
    local instances_json='[]'
    local instance_count=0
    
    # Get all instances
    if instances_list=$(gcloud compute instances list --project="$PROJECT_ID" --format=json 2>/dev/null); then
        instance_count=$(echo "$instances_list" | jq 'length' 2>/dev/null || echo 0)
        if [[ $instance_count -gt 0 ]]; then
            instances_json="$instances_list"
        fi
    fi
    
    # Update main JSON
    local temp_file=$(mktemp)
    jq --argjson instances "$instances_json" \
       '.resources.compute_instances = $instances' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Compute instances analysis completed: $instance_count instances found"
}

# Perform analysis and generate recommendations
perform_analysis() {
    log "🔍 Performing resource analysis and generating recommendations..."
    
    # Count total resources
    local total_resources
    total_resources=$(jq '[.resources[] | length] | add' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    
    # Identify potentially obsolete resources
    local obsolete_resources='[]'
    
    # Check for old/unused Cloud Run services
    local old_services
    old_services=$(jq '.resources.cloud_run[] | select(.last_deployed == "unknown" or .url == "none") | .name' "$OUTPUT_FILE" 2>/dev/null || echo '')
    
    # Check for empty storage buckets
    local empty_buckets
    empty_buckets=$(jq '.resources.storage_buckets[] | select(.estimated_object_count == 0) | .name' "$OUTPUT_FILE" 2>/dev/null || echo '')
    
    # Security concerns
    local security_concerns='[]'
    
    # Service accounts with many keys
    local sa_with_many_keys
    sa_with_many_keys=$(jq '.resources.iam_service_accounts[] | select(.key_count > 2) | .email' "$OUTPUT_FILE" 2>/dev/null || echo '')
    
    # Generate cleanup recommendations
    local recommendations='[
        "Review and remove unused Cloud Run services",
        "Clean up old container images in GCR",
        "Remove empty storage buckets",
        "Audit service account keys and remove unused ones",
        "Review and optimize BigQuery dataset retention policies",
        "Consolidate similar secrets in Secret Manager",
        "Review enabled APIs and disable unused ones"
    ]'
    
    # Update analysis section
    local temp_file=$(mktemp)
    jq --arg total_resources "$total_resources" \
       --argjson obsolete_resources "$obsolete_resources" \
       --argjson security_concerns "$security_concerns" \
       --argjson recommendations "$recommendations" \
       '.analysis = {
          total_resources: ($total_resources | tonumber),
          potentially_obsolete: $obsolete_resources,
          security_concerns: $security_concerns,
          cleanup_recommendations: $recommendations,
          analysis_timestamp: "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        }' "$OUTPUT_FILE" > "$temp_file" && mv "$temp_file" "$OUTPUT_FILE"
    
    log_success "Analysis completed - $total_resources total resources found"
}

# Generate human-readable summary
generate_summary() {
    log "📋 Generating inventory summary..."
    
    local summary_file="gcp_inventory_summary_$(date +%Y%m%d_%H%M%S).md"
    
    # Extract data from JSON
    local total_resources cloud_run_count bucket_count bq_datasets sa_count secrets_count
    total_resources=$(jq '.analysis.total_resources' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    cloud_run_count=$(jq '.resources.cloud_run | length' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    bucket_count=$(jq '.resources.storage_buckets | length' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    bq_datasets=$(jq '.resources.bigquery_datasets | length' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    sa_count=$(jq '.resources.iam_service_accounts | length' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    secrets_count=$(jq '.resources.secrets_manager | length' "$OUTPUT_FILE" 2>/dev/null || echo 0)
    
    cat > "$summary_file" << EOF
# GCP Resource Inventory Summary

**Analysis Date**: $(date)  
**Project**: $PROJECT_ID  
**Region**: $REGION  
**Total Resources Found**: $total_resources

## 📊 Resource Overview

### Cloud Run Services: $cloud_run_count
$(jq -r '.resources.cloud_run[] | "- **\(.name)**: \(.url // "No URL")"' "$OUTPUT_FILE" 2>/dev/null || echo "No Cloud Run services found")

### Storage Buckets: $bucket_count
$(jq -r '.resources.storage_buckets[] | "- **\(.name)**: \(.estimated_object_count) objects (\(.storage_class))"' "$OUTPUT_FILE" 2>/dev/null || echo "No storage buckets found")

### BigQuery Datasets: $bq_datasets
$(jq -r '.resources.bigquery_datasets[] | "- **\(.name)**: \(.table_count) tables"' "$OUTPUT_FILE" 2>/dev/null || echo "No BigQuery datasets found")

### Service Accounts: $sa_count
$(jq -r '.resources.iam_service_accounts[] | "- **\(.email)**: \(.key_count) keys"' "$OUTPUT_FILE" 2>/dev/null || echo "No service accounts found")

### Secrets Manager: $secrets_count
$(jq -r '.resources.secrets_manager[] | "- **\(.name)**: \(.version_count) versions"' "$OUTPUT_FILE" 2>/dev/null || echo "No secrets found")

## 🔍 Analysis Results

### DARWIN-Related Resources
$(jq -r '.resources | to_entries[] | .value[] | select(.is_darwin_related == true) | "- \(.name // .email // "Unknown")"' "$OUTPUT_FILE" 2>/dev/null | sort -u || echo "No DARWIN-specific resources identified")

### Cleanup Recommendations
$(jq -r '.analysis.cleanup_recommendations[] | "- \(.)"' "$OUTPUT_FILE" 2>/dev/null || echo "No recommendations available")

## 🎯 Next Steps

1. **Review the detailed JSON inventory**: \`$OUTPUT_FILE\`
2. **Run backup script** for critical resources
3. **Execute cleanup script** with dry-run mode first
4. **Monitor resources** after cleanup

---
Generated by GCP Inventory Analysis Script  
For cleanup operations, use: \`scripts/gcp_cleanup_legacy.sh\`
EOF
    
    log_success "Summary generated: $summary_file"
    echo ""
    echo -e "${CYAN}=== INVENTORY SUMMARY ===${NC}"
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
            --output=*)
                OUTPUT_FILE="${1#*=}"
                shift
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --help)
                echo "Usage: $0 [--project=PROJECT_ID] [--region=REGION] [--output=OUTPUT_FILE] [--verbose] [--dry-run]"
                echo ""
                echo "Options:"
                echo "  --project=PROJECT_ID  GCP Project ID to analyze"
                echo "  --region=REGION       Primary region (default: us-central1)"
                echo "  --output=OUTPUT_FILE  Output JSON file name"
                echo "  --verbose            Enable verbose logging"
                echo "  --dry-run            Analysis only, no modifications"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Execute analysis steps
    log "🚀 Starting comprehensive GCP inventory analysis..."
    
    initialize_inventory
    check_prerequisites
    analyze_project_info
    analyze_cloud_run
    analyze_container_registry
    analyze_storage_buckets
    analyze_bigquery
    analyze_service_accounts
    analyze_secrets
    analyze_apis
    analyze_domain_mappings
    analyze_compute_instances
    perform_analysis
    generate_summary
    
    # Final success message
    echo -e "${GREEN}${BOLD}
🎉 GCP INVENTORY ANALYSIS COMPLETED SUCCESSFULLY! 🎉

✅ Comprehensive resource inventory generated
✅ Analysis and recommendations provided  
✅ JSON and Markdown reports created
✅ Ready for backup and cleanup operations

📁 Files Generated:
   📄 JSON Inventory: $OUTPUT_FILE
   📋 Summary Report: gcp_inventory_summary_*.md

🎯 Next Steps:
   1. Review the inventory results
   2. Run backup script: ./scripts/gcp_backup_critical_data.sh
   3. Execute cleanup: ./scripts/gcp_cleanup_legacy.sh --dry-run

🔍 GCP RESOURCES ANALYZED AND DOCUMENTED! 🔍${NC}
"
}

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}Inventory analysis interrupted by user${NC}"; exit 130' INT TERM

# Execute main function
main "$@"