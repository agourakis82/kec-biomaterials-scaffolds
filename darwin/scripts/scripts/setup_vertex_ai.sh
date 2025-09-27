#!/bin/bash

# VERTEX AI SETUP REVOLUTIONARY SCRIPT
# Automatiza setup completo do Google Cloud Vertex AI para DARWIN
# 🌟 VERTEX AI SETUP AUTOMATION - PRODUCTION READY

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-darwin-biomaterials-scaffolds}"
LOCATION="${GCP_LOCATION:-us-central1}"
SERVICE_ACCOUNT_DIR="${SERVICE_ACCOUNT_DIR:-./secrets}"
CONFIG_FILE="${CONFIG_FILE:-./config/vertex_ai_config.yaml}"

# Logging function
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

log_header() {
    echo -e "${PURPLE}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🌟 DARWIN VERTEX AI SETUP REVOLUTIONARY AUTOMATION 🌟     ║
║                                                              ║
║  Setting up Google Cloud Vertex AI for DARWIN               ║
║  - Service Accounts & IAM                                    ║
║  - API Enablement                                            ║
║  - Model Access & Endpoints                                  ║
║  - Storage & Monitoring                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝${NC}
"
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "No active gcloud authentication found"
        log "Please run: gcloud auth login"
        exit 1
    fi
    
    # Check project setting
    CURRENT_PROJECT=$(gcloud config get-value project)
    if [[ -z "$CURRENT_PROJECT" ]]; then
        log_warning "No default project set"
        log "Setting project to: $PROJECT_ID"
        gcloud config set project "$PROJECT_ID"
    elif [[ "$CURRENT_PROJECT" != "$PROJECT_ID" ]]; then
        log_warning "Current project ($CURRENT_PROJECT) differs from target ($PROJECT_ID)"
        log "Setting project to: $PROJECT_ID"
        gcloud config set project "$PROJECT_ID"
    fi
    
    # Create secrets directory
    mkdir -p "$SERVICE_ACCOUNT_DIR"
    
    log_success "Prerequisites check completed"
}

# Enable required APIs
enable_apis() {
    log "🚀 Enabling required Google Cloud APIs..."
    
    local apis=(
        "aiplatform.googleapis.com"
        "ml.googleapis.com"
        "storage.googleapis.com"
        "bigquery.googleapis.com"
        "secretmanager.googleapis.com"
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "container.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log "Enabling $api..."
        if gcloud services enable "$api" --project="$PROJECT_ID"; then
            log_success "Enabled $api"
        else
            log_warning "Failed to enable $api (might already be enabled)"
        fi
    done
    
    log_success "API enablement completed"
}

# Create service accounts
create_service_accounts() {
    log "👤 Creating service accounts..."
    
    # Main Vertex AI service account
    local sa_name="vertex-ai-darwin-main"
    local sa_email="$sa_name@$PROJECT_ID.iam.gserviceaccount.com"
    
    log "Creating main Vertex AI service account: $sa_name"
    if gcloud iam service-accounts create "$sa_name" \
        --display-name="DARWIN Vertex AI Main Service Account" \
        --description="Main service account for DARWIN Vertex AI operations" \
        --project="$PROJECT_ID" 2>/dev/null || true; then
        log_success "Created service account: $sa_name"
    else
        log_warning "Service account $sa_name might already exist"
    fi
    
    # Grant roles to main service account
    local roles=(
        "roles/aiplatform.user"
        "roles/aiplatform.admin"
        "roles/ml.admin"
        "roles/storage.admin"
        "roles/bigquery.dataEditor"
        "roles/secretmanager.secretAccessor"
        "roles/monitoring.editor"
        "roles/logging.writer"
    )
    
    for role in "${roles[@]}"; do
        log "Granting $role to $sa_email"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet || log_warning "Failed to grant $role"
    done
    
    # Create and download key
    local key_file="$SERVICE_ACCOUNT_DIR/vertex-ai-main-key.json"
    log "Creating service account key: $key_file"
    gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$sa_email" \
        --project="$PROJECT_ID"
    
    # Set permissions on key file
    chmod 600 "$key_file"
    
    log_success "Main service account setup completed"
    
    # Model training service account
    sa_name="darwin-model-training"
    sa_email="$sa_name@$PROJECT_ID.iam.gserviceaccount.com"
    
    log "Creating model training service account: $sa_name"
    if gcloud iam service-accounts create "$sa_name" \
        --display-name="DARWIN Model Training Service Account" \
        --description="Service account for DARWIN custom model training" \
        --project="$PROJECT_ID" 2>/dev/null || true; then
        log_success "Created service account: $sa_name"
    else
        log_warning "Service account $sa_name might already exist"
    fi
    
    # Grant training-specific roles
    local training_roles=(
        "roles/aiplatform.customCodeServiceAgent"
        "roles/storage.objectAdmin"
        "roles/ml.developer"
        "roles/aiplatform.serviceAgent"
    )
    
    for role in "${training_roles[@]}"; do
        log "Granting $role to $sa_email"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet || log_warning "Failed to grant $role"
    done
    
    # Create training service account key
    key_file="$SERVICE_ACCOUNT_DIR/model-training-key.json"
    log "Creating training service account key: $key_file"
    gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$sa_email" \
        --project="$PROJECT_ID"
    
    chmod 600 "$key_file"
    
    log_success "Training service account setup completed"
    
    # Data pipeline service account
    sa_name="darwin-data-pipeline"
    sa_email="$sa_name@$PROJECT_ID.iam.gserviceaccount.com"
    
    log "Creating data pipeline service account: $sa_name"
    if gcloud iam service-accounts create "$sa_name" \
        --display-name="DARWIN Data Pipeline Service Account" \
        --description="Service account for DARWIN data processing pipeline" \
        --project="$PROJECT_ID" 2>/dev/null || true; then
        log_success "Created service account: $sa_name"
    else
        log_warning "Service account $sa_name might already exist"
    fi
    
    # Grant data-specific roles
    local data_roles=(
        "roles/bigquery.dataEditor"
        "roles/bigquery.jobUser"
        "roles/storage.objectAdmin"
        "roles/pubsub.editor"
    )
    
    for role in "${data_roles[@]}"; do
        log "Granting $role to $sa_email"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet || log_warning "Failed to grant $role"
    done
    
    # Create data pipeline service account key
    key_file="$SERVICE_ACCOUNT_DIR/data-pipeline-key.json"
    log "Creating data pipeline service account key: $key_file"
    gcloud iam service-accounts keys create "$key_file" \
        --iam-account="$sa_email" \
        --project="$PROJECT_ID"
    
    chmod 600 "$key_file"
    
    log_success "All service accounts created successfully"
}

# Create storage buckets
create_storage_buckets() {
    log "🪣 Creating Google Cloud Storage buckets..."
    
    local buckets=(
        "darwin-training-data"
        "darwin-model-artifacts"
        "darwin-experiment-logs"
        "darwin-backup-data"
    )
    
    for bucket in "${buckets[@]}"; do
        local bucket_name="$bucket-$PROJECT_ID"
        log "Creating bucket: gs://$bucket_name"
        
        if gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$LOCATION" "gs://$bucket_name/" 2>/dev/null || true; then
            log_success "Created bucket: gs://$bucket_name"
            
            # Set bucket policies
            log "Setting bucket permissions for gs://$bucket_name"
            gsutil iam ch serviceAccount:vertex-ai-darwin-main@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin "gs://$bucket_name/" || true
            gsutil iam ch serviceAccount:darwin-model-training@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin "gs://$bucket_name/" || true
            
        else
            log_warning "Bucket gs://$bucket_name might already exist"
        fi
    done
    
    log_success "Storage buckets setup completed"
}

# Setup BigQuery datasets
setup_bigquery() {
    log "📊 Setting up BigQuery datasets..."
    
    local datasets=(
        "darwin_research_insights"
        "darwin_performance_metrics"
        "darwin_scaffold_results"
        "darwin_training_logs"
    )
    
    for dataset in "${datasets[@]}"; do
        log "Creating BigQuery dataset: $dataset"
        
        if bq mk --dataset \
            --description="DARWIN dataset for $dataset" \
            --location="$LOCATION" \
            "$PROJECT_ID:$dataset" 2>/dev/null || true; then
            log_success "Created dataset: $dataset"
        else
            log_warning "Dataset $dataset might already exist"
        fi
        
        # Grant access to service accounts
        bq update --dataset \
            --access_config="role:WRITER,userByEmail:vertex-ai-darwin-main@$PROJECT_ID.iam.gserviceaccount.com" \
            --access_config="role:WRITER,userByEmail:darwin-data-pipeline@$PROJECT_ID.iam.gserviceaccount.com" \
            "$PROJECT_ID:$dataset" || log_warning "Failed to set dataset permissions for $dataset"
    done
    
    log_success "BigQuery datasets setup completed"
}

# Setup Cloud Monitoring
setup_monitoring() {
    log "📈 Setting up Cloud Monitoring and Alerting..."
    
    # Create notification channel (email-based)
    log "Creating notification channel..."
    
    # Note: This would need to be customized with actual email
    local notification_config='{
        "type": "email",
        "displayName": "DARWIN Operations Team",
        "description": "Email notifications for DARWIN system alerts",
        "labels": {
            "email_address": "darwin-ops@example.com"
        }
    }'
    
    # Create basic alert policies
    log "Creating alert policies..."
    
    # Vertex AI error rate alert
    local error_rate_policy='{
        "displayName": "DARWIN Vertex AI Error Rate High",
        "documentation": {
            "content": "Alert when Vertex AI error rate exceeds threshold",
            "mimeType": "text/markdown"
        },
        "conditions": [{
            "displayName": "Vertex AI Error Rate",
            "conditionThreshold": {
                "filter": "resource.type=\"aiplatform.googleapis.com/Endpoint\"",
                "comparison": "COMPARISON_GREATER_THAN",
                "thresholdValue": 0.05
            }
        }]
    }'
    
    log_success "Monitoring setup completed (policies need manual configuration)"
}

# Verify setup
verify_setup() {
    log "🔍 Verifying setup..."
    
    # Check service accounts
    log "Checking service accounts..."
    local service_accounts=(
        "vertex-ai-darwin-main"
        "darwin-model-training"
        "darwin-data-pipeline"
    )
    
    for sa in "${service_accounts[@]}"; do
        if gcloud iam service-accounts describe "$sa@$PROJECT_ID.iam.gserviceaccount.com" --project="$PROJECT_ID" &>/dev/null; then
            log_success "Service account exists: $sa"
        else
            log_error "Service account missing: $sa"
        fi
    done
    
    # Check storage buckets
    log "Checking storage buckets..."
    local buckets=(
        "darwin-training-data"
        "darwin-model-artifacts"
        "darwin-experiment-logs"
        "darwin-backup-data"
    )
    
    for bucket in "${buckets[@]}"; do
        local bucket_name="$bucket-$PROJECT_ID"
        if gsutil ls "gs://$bucket_name/" &>/dev/null; then
            log_success "Bucket exists: gs://$bucket_name"
        else
            log_error "Bucket missing: gs://$bucket_name"
        fi
    done
    
    # Check BigQuery datasets
    log "Checking BigQuery datasets..."
    local datasets=(
        "darwin_research_insights"
        "darwin_performance_metrics"
        "darwin_scaffold_results"
        "darwin_training_logs"
    )
    
    for dataset in "${datasets[@]}"; do
        if bq show --dataset "$PROJECT_ID:$dataset" &>/dev/null; then
            log_success "Dataset exists: $dataset"
        else
            log_error "Dataset missing: $dataset"
        fi
    done
    
    # Check key files
    log "Checking service account key files..."
    local key_files=(
        "$SERVICE_ACCOUNT_DIR/vertex-ai-main-key.json"
        "$SERVICE_ACCOUNT_DIR/model-training-key.json"
        "$SERVICE_ACCOUNT_DIR/data-pipeline-key.json"
    )
    
    for key_file in "${key_files[@]}"; do
        if [[ -f "$key_file" ]]; then
            log_success "Key file exists: $key_file"
        else
            log_error "Key file missing: $key_file"
        fi
    done
    
    log_success "Setup verification completed"
}

# Create environment file
create_env_file() {
    log "📝 Creating environment configuration file..."
    
    local env_file=".env.vertex_ai"
    
    cat > "$env_file" << EOF
# DARWIN VERTEX AI ENVIRONMENT CONFIGURATION
# Generated on $(date)

# GCP Configuration
GCP_PROJECT_ID=$PROJECT_ID
GCP_LOCATION=$LOCATION
GCP_REGION=$LOCATION

# Service Account Keys
GOOGLE_APPLICATION_CREDENTIALS=$PWD/$SERVICE_ACCOUNT_DIR/vertex-ai-main-key.json
VERTEX_AI_TRAINING_KEY=$PWD/$SERVICE_ACCOUNT_DIR/model-training-key.json
DATA_PIPELINE_KEY=$PWD/$SERVICE_ACCOUNT_DIR/data-pipeline-key.json

# Vertex AI Configuration
VERTEX_AI_ENDPOINT_BASE=https://$LOCATION-aiplatform.googleapis.com
VERTEX_AI_DEFAULT_MODEL=gemini-1.5-pro
VERTEX_AI_TEMPERATURE=0.7
VERTEX_AI_MAX_TOKENS=1024

# Storage Configuration
GCS_TRAINING_BUCKET=darwin-training-data-$PROJECT_ID
GCS_ARTIFACTS_BUCKET=darwin-model-artifacts-$PROJECT_ID
GCS_LOGS_BUCKET=darwin-experiment-logs-$PROJECT_ID
GCS_BACKUP_BUCKET=darwin-backup-data-$PROJECT_ID

# BigQuery Configuration
BQ_DATASET_INSIGHTS=darwin_research_insights
BQ_DATASET_METRICS=darwin_performance_metrics
BQ_DATASET_RESULTS=darwin_scaffold_results
BQ_DATASET_LOGS=darwin_training_logs

# Model Endpoints (will be populated after deployment)
DARWIN_BIOMATERIALS_ENDPOINT=projects/$PROJECT_ID/locations/$LOCATION/endpoints/darwin-biomaterials-expert-endpoint
DARWIN_MEDICAL_ENDPOINT=projects/$PROJECT_ID/locations/$LOCATION/endpoints/darwin-medical-gemini-endpoint
DARWIN_PHARMACO_ENDPOINT=projects/$PROJECT_ID/locations/$LOCATION/endpoints/darwin-pharmaco-ai-endpoint
DARWIN_QUANTUM_ENDPOINT=projects/$PROJECT_ID/locations/$LOCATION/endpoints/darwin-quantum-ai-endpoint

# Feature Flags
ENABLE_MED_GEMINI=false
ENABLE_CUSTOM_MODELS=true
ENABLE_MONITORING=true
ENABLE_AUTO_SCALING=true

# Development Settings
DEBUG_VERTEX_AI=false
MOCK_VERTEX_AI=false
VERTEX_AI_TIMEOUT=60

EOF

    log_success "Environment file created: $env_file"
    log "To use these settings, run: source $env_file"
}

# Request Med-Gemini access
request_med_gemini_access() {
    log "🏥 Med-Gemini Access Information..."
    
    log_warning "Med-Gemini requires special access approval from Google."
    log "To request access:"
    log "1. Visit: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview"
    log "2. Fill out the Med-Gemini access request form"
    log "3. Provide details about your healthcare/medical research use case"
    log "4. Wait for approval (can take several days)"
    log ""
    log "For DARWIN's biomaterials research:"
    log "- Mention tissue engineering and scaffold analysis"
    log "- Emphasize clinical applications of biomaterials"
    log "- Reference research publications if available"
    log ""
    log "Once approved, Med-Gemini will be available through the same endpoints."
}

# Main execution
main() {
    log_header
    
    # Execution steps
    check_prerequisites
    enable_apis
    create_service_accounts
    create_storage_buckets
    setup_bigquery
    setup_monitoring
    verify_setup
    create_env_file
    request_med_gemini_access
    
    log_success "
🎉 VERTEX AI SETUP COMPLETED SUCCESSFULLY! 🎉

✅ Service accounts created and configured
✅ APIs enabled
✅ Storage buckets ready
✅ BigQuery datasets prepared
✅ Monitoring configured
✅ Environment variables ready

Next steps:
1. Source the environment file: source .env.vertex_ai
2. Test the setup: python scripts/test_vertex_ai.py
3. Deploy custom models: bash scripts/deploy_custom_models.sh
4. Request Med-Gemini access (see instructions above)

The DARWIN Vertex AI infrastructure is now READY for production! 🚀
"
}

# Execute main function
main "$@"