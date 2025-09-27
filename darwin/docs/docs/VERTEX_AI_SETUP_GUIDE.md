# VERTEX AI SETUP GUIDE - DARWIN Production

🌟 **VERTEX AI SETUP REVOLUTIONARY GUIDE**  
Guia completo para setup e deploy do Google Cloud Vertex AI para sistema DARWIN

## 🚀 Overview

Este guia cobre o setup completo do Vertex AI para o sistema DARWIN, incluindo:
- **Med-Gemini + Gemini 1.5 Pro** access
- **Service Accounts** e autenticação
- **Custom Fine-Tuned Models** deployment
- **BigQuery + Storage** integration
- **Monitoring** e alerting
- **Production** deployment

## 📋 Prerequisites

### 1. Google Cloud Setup
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### 2. Project Setup
```bash
# Set project
export GCP_PROJECT_ID="darwin-biomaterials-scaffolds"
export GCP_LOCATION="us-central1"

gcloud config set project $GCP_PROJECT_ID
gcloud config set compute/region $GCP_LOCATION
```

### 3. Dependencies
```bash
# Install Python dependencies
pip install -r src/kec_unified_api/requirements.txt

# Additional GCP packages
pip install google-cloud-aiplatform google-cloud-storage google-cloud-bigquery
```

## ⚡ Quick Setup (Automated)

### 1. Run Automated Setup
```bash
# Make scripts executable (already done)
chmod +x scripts/setup_vertex_ai.sh scripts/test_vertex_ai.py

# Run automated setup
./scripts/setup_vertex_ai.sh
```

### 2. Load Environment Variables
```bash
# Source environment variables
source .env.vertex_ai
```

### 3. Test Setup
```bash
# Run comprehensive tests
python scripts/test_vertex_ai.py
```

## 🔧 Manual Setup (Detailed)

### 1. Enable APIs
```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable ml.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Create Service Accounts

#### Main Vertex AI Service Account
```bash
# Create service account
gcloud iam service-accounts create vertex-ai-darwin-main \
  --display-name="DARWIN Vertex AI Main Service Account" \
  --description="Main service account for DARWIN Vertex AI operations"

# Grant roles
SA_EMAIL="vertex-ai-darwin-main@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/ml.admin"

# Create and download key
mkdir -p secrets
gcloud iam service-accounts keys create secrets/vertex-ai-main-key.json \
  --iam-account=$SA_EMAIL
```

#### Training Service Account
```bash
# Create training service account
gcloud iam service-accounts create darwin-model-training \
  --display-name="DARWIN Model Training Service Account" \
  --description="Service account for DARWIN custom model training"

# Grant training-specific roles
TRAINING_SA="darwin-model-training@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$TRAINING_SA" \
  --role="roles/aiplatform.customCodeServiceAgent"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$TRAINING_SA" \
  --role="roles/storage.objectAdmin"

# Create key
gcloud iam service-accounts keys create secrets/model-training-key.json \
  --iam-account=$TRAINING_SA
```

### 3. Create Storage Buckets
```bash
# Create buckets
gsutil mb -p $GCP_PROJECT_ID -c STANDARD -l $GCP_LOCATION gs://darwin-training-data-$GCP_PROJECT_ID/
gsutil mb -p $GCP_PROJECT_ID -c STANDARD -l $GCP_LOCATION gs://darwin-model-artifacts-$GCP_PROJECT_ID/
gsutil mb -p $GCP_PROJECT_ID -c STANDARD -l $GCP_LOCATION gs://darwin-experiment-logs-$GCP_PROJECT_ID/
gsutil mb -p $GCP_PROJECT_ID -c STANDARD -l $GCP_LOCATION gs://darwin-backup-data-$GCP_PROJECT_ID/

# Set bucket permissions
gsutil iam ch serviceAccount:$SA_EMAIL:objectAdmin gs://darwin-training-data-$GCP_PROJECT_ID/
gsutil iam ch serviceAccount:$TRAINING_SA:objectAdmin gs://darwin-training-data-$GCP_PROJECT_ID/
```

### 4. Setup BigQuery Datasets
```bash
# Create datasets
bq mk --dataset --description="DARWIN research insights" --location=$GCP_LOCATION $GCP_PROJECT_ID:darwin_research_insights
bq mk --dataset --description="DARWIN performance metrics" --location=$GCP_LOCATION $GCP_PROJECT_ID:darwin_performance_metrics
bq mk --dataset --description="DARWIN scaffold results" --location=$GCP_LOCATION $GCP_PROJECT_ID:darwin_scaffold_results
bq mk --dataset --description="DARWIN training logs" --location=$GCP_LOCATION $GCP_PROJECT_ID:darwin_training_logs

# Grant dataset access
bq update --dataset \
  --access_config="role:WRITER,userByEmail:$SA_EMAIL" \
  $GCP_PROJECT_ID:darwin_research_insights
```

## 🎯 Custom Models Setup

### 1. DARWIN-BiomaterialsGPT
```python
from kec_unified_api.ai_agents.vertex_ai_fine_tuning import VertexAIFineTuningManager

# Initialize fine-tuning manager
manager = VertexAIFineTuningManager(
    project_id="darwin-biomaterials-scaffolds",
    location="us-central1"
)

await manager.initialize()

# Create biomaterials expert model
biomaterials_model = await manager.create_custom_biomaterials_model(
    training_data_path="gs://darwin-training-data/biomaterials/",
    model_name="darwin-biomaterials-expert"
)
```

### 2. DARWIN-MedicalGemini (Requires Med-Gemini Access)
```python
# Create medical expert model (requires Med-Gemini access approval)
medical_model = await manager.create_custom_medical_model(
    medical_dataset_path="gs://darwin-training-data/medical/",
    model_name="darwin-medical-gemini"
)
```

### 3. Deploy All DARWIN Models
```python
# Deploy all custom models
deployed_models = await manager.deploy_all_darwin_models()
```

## 🏥 Med-Gemini Access Request

Med-Gemini requer aprovação especial do Google. Para solicitar acesso:

### 1. Requisitos
- **Healthcare/Medical Research** use case
- **Academic or Clinical Institution** affiliation
- **Detailed project description**
- **Data privacy and security** plan

### 2. Request Process
1. Visit: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview
2. Fill out Med-Gemini access request form
3. Provide DARWIN project details:
   - **Biomaterials research** for medical applications
   - **Tissue engineering** and scaffold analysis
   - **Clinical applications** of biomaterials
   - **Research institution** affiliation

### 3. Expected Timeline
- **Review Process**: 5-15 business days
- **Approval Notification**: Via email
- **Model Access**: Available immediately after approval

## 📊 Monitoring & Alerting

### 1. Cloud Monitoring Setup
```bash
# Create notification channel
gcloud alpha monitoring channels create \
  --display-name="DARWIN Operations Team" \
  --type=email \
  --channel-labels=email_address=darwin-ops@example.com
```

### 2. Alert Policies
- **Error Rate Alert**: > 5% error rate
- **Latency Alert**: > 5 second response time
- **Quota Alert**: > 80% quota usage
- **Cost Alert**: > $500/month spend

### 3. Custom Dashboards
Access via: https://console.cloud.google.com/monitoring/dashboards

## 🧪 Testing & Validation

### 1. Basic Connectivity Test
```python
from kec_unified_api.services.vertex_ai_client import VertexAIClient

client = VertexAIClient()
await client.initialize()

# Test basic text generation
response = await client.generate_text(
    prompt="Hello, this is a test of Vertex AI integration",
    model=VertexAIModel.GEMINI_1_5_PRO
)

print(f"Response: {response.content}")
```

### 2. Comprehensive Testing
```bash
# Run full test suite
python scripts/test_vertex_ai.py

# Expected output:
# ✅ Environment Setup: PASSED
# ✅ Authentication: PASSED  
# ✅ Model Access: PASSED
# ✅ Storage Access: PASSED
# ✅ Text Generation: PASSED
```

### 3. Performance Benchmarking
```python
# Test multiple models
models_to_test = [
    VertexAIModel.GEMINI_1_5_PRO,
    VertexAIModel.GEMINI_1_5_FLASH,
    VertexAIModel.TEXT_BISON
]

for model in models_to_test:
    start_time = time.time()
    response = await client.generate_text("Test prompt", model)
    latency = (time.time() - start_time) * 1000
    print(f"{model.value}: {latency:.1f}ms")
```

## 🔧 Configuration

### 1. Environment Variables
```bash
# Core configuration
export GCP_PROJECT_ID="darwin-biomaterials-scaffolds"
export GCP_LOCATION="us-central1" 
export GOOGLE_APPLICATION_CREDENTIALS="./secrets/vertex-ai-main-key.json"

# Model configuration
export VERTEX_AI_DEFAULT_MODEL="gemini-1.5-pro"
export VERTEX_AI_TEMPERATURE="0.7"
export VERTEX_AI_MAX_TOKENS="1024"

# Storage configuration
export GCS_TRAINING_BUCKET="darwin-training-data-$GCP_PROJECT_ID"
export GCS_ARTIFACTS_BUCKET="darwin-model-artifacts-$GCP_PROJECT_ID"
```

### 2. Configuration File
Location: `config/vertex_ai_config.yaml`

Key sections:
- **Project Settings**: IDs, locations, regions
- **Service Accounts**: Credentials, roles, permissions  
- **Models**: Base models, custom models, endpoints
- **Storage**: Buckets, paths, permissions
- **Monitoring**: Metrics, alerts, dashboards

## 🚀 Production Deployment

### 1. Environment Setup
```bash
# Production environment
export ENVIRONMENT="production"
export ENABLE_MED_GEMINI="false"  # Set to true after access approval
export ENABLE_CUSTOM_MODELS="true"
export ENABLE_MONITORING="true"
```

### 2. Health Checks
```python
# Implement health check endpoint
@app.get("/health/vertex-ai")
async def vertex_ai_health():
    client = VertexAIClient()
    status = await client.get_client_status()
    return {
        "status": "healthy" if status["client_initialized"] else "unhealthy",
        "vertex_ai": status
    }
```

### 3. Load Balancing & Auto-scaling
- **Min Replicas**: 1
- **Max Replicas**: 10  
- **CPU Target**: 70%
- **Memory Target**: 80%

## 🔍 Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Fix: Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify authentication
gcloud auth list
```

#### 2. Permission Denied
```bash
# Fix: Check service account roles
gcloud projects get-iam-policy $GCP_PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:serviceAccount:$SA_EMAIL"
```

#### 3. Quota Exceeded
```bash
# Check quotas
gcloud compute project-info describe --project=$GCP_PROJECT_ID

# Request quota increase in GCP Console
```

#### 4. Model Not Available
- **Med-Gemini**: Requires access approval
- **Custom Models**: Need to be deployed first
- **Regional Availability**: Some models limited to specific regions

### Logs & Debugging
```bash
# View logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" --limit=50

# Enable debug logging
export DEBUG_VERTEX_AI=true
export VERTEX_AI_LOG_LEVEL=DEBUG
```

## 📋 Checklist

### Pre-Production Checklist
- [ ] **APIs Enabled**: All required GCP APIs
- [ ] **Service Accounts**: Created with proper roles
- [ ] **Storage Buckets**: Created with proper permissions
- [ ] **BigQuery Datasets**: Created and accessible
- [ ] **Authentication**: Service account keys working
- [ ] **Basic Testing**: Text generation working
- [ ] **Custom Models**: Training pipeline ready
- [ ] **Monitoring**: Alerts and dashboards configured
- [ ] **Documentation**: Updated and accessible

### Production Checklist  
- [ ] **Environment Variables**: Production values set
- [ ] **Security**: Service account keys secured
- [ ] **Monitoring**: Alerts tested and working
- [ ] **Performance**: Latency and throughput validated
- [ ] **Cost Monitoring**: Budget alerts configured
- [ ] **Backup**: Data backup strategy implemented
- [ ] **Load Testing**: System tested under load
- [ ] **Rollback Plan**: Deployment rollback prepared

## 🎉 Success Criteria

Setup is considered **SUCCESSFUL** when:

1. ✅ **All APIs enabled** and accessible
2. ✅ **Service accounts created** with proper permissions  
3. ✅ **Storage and BigQuery** working
4. ✅ **Text generation** working with base models
5. ✅ **Authentication** working seamlessly
6. ✅ **Monitoring** configured and alerting
7. ✅ **Tests passing** (80%+ success rate)
8. ✅ **Performance** meeting targets (<2s response time)

## 📞 Support

### Resources
- **GCP Documentation**: https://cloud.google.com/vertex-ai/docs
- **Med-Gemini Guide**: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview  
- **DARWIN Issues**: GitHub Issues
- **Emergency Contact**: darwin-ops@example.com

### Next Steps
1. **Complete Vertex AI setup** ✅
2. **Configure custom models** (next task)
3. **Setup BigQuery pipeline** (next task)
4. **Deploy to Cloud Run** (upcoming)
5. **Production testing** (final step)

---

🌟 **VERTEX AI SETUP COMPLETE - READY FOR PRODUCTION DEPLOYMENT!** 🚀