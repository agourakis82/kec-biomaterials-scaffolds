# CUSTOM MODELS GUIDE - DARWIN Specialized AI

🎯 **DARWIN CUSTOM MODELS DEPLOYMENT GUIDE**  
Guia completo para deploy e gerenciamento dos modelos IA especializados DARWIN

## 🚀 Overview

Os modelos custom DARWIN são agentes IA especializados fine-tuned para domínios específicos:

- **🧬 DARWIN-BiomaterialsGPT** - Expert em scaffolds e análise KEC
- **🏥 DARWIN-MedicalGemini** - Expert em diagnóstico clínico 
- **💊 DARWIN-PharmacoAI** - Expert em farmacologia de precisão
- **🌌 DARWIN-QuantumAI** - Expert em mecânica quântica
- **📊 DARWIN-MathematicsAI** - Expert em análise espectral
- **🧠 DARWIN-PhilosophyAI** - Expert em consciousness studies

## 📋 Prerequisites

### 1. Vertex AI Setup
```bash
# Complete Vertex AI setup first
./scripts/setup_vertex_ai.sh
source .env.vertex_ai
```

### 2. Verify Authentication
```bash
# Test Vertex AI connection
python scripts/test_vertex_ai.py
```

### 3. Training Data Preparation
```bash
# Verify training templates exist
ls -la data/training_templates/
```

## ⚡ Quick Deployment

### 1. Deploy All Models (Automated)
```bash
# Run automated deployment
./scripts/deploy_custom_models.sh

# This will:
# - Generate training datasets
# - Deploy models to Vertex AI
# - Create model registry
# - Test deployed models
```

### 2. Verify Deployment
```bash
# Check deployment status
cat custom_models_registry.json

# View deployment summary
ls -la deployment_summary_*.md
```

## 🔧 Individual Model Deployment

### DARWIN-BiomaterialsGPT
```python
from kec_unified_api.ai_agents.vertex_ai_fine_tuning import VertexAIFineTuningManager

manager = VertexAIFineTuningManager("darwin-biomaterials-scaffolds", "us-central1")
await manager.initialize()

# Deploy biomaterials expert
biomaterials_model = await manager.create_custom_biomaterials_model(
    training_data_path="gs://darwin-training-data/biomaterials/",
    model_name="darwin-biomaterials-expert"
)

print(f"Model deployed: {biomaterials_model.model_id}")
```

### DARWIN-MedicalGemini
```python
# Deploy medical expert (requires Med-Gemini access)
medical_model = await manager.create_custom_medical_model(
    medical_dataset_path="gs://darwin-training-data/medical/",
    model_name="darwin-medical-gemini"
)
```

### DARWIN-PharmacoAI
```python
# Deploy pharmacology expert
pharmaco_model = await manager.create_custom_pharmacology_model(
    pharmaco_dataset_path="gs://darwin-training-data/pharmacology/",
    model_name="darwin-pharmaco-ai"
)
```

### DARWIN-QuantumAI
```python
# Deploy quantum mechanics expert
quantum_model = await manager.create_custom_quantum_model(
    quantum_dataset_path="gs://darwin-training-data/quantum/",
    model_name="darwin-quantum-ai"
)
```

## 🧪 Testing Custom Models

### Basic Functionality Test
```python
from kec_unified_api.services.vertex_ai_client import VertexAIClient, VertexAIModel

client = VertexAIClient()
await client.initialize()

# Test biomaterials expert
response = await client.generate_text(
    prompt="What are the optimal KEC metrics for bone tissue engineering scaffolds?",
    model=VertexAIModel.DARWIN_BIOMATERIALS,
    max_tokens=200
)

print(f"Expert response: {response.content}")
```

### Domain Specialization Validation
```python
# Test domain-specific knowledge
test_cases = {
    VertexAIModel.DARWIN_BIOMATERIALS: "Analyze scaffold biocompatibility using KEC metrics",
    VertexAIModel.DARWIN_MEDICAL: "Differential diagnosis for patient with chest pain and elevated troponin",  
    VertexAIModel.DARWIN_PHARMACOLOGY: "Design precision dosing for CYP2D6 poor metabolizer",
    VertexAIModel.DARWIN_QUANTUM: "Explain quantum coherence in biomaterial scaffolds"
}

for model, prompt in test_cases.items():
    try:
        response = await client.generate_text(prompt=prompt, model=model)
        print(f"{model.value}: ✅ Specialized response generated")
    except Exception as e:
        print(f"{model.value}: ❌ Not available - {e}")
```

## 📊 Training Dataset Structure

### Biomaterials Dataset
```json
{
  "category": "KEC_Metrics_Analysis",
  "examples": [
    {
      "input": "What are optimal KEC metrics for bone scaffolds?",
      "output": "For bone scaffolds: H_spectral: 7.2-8.8, k_forman_mean: 0.25-0.45..."
    }
  ]
}
```

### Medical Dataset  
```json
{
  "category": "Clinical_Diagnosis", 
  "examples": [
    {
      "input": "Patient with chest pain and elevated troponin...",
      "output": "Differential diagnosis: NSTEMI based on elevated troponin..."
    }
  ]
}
```

### Pharmacology Dataset
```json
{
  "category": "Precision_Pharmacology",
  "examples": [
    {
      "input": "CYP2D6 poor metabolizer requiring risperidone...",
      "output": "Reduce initial dose 50-75%, enhanced EPS monitoring..."
    }
  ]
}
```

## 🎯 AutoGen Integration

### Connect Custom Models to Research Team
```python
from kec_unified_api.ai_agents.research_team import ResearchTeamCoordinator

# Initialize research team with custom models
team = ResearchTeamCoordinator()
await team.initialize()

# Use custom models in collaborative research
research_request = CollaborativeResearchRequest(
    research_question="Optimize scaffold design for neural tissue engineering",
    target_specializations=[
        AgentSpecialization.BIOMATERIALS,
        AgentSpecialization.QUANTUM_MECHANICS,
        AgentSpecialization.MATHEMATICS
    ]
)

result = await team.collaborative_research(research_request)
```

### Agent Specialization Mapping
```python
CUSTOM_MODEL_MAPPING = {
    AgentSpecialization.BIOMATERIALS: VertexAIModel.DARWIN_BIOMATERIALS,
    AgentSpecialization.CLINICAL_PSYCHIATRY: VertexAIModel.DARWIN_MEDICAL,
    AgentSpecialization.PHARMACOLOGY: VertexAIModel.DARWIN_PHARMACOLOGY,
    AgentSpecialization.QUANTUM_MECHANICS: VertexAIModel.DARWIN_QUANTUM,
    AgentSpecialization.MATHEMATICS: VertexAIModel.DARWIN_MATHEMATICS,
    AgentSpecialization.PHILOSOPHY: VertexAIModel.DARWIN_PHILOSOPHY
}
```

## 🔧 Model Management

### Check Model Status
```python
# Get all custom models status
models_status = await manager.get_custom_models_status()
print(f"Custom models: {models_status['custom_models_count']}")
```

### Update Model Endpoints
```bash
# List endpoints
gcloud ai endpoints list --region=us-central1

# Get endpoint details  
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
```

### Model Performance Monitoring
```python
# Monitor model performance
performance_metrics = client.model_metrics
print(f"Total requests: {performance_metrics['total_requests']}")
print(f"Average response time: {performance_metrics['average_response_time']:.2f}ms")
print(f"Model usage: {performance_metrics['model_usage']}")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Deployment Failed
```bash
# Check training job status
gcloud ai custom-jobs list --region=us-central1

# View training logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

#### 2. Model Not Available
```python
# Check if model endpoint is ready
available_models = await client.get_available_models()
print(f"Available custom models: {available_models['custom_models']}")
```

#### 3. Poor Model Performance
- **Insufficient Training Data**: Increase dataset size (>5000 examples recommended)
- **Domain Mismatch**: Verify training examples match target use cases
- **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs

#### 4. High Latency
- **Cold Start**: First request takes longer (warm-up effect)
- **Model Size**: Consider using smaller variants for faster inference
- **Batch Processing**: Use batch predictions for multiple requests

### Debug Commands
```bash
# Check model files
gsutil ls gs://darwin-model-artifacts-PROJECT_ID/

# View training metrics
cat training_metrics.json

# Test model connectivity
curl -X POST "ENDPOINT_URL" -H "Content-Type: application/json" -d '{"prompt": "test"}'
```

## 📈 Performance Optimization

### Model Caching
```python
# Enable model caching for faster responses
client_config = VertexAIConfig(
    project_id="darwin-biomaterials-scaffolds",
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)
```

### Batch Processing
```python
# Process multiple requests efficiently
prompts = [
    "Analyze scaffold A",
    "Analyze scaffold B", 
    "Analyze scaffold C"
]

responses = await client.batch_generate_text(
    prompts=prompts,
    model=VertexAIModel.DARWIN_BIOMATERIALS
)
```

### Auto-scaling Configuration
```yaml
# deployment.yaml
spec:
  template:
    spec:
      containerConcurrency: 10
      autoscaling.knative.dev/minScale: "1"
      autoscaling.knative.dev/maxScale: "10"
      autoscaling.knative.dev/target: "70"
```

## 🎉 Success Metrics

Deployment is considered **SUCCESSFUL** when:

1. ✅ **All models deployed** and endpoints active
2. ✅ **Domain specialization verified** (>70% relevant keywords in responses)  
3. ✅ **Performance targets met** (<2s response time, >95% uptime)
4. ✅ **AutoGen integration** working seamlessly
5. ✅ **Quality assessment** passes expert review

## 📋 Production Checklist

### Pre-Production
- [ ] **Training Data Quality**: Expert reviewed, diverse examples
- [ ] **Model Validation**: Domain experts approve responses
- [ ] **Performance Testing**: Load testing completed
- [ ] **Integration Testing**: AutoGen research team functional
- [ ] **Security Review**: Model access controls configured

### Production Deployment
- [ ] **Monitoring Setup**: Performance dashboards active
- [ ] **Auto-scaling**: Configured for expected load
- [ ] **Backup Strategy**: Model artifacts backed up
- [ ] **Rollback Plan**: Previous model version available
- [ ] **Documentation**: Updated for operations team

## 📞 Support

### Resources
- **Training Issues**: Check training logs in GCS bucket
- **Model Performance**: Monitor Vertex AI metrics console
- **Integration Problems**: Test with simple prompts first
- **Emergency Support**: darwin-models@example.com

### Next Steps
1. **Complete model deployment** ✅
2. **Test domain specialization** ✅  
3. **Integrate with AutoGen** (next task)
4. **Deploy to production** (upcoming)
5. **Monitor and optimize** (ongoing)

---

🎯 **CUSTOM MODELS READY - DARWIN AI SPECIALISTS DEPLOYED!** 🚀