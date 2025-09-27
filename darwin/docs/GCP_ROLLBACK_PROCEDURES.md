# GCP ROLLBACK PROCEDURES - DARWIN PROJECT

**Document Version**: 1.0.0  
**Created**: 2025-09-22  
**Project**: DARWIN Biomaterials Scaffolds  
**Purpose**: Comprehensive rollback procedures for GCP cleanup and deployment operations

## 📋 Table of Contents

1. [Emergency Rollback Quick Reference](#emergency-rollback-quick-reference)
2. [Pre-Rollback Checklist](#pre-rollback-checklist)
3. [Cloud Run Service Rollback](#cloud-run-service-rollback)
4. [Container Images Restoration](#container-images-restoration)
5. [BigQuery Data Restoration](#bigquery-data-restoration)
6. [Storage Buckets Restoration](#storage-buckets-restoration)
7. [Secrets Manager Restoration](#secrets-manager-restoration)
8. [IAM Service Accounts Restoration](#iam-service-accounts-restoration)
9. [DNS and Domain Mappings Restoration](#dns-and-domain-mappings-restoration)
10. [Complete System Rollback](#complete-system-rollback)
11. [Troubleshooting](#troubleshooting)
12. [Emergency Contacts](#emergency-contacts)

---

## 🚨 Emergency Rollback Quick Reference

### Immediate Actions (Critical Failure)
```bash
# 1. Stop all running cleanup operations
pkill -f gcp_cleanup_legacy.sh

# 2. Restore critical services immediately
gcloud run deploy darwin-backend-api \
  --image=gcr.io/pcs-helio/darwin-backend:previous \
  --region=us-central1

gcloud run deploy darwin-frontend-web \
  --image=gcr.io/pcs-helio/darwin-frontend:previous \
  --region=us-central1

# 3. Check service status
gcloud run services list --region=us-central1
```

### Emergency Service URLs
- **Production Backend**: https://api.agourakis.med.br
- **Production Frontend**: https://darwin.agourakis.med.br
- **Backup Backend**: [Cloud Run URL from deployment]
- **Backup Frontend**: [Cloud Run URL from deployment]

---

## ✅ Pre-Rollback Checklist

### Before Starting Rollback

- [ ] **Identify the Issue**: Document what went wrong and when
- [ ] **Assess Impact**: Determine scope of services/data affected
- [ ] **Locate Backups**: Confirm backup files are accessible
- [ ] **Check Dependencies**: Identify dependent systems
- [ ] **Notify Stakeholders**: Inform relevant teams about rollback
- [ ] **Prepare Environment**: Set up clean working environment

### Required Information

- **Backup Directory**: Location of backup files
- **Deployment ID**: From failed deployment
- **Time Range**: When the issue occurred
- **Affected Resources**: List of impacted GCP resources
- **Previous Working State**: Last known good configuration

### Required Tools

```bash
# Verify required tools are available
command -v gcloud || echo "Install gcloud CLI"
command -v gsutil || echo "Install gsutil" 
command -v bq || echo "Install bq CLI"
command -v jq || echo "Install jq"
command -v curl || echo "Install curl"
```

---

## ☁️ Cloud Run Service Rollback

### Scenario 1: Service Deployment Failed

**Symptoms**: Service not responding, 503 errors, deployment stuck

**Quick Fix**:
```bash
# Rollback to previous revision
gcloud run services update-traffic darwin-backend-api \
  --to-revisions=LATEST=0,previous-revision=100 \
  --region=us-central1

# Or deploy previous image
gcloud run deploy darwin-backend-api \
  --image=gcr.io/pcs-helio/darwin-backend:previous \
  --region=us-central1
```

**Complete Rollback**:
```bash
#!/bin/bash
# Cloud Run Service Rollback Script

PROJECT_ID="pcs-helio"
REGION="us-central1"
SERVICES=("darwin-backend-api" "darwin-frontend-web")

for service in "${SERVICES[@]}"; do
    echo "Rolling back service: $service"
    
    # List revisions
    gcloud run revisions list --service=$service --region=$REGION
    
    # Get previous revision
    PREVIOUS_REV=$(gcloud run revisions list \
        --service=$service --region=$REGION \
        --format="value(metadata.name)" --limit=2 | tail -1)
    
    if [[ -n "$PREVIOUS_REV" ]]; then
        # Rollback to previous revision
        gcloud run services update-traffic $service \
            --to-revisions=$PREVIOUS_REV=100 \
            --region=$REGION
        
        echo "✅ Rolled back $service to revision: $PREVIOUS_REV"
    else
        echo "❌ No previous revision found for $service"
    fi
done
```

### Scenario 2: Service Accidentally Deleted

**Recovery Steps**:
1. **Check Backup Configuration**: Locate backup deployment config
2. **Recreate Service**: Deploy from backup image
3. **Restore Configuration**: Apply environment variables and settings
4. **Test Functionality**: Verify service is working

```bash
# Recreate deleted service
gcloud run deploy darwin-backend-api \
  --image=gcr.io/pcs-helio/darwin-backend:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8090 \
  --memory=4Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=20 \
  --set-env-vars="ENVIRONMENT=production,CORS_ORIGINS=https://darwin.agourakis.med.br"
```

---

## 🐳 Container Images Restoration

### Scenario 1: Images Accidentally Deleted

**Recovery from Registry**:
```bash
# Check if images still exist
gcloud container images list --repository=gcr.io/pcs-helio

# If images exist, retag them
gcloud container images add-tag \
  gcr.io/pcs-helio/darwin-backend:backup \
  gcr.io/pcs-helio/darwin-backend:latest

# Rebuild if completely lost
cd /path/to/source
docker build -t gcr.io/pcs-helio/darwin-backend:restored .
docker push gcr.io/pcs-helio/darwin-backend:restored
```

**Recovery from Backup**:
```bash
# If backup images exist locally
docker load < darwin-backend-backup.tar
docker tag darwin-backend:backup gcr.io/pcs-helio/darwin-backend:restored
docker push gcr.io/pcs-helio/darwin-backend:restored
```

### Scenario 2: Corrupted Images

**Steps**:
1. **Identify Last Good Image**: Check deployment history
2. **Rebuild from Source**: Use backup source code
3. **Verify Integrity**: Test image functionality
4. **Deploy Fixed Version**: Update services

---

## 📊 BigQuery Data Restoration

### Scenario 1: Dataset Accidentally Deleted

**Recovery from Backup**:
```bash
#!/bin/bash
# BigQuery Dataset Restoration Script

PROJECT_ID="pcs-helio"
BACKUP_DIR="/path/to/backup/bigquery"

# Recreate dataset
bq mk --dataset \
  --description="DARWIN dataset restored from backup" \
  --location=us-central1 \
  $PROJECT_ID:darwin_research_insights

# Restore tables from backup
cd $BACKUP_DIR
for backup_file in darwin_research_insights_*.json; do
    table_name=$(echo $backup_file | sed 's/darwin_research_insights_\(.*\)\.json/\1/')
    
    echo "Restoring table: $table_name"
    bq load --source_format=NEWLINE_DELIMITED_JSON \
        $PROJECT_ID:darwin_research_insights.$table_name \
        $backup_file
done
```

### Scenario 2: Data Corruption

**Recovery Steps**:
1. **Stop Data Ingestion**: Prevent further corruption
2. **Identify Corrupt Data Range**: Find time range of bad data
3. **Restore from Backup**: Import clean backup data
4. **Validate Data Integrity**: Run data quality checks

```bash
# Delete corrupt data in time range
bq query --use_legacy_sql=false "
DELETE FROM \`$PROJECT_ID.darwin_research_insights.insights\`
WHERE timestamp BETWEEN '2025-09-22 10:00:00' AND '2025-09-22 12:00:00'
"

# Restore from backup for that time range
bq load --source_format=NEWLINE_DELIMITED_JSON \
    $PROJECT_ID:darwin_research_insights.insights \
    backup_20250922_clean.json
```

---

## 🪣 Storage Buckets Restoration

### Scenario 1: Bucket Accidentally Deleted

**Recovery Process**:
```bash
#!/bin/bash
# Storage Bucket Restoration Script

PROJECT_ID="pcs-helio"
BACKUP_DIR="/path/to/backup/storage"

# Recreate bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central1 \
  gs://darwin-training-data-$PROJECT_ID

# Set bucket permissions
gsutil iam ch serviceAccount:vertex-ai-darwin-main@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin \
  gs://darwin-training-data-$PROJECT_ID

# Restore files from backup
cd $BACKUP_DIR/darwin-training-data-$PROJECT_ID
gsutil -m cp -r * gs://darwin-training-data-$PROJECT_ID/

echo "✅ Bucket restored: gs://darwin-training-data-$PROJECT_ID"
```

### Scenario 2: Critical Files Deleted

**Individual File Recovery**:
```bash
# Check if files exist in backup
ls -la /path/to/backup/storage/darwin-training-data-*/

# Restore specific files
gsutil cp /path/to/backup/storage/critical-file.json \
  gs://darwin-training-data-$PROJECT_ID/

# Verify restoration
gsutil ls gs://darwin-training-data-$PROJECT_ID/critical-file.json
```

---

## 🔐 Secrets Manager Restoration

### Scenario 1: Secrets Accidentally Deleted

**Recovery Process**:
```bash
#!/bin/bash
# Secrets Restoration Script

PROJECT_ID="pcs-helio"
BACKUP_DIR="/path/to/backup/secrets"

cd $BACKUP_DIR

# Recreate secrets from backup metadata
for metadata_file in *_metadata.json; do
    secret_name=$(echo $metadata_file | sed 's/_metadata.json//')
    
    echo "Recreating secret: $secret_name"
    
    # Create secret structure
    gcloud secrets create $secret_name \
        --replication-policy=automatic \
        --project=$PROJECT_ID
    
    # Restore IAM policies
    if [[ -f "${secret_name}_iam.json" ]]; then
        gcloud secrets set-iam-policy $secret_name \
            ${secret_name}_iam.json \
            --project=$PROJECT_ID
    fi
    
    echo "⚠️  Secret values must be manually restored: $secret_name"
done
```

**Manual Secret Values Restoration**:
```bash
# Restore secret values (requires manual input for security)
echo "new_api_key_value" | gcloud secrets versions add darwin-openai-api-key \
  --data-file=- --project=$PROJECT_ID
```

### Scenario 2: Secret Values Corrupted

**Recovery Steps**:
1. **Identify Previous Version**: List secret versions
2. **Activate Previous Version**: Switch to working version
3. **Test Functionality**: Verify applications work

```bash
# List secret versions
gcloud secrets versions list darwin-openai-api-key --project=$PROJECT_ID

# Reactivate previous version
gcloud secrets versions enable VERSION_ID --secret=darwin-openai-api-key --project=$PROJECT_ID
```

---

## 👤 IAM Service Accounts Restoration

### Scenario 1: Service Account Deleted

**Recovery Process**:
```bash
#!/bin/bash
# Service Account Restoration Script

PROJECT_ID="pcs-helio"
BACKUP_DIR="/path/to/backup/iam"

# Recreate service account
gcloud iam service-accounts create vertex-ai-darwin-main \
    --display-name="DARWIN Vertex AI Main Service Account" \
    --description="Restored from backup" \
    --project=$PROJECT_ID

# Restore IAM policies from backup
if [[ -f "$BACKUP_DIR/project_iam_policy.json" ]]; then
    gcloud projects set-iam-policy $PROJECT_ID \
        $BACKUP_DIR/project_iam_policy.json
fi

# Restore individual service account policies
cd $BACKUP_DIR/service_account_policies
for policy_file in *_policy.json; do
    sa_email=$(echo $policy_file | sed 's/_policy.json//' | tr '_' '@' | sed 's/@/@/')
    
    echo "Restoring policy for: $sa_email"
    gcloud iam service-accounts set-iam-policy $sa_email $policy_file
done
```

### Scenario 2: Permissions Issues

**Recovery Steps**:
```bash
# Grant essential roles to service account
SA_EMAIL="vertex-ai-darwin-main@$PROJECT_ID.iam.gserviceaccount.com"

ROLES=(
    "roles/aiplatform.user"
    "roles/storage.admin"
    "roles/secretmanager.secretAccessor"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role"
done
```

---

## 🌐 DNS and Domain Mappings Restoration

### Scenario 1: Domain Mapping Deleted

**Recovery Process**:
```bash
# Recreate domain mappings
DOMAINS=("api.agourakis.med.br" "darwin.agourakis.med.br")
SERVICES=("darwin-backend-api" "darwin-frontend-web")

for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    service="${SERVICES[$i]}"
    
    echo "Restoring domain mapping: $domain -> $service"
    
    gcloud run domain-mappings create \
        --service=$service \
        --domain=$domain \
        --region=us-central1 \
        --quiet
done
```

### Scenario 2: DNS Issues

**Recovery Steps**:
1. **Check DNS Records**: Verify CNAME records are correct
2. **Re-verify Domain**: Re-verify domain ownership
3. **Wait for Propagation**: Allow 24-48 hours for DNS propagation

```bash
# Check current DNS resolution
nslookup api.agourakis.med.br
nslookup darwin.agourakis.med.br

# Verify domain mappings
gcloud run domain-mappings list --region=us-central1
```

---

## 🔄 Complete System Rollback

### Full System Recovery Process

**Step 1: Assessment and Preparation**
```bash
#!/bin/bash
# Complete System Rollback - Assessment Phase

echo "🔍 Starting complete system assessment..."

# Check current state
gcloud run services list --region=us-central1
gsutil ls -p pcs-helio
bq ls --project_id=pcs-helio

# Locate backups
BACKUP_BASE_DIR="/path/to/backup"
if [[ -d "$BACKUP_BASE_DIR" ]]; then
    echo "✅ Backup directory found: $BACKUP_BASE_DIR"
    ls -la "$BACKUP_BASE_DIR"
else
    echo "❌ Backup directory not found!"
    exit 1
fi
```

**Step 2: Service Restoration**
```bash
# Restore Cloud Run services first (highest priority)
./restore_cloud_run_services.sh

# Restore domain mappings
./restore_domain_mappings.sh

# Restore secrets (structure only)
./restore_secrets_structure.sh
```

**Step 3: Data Restoration**
```bash
# Restore BigQuery data
./restore_bigquery_data.sh

# Restore critical storage buckets
./restore_storage_buckets.sh

# Restore IAM configuration
./restore_iam_configuration.sh
```

**Step 4: Validation and Testing**
```bash
# Run comprehensive validation
./validate_system_restoration.sh

# Test critical functionality
curl https://api.agourakis.med.br/health
curl https://darwin.agourakis.med.br

# Monitor for 30 minutes
./monitor_restored_system.sh
```

---

## 🔧 Troubleshooting

### Common Rollback Issues

#### Issue 1: "Service Not Found" Error
**Symptoms**: gcloud commands return "service not found"
**Cause**: Service was completely deleted
**Solution**:
```bash
# Check if service actually exists
gcloud run services list --region=us-central1 | grep darwin

# If not found, recreate from backup
gcloud run deploy darwin-backend-api \
  --image=gcr.io/pcs-helio/darwin-backend:backup \
  --region=us-central1 \
  [other-config-options]
```

#### Issue 2: Permission Denied During Restoration
**Symptoms**: "Access denied" or "Insufficient permissions"
**Cause**: Service account lacks necessary roles
**Solution**:
```bash
# Check current user permissions
gcloud auth list
gcloud projects get-iam-policy pcs-helio

# Grant necessary roles
gcloud projects add-iam-policy-binding pcs-helio \
  --member="user:$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" \
  --role="roles/editor"
```

#### Issue 3: Data Corruption After Restoration
**Symptoms**: Applications show errors, data inconsistency
**Cause**: Backup data is corrupted or incomplete
**Solution**:
```bash
# Validate backup integrity
cd /path/to/backup
find . -name "*.json" -exec jq . {} \; > /dev/null

# Re-restore from alternative backup
./restore_from_alternative_backup.sh
```

#### Issue 4: SSL Certificate Issues
**Symptoms**: HTTPS not working for custom domains
**Cause**: SSL certificates need time to provision
**Solution**:
```bash
# Check certificate status
gcloud run domain-mappings describe api.agourakis.med.br --region=us-central1

# Wait 24-48 hours for automatic provisioning
# Or delete and recreate mapping
gcloud run domain-mappings delete api.agourakis.med.br --region=us-central1
# Then recreate...
```

### Rollback Failure Recovery

If rollback itself fails:

1. **Document the Issue**: Record exact error messages
2. **Try Alternative Approach**: Use different restoration method
3. **Partial Rollback**: Restore critical services first
4. **Seek Expert Help**: Contact GCP support if needed
5. **Manual Recovery**: Manually recreate resources if needed

### Health Checks After Rollback

```bash
#!/bin/bash
# Post-Rollback Health Check Script

echo "🏥 Running post-rollback health checks..."

# Service health checks
services=("https://api.agourakis.med.br/health" "https://darwin.agourakis.med.br")
for url in "${services[@]}"; do
    if curl -f "$url" --max-time 30; then
        echo "✅ $url: Healthy"
    else
        echo "❌ $url: Unhealthy"
    fi
done

# Data integrity checks
echo "Checking BigQuery data..."
bq query --use_legacy_sql=false "
SELECT COUNT(*) as record_count 
FROM \`pcs-helio.darwin_research_insights.insights\` 
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
"

# Storage checks
echo "Checking critical storage..."
gsutil ls gs://darwin-training-data-pcs-helio/ | head -10

echo "🏥 Health check completed"
```

---

## 📞 Emergency Contacts

### Internal Team
- **Technical Lead**: demetrios@agourakis.med.br
- **Operations Team**: ops@agourakis.med.br
- **Development Team**: dev@agourakis.med.br

### External Support
- **Google Cloud Support**: [Support Case Portal]
- **DNS Provider Support**: [Domain Provider Support]
- **Monitoring Service**: [Monitoring Provider Support]

### Emergency Escalation
1. **Level 1**: Technical team member
2. **Level 2**: Technical lead + Operations team
3. **Level 3**: Management + External GCP support
4. **Level 4**: Executive leadership + Premium support

---

## 📋 Rollback Checklist Templates

### Quick Rollback Checklist
- [ ] **Stop the Problem**: Kill running processes causing issues
- [ ] **Assess Damage**: Identify what's broken/missing  
- [ ] **Locate Backups**: Find and verify backup files
- [ ] **Restore Critical Services**: Get main services running
- [ ] **Verify Functionality**: Test that restoration worked
- [ ] **Monitor System**: Watch for issues post-rollback
- [ ] **Document Incident**: Record what happened and how it was fixed

### Complete Rollback Checklist
- [ ] **Pre-Rollback Assessment**
  - [ ] Document current broken state
  - [ ] Identify scope of needed rollback
  - [ ] Locate all required backups
  - [ ] Prepare rollback environment
- [ ] **Infrastructure Rollback**
  - [ ] Restore Cloud Run services
  - [ ] Restore domain mappings  
  - [ ] Restore container images
  - [ ] Validate infrastructure
- [ ] **Data Rollback**
  - [ ] Restore BigQuery datasets
  - [ ] Restore Storage buckets
  - [ ] Restore Secrets (structure)
  - [ ] Validate data integrity
- [ ] **Security Rollback**
  - [ ] Restore service accounts
  - [ ] Restore IAM policies
  - [ ] Restore access controls
  - [ ] Validate security settings
- [ ] **Validation and Testing**
  - [ ] Run health checks
  - [ ] Test critical functionality
  - [ ] Performance testing
  - [ ] User acceptance testing
- [ ] **Post-Rollback Activities**
  - [ ] Monitor system stability
  - [ ] Update documentation
  - [ ] Incident post-mortem
  - [ ] Process improvements

---

## 📚 Related Documentation

- **GCP Cleanup Script**: `scripts/gcp_cleanup_legacy.sh --help`
- **Backup Script**: `scripts/gcp_backup_critical_data.sh --help`
- **Deploy Script**: `scripts/deploy_darwin_production_optimized.sh --help`
- **Test Suite**: `scripts/test_scripts_dry_run.sh --help`

---

## 📄 Document Maintenance

**Last Updated**: 2025-09-22  
**Review Schedule**: Monthly  
**Update Triggers**: Major deployment changes, incident learnings  
**Owner**: DARWIN Technical Team  

**Version History**:
- v1.0.0 (2025-09-22): Initial comprehensive rollback procedures

---

**⚠️ IMPORTANT**: This document should be reviewed and updated after every major incident or system change. Keep backup procedures tested and current.

**🔒 SECURITY NOTE**: Never include actual secret values or sensitive data in this documentation. Always reference secure storage locations.