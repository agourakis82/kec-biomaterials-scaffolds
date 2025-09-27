# DARWIN GCP Permissions Fix

This document explains the service account permission issues encountered during DARWIN deployment and how to fix them.

## Issues Identified

### 1. Cloud Logging Permission Issue
**Error:** `The service account running this build projects/pcs-helio/serviceAccounts/862291807500-compute@developer.gserviceaccount.com does not have permission to write logs to Cloud Logging.`

**Root Cause:** The default Compute Engine service account used by Cloud Build lacks the `roles/logging.logWriter` role.

### 2. Cloud Run IAM Policy Issue
**Error:** `Setting IAM policy failed, try "gcloud beta run services add-iam-policy-binding --region=us-central1 --member=allUsers --role=roles/run.invoker darwin-rag"`

**Root Cause:** The deployment script wasn't properly setting the Cloud Run service to allow public access.

## Solutions Implemented

### 1. Updated Bootstrap Script (`bootstrap.sh`)
Added the missing `roles/logging.logWriter` role to the custom service account:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/logging.logWriter"
```

### 2. Updated Deploy Script (`deploy.sh`)
Added explicit IAM policy binding for public access:

```bash
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member="allUsers" \
  --role="roles/run.invoker"
```

### 3. Created Fix Script (`fix-permissions.sh`)
A standalone script to fix existing deployments without re-running the full bootstrap process.

## How to Use

### For New Deployments
1. Run the updated bootstrap script:
   ```bash
   cd infra/darwin
   bash cloud/gcp/bootstrap.sh
   ```

2. Deploy using the updated deploy script:
   ```bash
   bash cloud/gcp/deploy.sh
   ```

### For Existing Deployments (Recommended)
Run the fix script to resolve current issues:

```bash
cd infra/darwin
bash cloud/gcp/fix-permissions.sh
```

This script will:
- Add `roles/logging.logWriter` to both custom and default service accounts
- Set proper Cloud Run IAM policy for public access
- Verify the current permissions
- Display service status and URL

## Service Account Roles Summary

The `darwin-runner` service account now has the following roles:

| Role | Purpose |
|------|---------|
| `roles/run.admin` | Deploy and manage Cloud Run services |
| `roles/aiplatform.user` | Access Vertex AI services |
| `roles/secretmanager.secretAccessor` | Access secrets |
| `roles/bigquery.admin` | Manage BigQuery datasets and tables |
| `roles/logging.logWriter` | Write logs to Cloud Logging |

## Verification

After running the fix script, verify the deployment:

1. **Check service status:**
   ```bash
   gcloud run services describe darwin-rag --region=us-central1
   ```

2. **Test the service:**
   ```bash
   curl -X GET "https://darwin-rag-862291807500.us-central1.run.app/health"
   ```

3. **Check logs:**
   ```bash
   gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=darwin-rag" --limit=10
   ```

## Troubleshooting

### If you still see logging issues:
1. Ensure your application uses Google Cloud Logging libraries
2. Check that the service account is correctly attached to the Cloud Run service
3. Verify the service account has the `roles/logging.logWriter` role

### If you see IAM policy issues:
1. Run the fix script again
2. Manually set the IAM policy:
   ```bash
   gcloud run services add-iam-policy-binding darwin-rag \
     --region=us-central1 \
     --member=allUsers \
     --role=roles/run.invoker
   ```

## Security Considerations

- The service follows the principle of least privilege
- Public access is intentionally enabled for the API endpoints
- API key authentication is implemented at the application level
- Consider implementing additional security measures for production use

## Next Steps

1. Monitor the service for any remaining permission issues
2. Set up proper monitoring and alerting
3. Consider implementing structured logging in the application
4. Review and audit service account permissions regularly
