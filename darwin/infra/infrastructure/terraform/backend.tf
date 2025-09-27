# =============================================================================
# DARWIN Infrastructure - Remote State Backend Configuration
# =============================================================================

# =============================================================================
# Terraform State Backend Configuration
# =============================================================================

# Note: This configuration is also defined in main.tf
# It's duplicated here for clarity and documentation purposes
# The backend block in main.tf takes precedence

# terraform {
#   backend "gcs" {
#     # Bucket name for storing Terraform state
#     bucket = "darwin-terraform-state-bucket"
#
#     # Path within the bucket to store state files
#     prefix = "terraform/state"
#
#     # Optional: Specify the location for the bucket
#     # location = "US"
#
#     # Optional: Enable versioning for state files
#     # versioning = true
#
#     # Optional: Specify encryption key
#     # encryption_key = "projects/YOUR_PROJECT_ID/locations/global/keyRings/YOUR_KEYRING/cryptoKeys/YOUR_KEY"
#   }
# }

# =============================================================================
# State Backend Initialization Resource
# =============================================================================

# This resource ensures the state bucket exists before trying to use it
# Run with: terraform apply -target=google_storage_bucket.terraform_state
resource "google_storage_bucket" "terraform_state" {
  name     = "darwin-terraform-state-bucket"
  location = "US"
  project  = var.project_id
  
  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
  
  # Enable versioning for state file history
  versioning {
    enabled = true
  }
  
  # Encryption at rest

  
  # Uniform bucket-level access
  uniform_bucket_level_access = true
  
  # Retention policy (optional)
  retention_policy {
    retention_period = 2592000 # 30 days
  }
  
  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  # Version lifecycle management
  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }
  
  # Labels
  labels = {
    environment = var.environment
    purpose     = "terraform-state"
    team        = "kec-biomaterials"
    managed_by  = "terraform"
  }
}

# =============================================================================
# State Bucket IAM Configuration
# =============================================================================

# IAM policy for the state bucket
resource "google_storage_bucket_iam_binding" "terraform_state_admin" {
  bucket = google_storage_bucket.terraform_state.name
  role   = "roles/storage.admin"
  
  members = [
    "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
    # Add other service accounts or users as needed
    # "user:your-email@domain.com",
    # "serviceAccount:terraform@${var.project_id}.iam.gserviceaccount.com"
  ]
  
  depends_on = [google_storage_bucket.terraform_state]
}

# =============================================================================
# State Locking (using Cloud Storage)
# =============================================================================

# Cloud Storage doesn't support native locking, but we can implement
# a simple locking mechanism using object metadata

resource "google_storage_bucket_object" "terraform_lock_info" {
  name   = "terraform.lock.info"
  bucket = google_storage_bucket.terraform_state.name
  content = jsonencode({
    created_at = timestamp()
    created_by = "terraform"
    purpose    = "state_locking_info"
    environment = var.environment
  })
  
  # Metadata
  metadata = {
    purpose     = "terraform-lock"
    environment = var.environment
  }
  
  depends_on = [google_storage_bucket.terraform_state]
}

# =============================================================================
# Backup Configuration for State
# =============================================================================

# Additional backup bucket for state files
resource "google_storage_bucket" "terraform_state_backup" {
  count    = var.enable_backup ? 1 : 0
  name     = "darwin-terraform-state-backup-${var.environment}"
  location = "US"
  project  = var.project_id
  
  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
  
  # Enable versioning
  versioning {
    enabled = true
  }
  
  # Uniform bucket-level access
  uniform_bucket_level_access = true
  
  # Labels
  labels = {
    environment = var.environment
    purpose     = "terraform-state-backup"
    team        = "kec-biomaterials"
    managed_by  = "terraform"
  }
}

# =============================================================================
# State Bucket Monitoring
# =============================================================================

# Log sink for state bucket access
resource "google_logging_project_sink" "terraform_state_audit" {
  name        = "terraform-state-audit"
  destination = "storage.googleapis.com/${google_storage_bucket.terraform_state.name}"
  
  filter = <<EOF
resource.type="gcs_bucket"
resource.labels.bucket_name="${google_storage_bucket.terraform_state.name}"
EOF

  unique_writer_identity = true
  
  depends_on = [google_storage_bucket.terraform_state]
}

# Grant the logging service account write access to the bucket
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = google_logging_project_sink.terraform_state_audit.writer_identity
  
  depends_on = [google_logging_project_sink.terraform_state_audit]
}

# =============================================================================
# State Bucket Outputs
# =============================================================================

output "terraform_state_bucket" {
  description = "Terraform state bucket name"
  value       = google_storage_bucket.terraform_state.name
}

output "terraform_state_bucket_url" {
  description = "Terraform state bucket URL"
  value       = google_storage_bucket.terraform_state.url
}

output "terraform_state_backup_bucket" {
  description = "Terraform state backup bucket name"
  value       = var.enable_backup ? google_storage_bucket.terraform_state_backup[0].name : null
}