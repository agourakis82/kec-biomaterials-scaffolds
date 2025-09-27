# =============================================================================
# DARWIN Security Module - Main Configuration
# Service Accounts, IAM, Secrets Management, Security Policies
# =============================================================================

# =============================================================================
# Local Values
# =============================================================================

locals {
  # Service account names
  service_accounts = {
    terraform    = "${var.project_name}-${var.environment}-terraform-sa"
    backend      = "${var.project_name}-${var.environment}-backend-sa"
    frontend     = "${var.project_name}-${var.environment}-frontend-sa"
    monitoring   = "${var.project_name}-${var.environment}-monitoring-sa"
    cicd         = "${var.project_name}-${var.environment}-cicd-sa"
    database     = "${var.project_name}-${var.environment}-database-sa"
  }
  
  # Secret names
  secrets = {
    database_url     = "${var.project_name}-${var.environment}-database-url"
    database_password = "${var.project_name}-${var.environment}-database-password"
    redis_url        = "${var.project_name}-${var.environment}-redis-url"
    redis_password   = "${var.project_name}-${var.environment}-redis-password"
    jwt_secret       = "${var.project_name}-${var.environment}-jwt-secret"
    api_keys         = "${var.project_name}-${var.environment}-api-keys"
    encryption_key   = "${var.project_name}-${var.environment}-encryption-key"
  }
}

# =============================================================================
# KMS Key Ring and Keys for Encryption
# =============================================================================

resource "google_kms_key_ring" "main" {
  name     = "${var.project_name}-${var.environment}-keyring"
  location = var.kms_location
  project  = var.project_id
}

resource "google_kms_crypto_key" "database_key" {
  name     = "database-encryption-key"
  key_ring = google_kms_key_ring.main.id
  purpose  = "ENCRYPT_DECRYPT"
  
  rotation_period = var.key_rotation_period
  
  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
  
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "storage_key" {
  name     = "storage-encryption-key"
  key_ring = google_kms_key_ring.main.id
  purpose  = "ENCRYPT_DECRYPT"
  
  rotation_period = var.key_rotation_period
  
  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
  
  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "backup_key" {
  name     = "backup-encryption-key"
  key_ring = google_kms_key_ring.main.id
  purpose  = "ENCRYPT_DECRYPT"
  
  rotation_period = var.key_rotation_period
  
  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
  
  lifecycle {
    prevent_destroy = true
  }
}

# =============================================================================
# Service Accounts with Least Privilege
# =============================================================================

# Terraform Service Account
resource "google_service_account" "terraform" {
  account_id   = local.service_accounts.terraform
  display_name = "DARWIN Terraform Service Account"
  description  = "Service account for Terraform infrastructure management"
  project      = var.project_id
}

# Backend Application Service Account
resource "google_service_account" "backend" {
  account_id   = local.service_accounts.backend
  display_name = "DARWIN Backend Service Account"
  description  = "Service account for backend application with minimal required permissions"
  project      = var.project_id
}

# Frontend Application Service Account
resource "google_service_account" "frontend" {
  account_id   = local.service_accounts.frontend
  display_name = "DARWIN Frontend Service Account"
  description  = "Service account for frontend application with read-only permissions"
  project      = var.project_id
}

# Monitoring Service Account
resource "google_service_account" "monitoring" {
  account_id   = local.service_accounts.monitoring
  display_name = "DARWIN Monitoring Service Account"
  description  = "Service account for monitoring and alerting systems"
  project      = var.project_id
}

# CI/CD Service Account
resource "google_service_account" "cicd" {
  account_id   = local.service_accounts.cicd
  display_name = "DARWIN CI/CD Service Account"
  description  = "Service account for Cloud Build and deployment pipelines"
  project      = var.project_id
}

# Database Service Account
resource "google_service_account" "database" {
  account_id   = local.service_accounts.database
  display_name = "DARWIN Database Service Account"
  description  = "Service account for database operations and backups"
  project      = var.project_id
}

# =============================================================================
# IAM Policy Bindings with Least Privilege
# =============================================================================

# Terraform Service Account Roles
resource "google_project_iam_member" "terraform_roles" {
  for_each = toset([
    "roles/compute.admin",
    "roles/run.admin",
    "roles/cloudsql.admin",
    "roles/redis.admin",
    "roles/storage.admin",
    "roles/secretmanager.admin",
    "roles/monitoring.admin",
    "roles/logging.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountKeyAdmin",
    "roles/serviceusage.serviceUsageAdmin"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.terraform.email}"
}

# Backend Service Account Roles (Minimal)
resource "google_project_iam_member" "backend_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/redis.editor",
    "roles/storage.objectAdmin",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent",
    "roles/aiplatform.user"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# Frontend Service Account Roles (Read-only)
resource "google_project_iam_member" "frontend_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

# Monitoring Service Account Roles
resource "google_project_iam_member" "monitoring_roles" {
  for_each = toset([
    "roles/monitoring.admin",
    "roles/logging.admin",
    "roles/cloudsql.viewer",
    "roles/redis.viewer",
    "roles/run.viewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.monitoring.email}"
}

# CI/CD Service Account Roles
resource "google_project_iam_member" "cicd_roles" {
  for_each = toset([
    "roles/run.admin",
    "roles/storage.admin",
    "roles/cloudbuild.builds.builder",
    "roles/servicemanagement.serviceController",
    "roles/container.developer",
    "roles/iam.serviceAccountUser"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

# Database Service Account Roles
resource "google_project_iam_member" "database_roles" {
  for_each = toset([
    "roles/cloudsql.admin",
    "roles/storage.objectAdmin",
    "roles/logging.logWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.database.email}"
}

# =============================================================================
# KMS IAM Permissions
# =============================================================================

# Backend service account KMS access
resource "google_kms_crypto_key_iam_member" "backend_database_key" {
  crypto_key_id = google_kms_crypto_key.database_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_kms_crypto_key_iam_member" "backend_storage_key" {
  crypto_key_id = google_kms_crypto_key.storage_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.backend.email}"
}

# Database service account KMS access
resource "google_kms_crypto_key_iam_member" "database_backup_key" {
  crypto_key_id = google_kms_crypto_key.backup_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.database.email}"
}

# =============================================================================
# Secrets Management
# =============================================================================

# Database password secret
resource "random_password" "database_password" {
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true
}

resource "google_secret_manager_secret" "database_password" {
  secret_id = local.secrets.database_password
  project   = var.project_id
  
  replication {
    automatic = true
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "database_password" {
  secret      = google_secret_manager_secret.database_password.id
  secret_data = random_password.database_password.result
}

# JWT secret
resource "random_password" "jwt_secret" {
  length  = 64
  special = true
  upper   = true
  lower   = true
  numeric = true
}

resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = local.secrets.jwt_secret
  project   = var.project_id
  
  replication {
    automatic = true
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = random_password.jwt_secret.result
}

# API keys secret (placeholder for external APIs)
resource "google_secret_manager_secret" "api_keys" {
  secret_id = local.secrets.api_keys
  project   = var.project_id
  
  replication {
    automatic = true
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "api_keys" {
  secret = google_secret_manager_secret.api_keys.id
  secret_data = jsonencode({
    openai_api_key     = "sk-placeholder"
    anthropic_api_key  = "placeholder"
    google_ai_key      = "placeholder"
    external_api_key   = "placeholder"
  })
}

# Encryption key for application-level encryption
resource "random_password" "encryption_key" {
  length  = 32
  special = false
  upper   = true
  lower   = true
  numeric = true
}

resource "google_secret_manager_secret" "encryption_key" {
  secret_id = local.secrets.encryption_key
  project   = var.project_id
  
  replication {
    automatic = true
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "encryption_key" {
  secret      = google_secret_manager_secret.encryption_key.id
  secret_data = random_password.encryption_key.result
}

# =============================================================================
# Secret Manager IAM
# =============================================================================

# Backend service account access to secrets
resource "google_secret_manager_secret_iam_member" "backend_secrets" {
  for_each = {
    database_password = google_secret_manager_secret.database_password.secret_id
    jwt_secret       = google_secret_manager_secret.jwt_secret.secret_id
    api_keys         = google_secret_manager_secret.api_keys.secret_id
    encryption_key   = google_secret_manager_secret.encryption_key.secret_id
  }
  
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.backend.email}"
  project   = var.project_id
}

# CI/CD service account access to specific secrets
resource "google_secret_manager_secret_iam_member" "cicd_secrets" {
  for_each = {
    database_password = google_secret_manager_secret.database_password.secret_id
    api_keys         = google_secret_manager_secret.api_keys.secret_id
  }
  
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cicd.email}"
  project   = var.project_id
}

# =============================================================================
# Organization Policies (if org admin)
# =============================================================================

# Restrict public IP creation (if organization policies are available)
resource "google_project_organization_policy" "restrict_public_ips" {
  count      = var.enable_org_policies ? 1 : 0
  project    = var.project_id
  constraint = "compute.vmExternalIpAccess"
  
  list_policy {
    deny {
      all = true
    }
  }
}

# Require OS Login
resource "google_project_organization_policy" "require_os_login" {
  count      = var.enable_org_policies ? 1 : 0
  project    = var.project_id
  constraint = "compute.requireOsLogin"
  
  boolean_policy {
    enforced = true
  }
}

# Restrict shared VPC subnetworks
resource "google_project_organization_policy" "restrict_shared_vpc" {
  count      = var.enable_org_policies ? 1 : 0
  project    = var.project_id
  constraint = "compute.restrictSharedVpcSubnetworks"
  
  list_policy {
    allow {
      all = true
    }
  }
}

# =============================================================================
# Security Command Center (if available)
# =============================================================================

# Enable Security Command Center API
resource "google_project_service" "security_center" {
  count   = var.enable_security_center ? 1 : 0
  project = var.project_id
  service = "securitycenter.googleapis.com"
  
  disable_on_destroy = false
}

# =============================================================================
# Audit Logging Configuration
# =============================================================================

resource "google_project_iam_audit_config" "audit_config" {
  project = var.project_id
  service = "allServices"
  
  audit_log_config {
    log_type = "ADMIN_READ"
  }
  
  audit_log_config {
    log_type = "DATA_WRITE"
  }
  
  audit_log_config {
    log_type = "DATA_READ"
    exempted_members = [
      "serviceAccount:${google_service_account.monitoring.email}",
    ]
  }
}

# =============================================================================
# Network Security Policies
# =============================================================================

# Cloud Armor Security Policy
resource "google_compute_security_policy" "main" {
  name    = "${var.project_name}-${var.environment}-security-policy"
  project = var.project_id
  
  description = "Security policy for ${var.project_name} ${var.environment}"
  
  # Default rule - allow all (will be restricted by specific rules)
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "default rule"
  }
  
  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = var.rate_limit_requests_per_minute
        interval_sec = 60
      }
      ban_duration_sec = var.ban_duration_seconds
    }
    description = "Rate limiting rule"
  }
  
  # Block known bad IPs (example)
  dynamic "rule" {
    for_each = var.blocked_ip_ranges
    content {
      action   = "deny(403)"
      priority = "500"
      match {
        versioned_expr = "SRC_IPS_V1"
        config {
          src_ip_ranges = [rule.value]
        }
      }
      description = "Block known malicious IPs"
    }
  }
  
  # SQL injection protection
  rule {
    action   = "deny(403)"
    priority = "600"
    match {
      expr {
        expression = "origin.region_code == 'CN' || origin.region_code == 'RU'"
      }
    }
    description = "Block traffic from certain regions (optional)"
  }
  
  # Advanced DDoS protection
  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
      rule_visibility = "STANDARD"
    }
  }
}

# =============================================================================
# SSL Policies
# =============================================================================

resource "google_compute_ssl_policy" "main" {
  name    = "${var.project_name}-${var.environment}-ssl-policy"
  project = var.project_id
  profile = var.ssl_policy_profile
  min_tls_version = var.min_tls_version
  
  description = "SSL policy for ${var.project_name} ${var.environment}"
  
  dynamic "custom_features" {
    for_each = var.ssl_policy_profile == "CUSTOM" ? [1] : []
    content {
      features = var.custom_ssl_features
    }
  }
}

# =============================================================================
# Identity-Aware Proxy (IAP) Configuration
# =============================================================================

# IAP brand (required for IAP)
resource "google_iap_brand" "main" {
  count             = var.enable_iap ? 1 : 0
  support_email     = var.iap_support_email
  application_title = "${var.project_name} ${var.environment}"
  project           = var.project_id
}

# IAP OAuth client
resource "google_iap_client" "main" {
  count        = var.enable_iap ? 1 : 0
  display_name = "${var.project_name} ${var.environment} IAP Client"
  brand        = google_iap_brand.main[0].name
}

# =============================================================================
# Binary Authorization (if needed)
# =============================================================================

resource "google_binary_authorization_policy" "main" {
  count   = var.enable_binary_authorization ? 1 : 0
  project = var.project_id
  
  description = "Binary authorization policy for ${var.project_name}"
  
  default_admission_rule {
    evaluation_mode  = "REQUIRE_ATTESTATION"
    enforcement_mode = "ENFORCED_BLOCK_AND_AUDIT_LOG"
    
    require_attestations_by = var.attestation_authorities
  }
  
  # Cluster admission rules for specific clusters
  dynamic "cluster_admission_rules" {
    for_each = var.cluster_admission_rules
    content {
      cluster                = cluster_admission_rules.value.cluster
      evaluation_mode        = cluster_admission_rules.value.evaluation_mode
      enforcement_mode       = cluster_admission_rules.value.enforcement_mode
      require_attestations_by = cluster_admission_rules.value.require_attestations_by
    }
  }
}

# =============================================================================
# Security Scanning and Compliance
# =============================================================================

# Enable Container Analysis API for vulnerability scanning
resource "google_project_service" "container_analysis" {
  project = var.project_id
  service = "containeranalysis.googleapis.com"
  
  disable_on_destroy = false
}

# Enable Web Security Scanner API
resource "google_project_service" "web_security_scanner" {
  count   = var.enable_web_security_scanner ? 1 : 0
  project = var.project_id
  service = "websecurityscanner.googleapis.com"
  
  disable_on_destroy = false
}

# =============================================================================
# Access Context Manager (if needed)
# =============================================================================

# Access policy (organization level)
resource "google_access_context_manager_access_policy" "main" {
  count  = var.enable_access_context_manager ? 1 : 0
  parent = var.organization_id
  title  = "${var.project_name} Access Policy"
}

# Service perimeter for enhanced security
resource "google_access_context_manager_service_perimeter" "main" {
  count  = var.enable_access_context_manager ? 1 : 0
  parent = google_access_context_manager_access_policy.main[0].name
  name   = "${var.project_name}-${var.environment}-perimeter"
  title  = "${var.project_name} ${var.environment} Service Perimeter"
  
  status {
    restricted_services = var.restricted_services
    resources          = ["projects/${var.project_id}"]
    
    vpc_accessible_services {
      enable_restriction = true
      allowed_services   = var.vpc_accessible_services
    }
  }
}

# =============================================================================
# Security Headers and Policies
# =============================================================================

# Security headers for Cloud Run services (via environment variables)
locals {
  security_headers = {
    "Strict-Transport-Security" = "max-age=31536000; includeSubDomains"
    "X-Content-Type-Options"   = "nosniff"
    "X-Frame-Options"          = "DENY"
    "X-XSS-Protection"         = "1; mode=block"
    "Referrer-Policy"          = "strict-origin-when-cross-origin"
    "Content-Security-Policy"  = var.content_security_policy
    "Permissions-Policy"       = "geolocation=(), microphone=(), camera=()"
  }
}