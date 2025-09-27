# =============================================================================
# DARWIN Security Module - Variables
# =============================================================================

# =============================================================================
# Required Variables
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
}

# =============================================================================
# Optional Variables
# =============================================================================

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# =============================================================================
# KMS Configuration
# =============================================================================

variable "kms_location" {
  description = "Location for KMS key ring"
  type        = string
  default     = "global"
}

variable "key_rotation_period" {
  description = "Key rotation period in seconds"
  type        = string
  default     = "7776000s"  # 90 days
  
  validation {
    condition     = can(regex("^[0-9]+s$", var.key_rotation_period))
    error_message = "Key rotation period must be in seconds format (e.g., '7776000s')."
  }
}

variable "enable_kms_encryption" {
  description = "Enable KMS encryption for resources"
  type        = bool
  default     = true
}

# =============================================================================
# Organization Policies
# =============================================================================

variable "enable_org_policies" {
  description = "Enable organization policies (requires org admin permissions)"
  type        = bool
  default     = false
}

variable "organization_id" {
  description = "Organization ID for access context manager"
  type        = string
  default     = ""
}

# =============================================================================
# Security Center Configuration
# =============================================================================

variable "enable_security_center" {
  description = "Enable Security Command Center"
  type        = bool
  default     = false
}

variable "enable_web_security_scanner" {
  description = "Enable Web Security Scanner"
  type        = bool
  default     = false
}

# =============================================================================
# Cloud Armor Configuration
# =============================================================================

variable "rate_limit_requests_per_minute" {
  description = "Rate limit requests per minute"
  type        = number
  default     = 1000
  
  validation {
    condition     = var.rate_limit_requests_per_minute > 0
    error_message = "Rate limit must be greater than 0."
  }
}

variable "ban_duration_seconds" {
  description = "Duration to ban IP addresses in seconds"
  type        = number
  default     = 600  # 10 minutes
  
  validation {
    condition     = var.ban_duration_seconds >= 60 && var.ban_duration_seconds <= 86400
    error_message = "Ban duration must be between 60 seconds and 24 hours."
  }
}

variable "blocked_ip_ranges" {
  description = "List of IP ranges to block"
  type        = list(string)
  default     = []
}

variable "allowed_countries" {
  description = "List of allowed country codes (ISO 3166-1 alpha-2)"
  type        = list(string)
  default     = []  # Empty means all countries allowed
}

# =============================================================================
# SSL Policy Configuration
# =============================================================================

variable "ssl_policy_profile" {
  description = "SSL policy profile"
  type        = string
  default     = "MODERN"
  
  validation {
    condition     = contains(["COMPATIBLE", "MODERN", "RESTRICTED", "CUSTOM"], var.ssl_policy_profile)
    error_message = "SSL policy profile must be COMPATIBLE, MODERN, RESTRICTED, or CUSTOM."
  }
}

variable "min_tls_version" {
  description = "Minimum TLS version"
  type        = string
  default     = "TLS_1_2"
  
  validation {
    condition     = contains(["TLS_1_0", "TLS_1_1", "TLS_1_2", "TLS_1_3"], var.min_tls_version)
    error_message = "Minimum TLS version must be TLS_1_0, TLS_1_1, TLS_1_2, or TLS_1_3."
  }
}

variable "custom_ssl_features" {
  description = "Custom SSL features (only used with CUSTOM profile)"
  type        = list(string)
  default     = []
}

# =============================================================================
# Identity-Aware Proxy Configuration
# =============================================================================

variable "enable_iap" {
  description = "Enable Identity-Aware Proxy"
  type        = bool
  default     = false
}

variable "iap_support_email" {
  description = "Support email for IAP brand"
  type        = string
  default     = "admin@agourakis.med.br"
}

variable "iap_authorized_users" {
  description = "List of users authorized to access IAP-protected resources"
  type        = list(string)
  default     = []
}

variable "iap_authorized_groups" {
  description = "List of groups authorized to access IAP-protected resources"
  type        = list(string)
  default     = []
}

# =============================================================================
# Binary Authorization Configuration
# =============================================================================

variable "enable_binary_authorization" {
  description = "Enable Binary Authorization for container images"
  type        = bool
  default     = false
}

variable "attestation_authorities" {
  description = "List of attestation authority resource names"
  type        = list(string)
  default     = []
}

variable "cluster_admission_rules" {
  description = "Cluster-specific admission rules for Binary Authorization"
  type = list(object({
    cluster                = string
    evaluation_mode        = string
    enforcement_mode       = string
    require_attestations_by = list(string)
  }))
  default = []
}

# =============================================================================
# Access Context Manager Configuration
# =============================================================================

variable "enable_access_context_manager" {
  description = "Enable Access Context Manager for VPC Service Controls"
  type        = bool
  default     = false
}

variable "restricted_services" {
  description = "List of services to restrict with VPC Service Controls"
  type        = list(string)
  default = [
    "storage.googleapis.com",
    "sql-component.googleapis.com",
    "redis.googleapis.com"
  ]
}

variable "vpc_accessible_services" {
  description = "List of services accessible from VPC"
  type        = list(string)
  default = [
    "storage.googleapis.com",
    "sql-component.googleapis.com",
    "redis.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com"
  ]
}

# =============================================================================
# Audit Configuration
# =============================================================================

variable "enable_audit_logs" {
  description = "Enable audit logging"
  type        = bool
  default     = true
}

variable "audit_log_config" {
  description = "Audit log configuration"
  type = object({
    log_admin_read  = bool
    log_data_read   = bool
    log_data_write  = bool
  })
  default = {
    log_admin_read  = true
    log_data_read   = false  # Can be expensive
    log_data_write  = true
  }
}

variable "audit_exempted_members" {
  description = "Members exempted from audit logging"
  type        = list(string)
  default     = []
}

# =============================================================================
# Security Headers Configuration
# =============================================================================

variable "content_security_policy" {
  description = "Content Security Policy header value"
  type        = string
  default     = "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:"
}

variable "enable_security_headers" {
  description = "Enable security headers on services"
  type        = bool
  default     = true
}

variable "hsts_max_age" {
  description = "HSTS max age in seconds"
  type        = number
  default     = 31536000  # 1 year
  
  validation {
    condition     = var.hsts_max_age >= 0
    error_message = "HSTS max age must be non-negative."
  }
}

# =============================================================================
# Secret Management Configuration
# =============================================================================

variable "secret_replication_policy" {
  description = "Secret Manager replication policy"
  type        = string
  default     = "automatic"
  
  validation {
    condition     = contains(["automatic", "user_managed"], var.secret_replication_policy)
    error_message = "Secret replication policy must be 'automatic' or 'user_managed'."
  }
}

variable "secret_rotation_period_days" {
  description = "Secret rotation period in days"
  type        = number
  default     = 90
  
  validation {
    condition     = var.secret_rotation_period_days >= 1 && var.secret_rotation_period_days <= 365
    error_message = "Secret rotation period must be between 1 and 365 days."
  }
}

# =============================================================================
# Firewall Configuration
# =============================================================================

variable "enable_firewall_logging" {
  description = "Enable firewall rule logging"
  type        = bool
  default     = true
}

variable "firewall_log_sampling_rate" {
  description = "Firewall log sampling rate (0.0 to 1.0)"
  type        = number
  default     = 1.0
  
  validation {
    condition     = var.firewall_log_sampling_rate >= 0.0 && var.firewall_log_sampling_rate <= 1.0
    error_message = "Firewall log sampling rate must be between 0.0 and 1.0."
  }
}

# =============================================================================
# Compliance Configuration
# =============================================================================

variable "compliance_standards" {
  description = "List of compliance standards to adhere to"
  type        = list(string)
  default     = ["SOC2", "HIPAA", "GDPR"]
}

variable "data_classification" {
  description = "Data classification level"
  type        = string
  default     = "confidential"
  
  validation {
    condition     = contains(["public", "internal", "confidential", "restricted"], var.data_classification)
    error_message = "Data classification must be public, internal, confidential, or restricted."
  }
}

variable "enable_data_loss_prevention" {
  description = "Enable Cloud DLP for data protection"
  type        = bool
  default     = false
}

# =============================================================================
# Network Security Configuration
# =============================================================================

variable "enable_private_service_access" {
  description = "Enable private service access"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs for security monitoring"
  type        = bool
  default     = true
}

variable "flow_log_sampling_rate" {
  description = "VPC flow log sampling rate (0.0 to 1.0)"
  type        = number
  default     = 0.5
  
  validation {
    condition     = var.flow_log_sampling_rate >= 0.0 && var.flow_log_sampling_rate <= 1.0
    error_message = "Flow log sampling rate must be between 0.0 and 1.0."
  }
}

# =============================================================================
# Container Security Configuration
# =============================================================================

variable "enable_vulnerability_scanning" {
  description = "Enable container vulnerability scanning"
  type        = bool
  default     = true
}

variable "vulnerability_scan_policy" {
  description = "Vulnerability scan policy"
  type        = string
  default     = "scan_on_push"
  
  validation {
    condition     = contains(["scan_on_push", "continuous_scan", "manual_scan"], var.vulnerability_scan_policy)
    error_message = "Vulnerability scan policy must be scan_on_push, continuous_scan, or manual_scan."
  }
}

# =============================================================================
# Authentication and Authorization
# =============================================================================

variable "enable_workload_identity" {
  description = "Enable Workload Identity for GKE (if using Kubernetes)"
  type        = bool
  default     = false
}

variable "oauth_client_configuration" {
  description = "OAuth client configuration"
  type = object({
    client_id     = string
    client_secret = string
    redirect_uris = list(string)
  })
  default = {
    client_id     = ""
    client_secret = ""
    redirect_uris = []
  }
  sensitive = true
}

# =============================================================================
# Incident Response Configuration
# =============================================================================

variable "incident_response_contacts" {
  description = "List of contacts for incident response"
  type = list(object({
    name  = string
    email = string
    role  = string
  }))
  default = []
}

variable "security_contact_email" {
  description = "Primary security contact email"
  type        = string
  default     = "security@agourakis.med.br"
}

# =============================================================================
# Backup and Recovery Security
# =============================================================================

variable "backup_encryption_key" {
  description = "KMS key for backup encryption"
  type        = string
  default     = ""
}

variable "enable_backup_encryption" {
  description = "Enable encryption for backups"
  type        = bool
  default     = true
}

variable "backup_retention_policy" {
  description = "Backup retention policy in days"
  type        = number
  default     = 90
  
  validation {
    condition     = var.backup_retention_policy >= 1
    error_message = "Backup retention policy must be at least 1 day."
  }
}

# =============================================================================
# Threat Detection Configuration
# =============================================================================

variable "enable_anomaly_detection" {
  description = "Enable anomaly detection"
  type        = bool
  default     = true
}

variable "threat_detection_sensitivity" {
  description = "Threat detection sensitivity level"
  type        = string
  default     = "medium"
  
  validation {
    condition     = contains(["low", "medium", "high"], var.threat_detection_sensitivity)
    error_message = "Threat detection sensitivity must be low, medium, or high."
  }
}

# =============================================================================
# Data Protection Configuration
# =============================================================================

variable "enable_field_level_encryption" {
  description = "Enable field-level encryption for sensitive data"
  type        = bool
  default     = true
}

variable "pii_detection_enabled" {
  description = "Enable PII detection and protection"
  type        = bool
  default     = true
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 365
  
  validation {
    condition     = var.data_retention_days >= 1
    error_message = "Data retention days must be at least 1."
  }
}

# =============================================================================
# Network Security Variables
# =============================================================================

variable "enable_ddos_protection" {
  description = "Enable DDoS protection"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = true
}

variable "waf_rule_exclusions" {
  description = "WAF rule exclusions"
  type        = list(string)
  default     = []
}

# =============================================================================
# API Security Configuration
# =============================================================================

variable "api_rate_limits" {
  description = "API rate limits configuration"
  type = object({
    requests_per_second = number
    burst_limit        = number
    quota_per_day      = number
  })
  default = {
    requests_per_second = 100
    burst_limit        = 200
    quota_per_day      = 100000
  }
}

variable "enable_api_key_authentication" {
  description = "Enable API key authentication"
  type        = bool
  default     = true
}

variable "api_key_restrictions" {
  description = "API key restrictions"
  type = object({
    browser_key_restrictions = list(string)
    server_key_restrictions  = list(string)
    android_key_restrictions = list(string)
    ios_key_restrictions     = list(string)
  })
  default = {
    browser_key_restrictions = []
    server_key_restrictions  = []
    android_key_restrictions = []
    ios_key_restrictions     = []
  }
}

# =============================================================================
# Certificate Management
# =============================================================================

variable "certificate_transparency_enabled" {
  description = "Enable Certificate Transparency logging"
  type        = bool
  default     = true
}

variable "certificate_validity_days" {
  description = "Certificate validity period in days"
  type        = number
  default     = 365
  
  validation {
    condition     = var.certificate_validity_days >= 1 && var.certificate_validity_days <= 825
    error_message = "Certificate validity must be between 1 and 825 days."
  }
}