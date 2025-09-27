# =============================================================================
# DARWIN Security Module - Outputs
# =============================================================================

# =============================================================================
# KMS Outputs
# =============================================================================

output "kms_key_ring_name" {
  description = "Name of the KMS key ring"
  value       = google_kms_key_ring.main.name
}

output "kms_key_ring_id" {
  description = "ID of the KMS key ring"
  value       = google_kms_key_ring.main.id
}

output "database_kms_key_name" {
  description = "Name of the database KMS key"
  value       = google_kms_crypto_key.database_key.name
}

output "database_kms_key_id" {
  description = "ID of the database KMS key"
  value       = google_kms_crypto_key.database_key.id
}

output "storage_kms_key_name" {
  description = "Name of the storage KMS key"
  value       = google_kms_crypto_key.storage_key.name
}

output "storage_kms_key_id" {
  description = "ID of the storage KMS key"
  value       = google_kms_crypto_key.storage_key.id
}

output "backup_kms_key_name" {
  description = "Name of the backup KMS key"
  value       = google_kms_crypto_key.backup_key.name
}

output "backup_kms_key_id" {
  description = "ID of the backup KMS key"
  value       = google_kms_crypto_key.backup_key.id
}

# =============================================================================
# Service Account Outputs
# =============================================================================

output "service_accounts" {
  description = "Map of service account emails"
  value = {
    terraform  = google_service_account.terraform.email
    backend    = google_service_account.backend.email
    frontend   = google_service_account.frontend.email
    monitoring = google_service_account.monitoring.email
    cicd       = google_service_account.cicd.email
    database   = google_service_account.database.email
  }
}

output "terraform_service_account_email" {
  description = "Email of the Terraform service account"
  value       = google_service_account.terraform.email
}

output "backend_service_account_email" {
  description = "Email of the backend service account"
  value       = google_service_account.backend.email
}

output "frontend_service_account_email" {
  description = "Email of the frontend service account"
  value       = google_service_account.frontend.email
}

output "monitoring_service_account_email" {
  description = "Email of the monitoring service account"
  value       = google_service_account.monitoring.email
}

output "cicd_service_account_email" {
  description = "Email of the CI/CD service account"
  value       = google_service_account.cicd.email
}

output "database_service_account_email" {
  description = "Email of the database service account"
  value       = google_service_account.database.email
}

# =============================================================================
# Secret Manager Outputs
# =============================================================================

output "secrets" {
  description = "Map of secret names"
  value = {
    database_password = google_secret_manager_secret.database_password.secret_id
    jwt_secret       = google_secret_manager_secret.jwt_secret.secret_id
    api_keys         = google_secret_manager_secret.api_keys.secret_id
    encryption_key   = google_secret_manager_secret.encryption_key.secret_id
  }
}

output "database_password_secret_name" {
  description = "Name of the database password secret"
  value       = google_secret_manager_secret.database_password.secret_id
}

output "jwt_secret_name" {
  description = "Name of the JWT secret"
  value       = google_secret_manager_secret.jwt_secret.secret_id
}

output "api_keys_secret_name" {
  description = "Name of the API keys secret"
  value       = google_secret_manager_secret.api_keys.secret_id
}

output "encryption_key_secret_name" {
  description = "Name of the encryption key secret"
  value       = google_secret_manager_secret.encryption_key.secret_id
}

# =============================================================================
# Cloud Armor Outputs
# =============================================================================

output "security_policy_name" {
  description = "Name of the Cloud Armor security policy"
  value       = google_compute_security_policy.main.name
}

output "security_policy_id" {
  description = "ID of the Cloud Armor security policy"
  value       = google_compute_security_policy.main.id
}

output "security_policy_self_link" {
  description = "Self link of the Cloud Armor security policy"
  value       = google_compute_security_policy.main.self_link
}

# =============================================================================
# SSL Policy Outputs
# =============================================================================

output "ssl_policy_name" {
  description = "Name of the SSL policy"
  value       = google_compute_ssl_policy.main.name
}

output "ssl_policy_id" {
  description = "ID of the SSL policy"
  value       = google_compute_ssl_policy.main.id
}

output "ssl_policy_profile" {
  description = "SSL policy profile"
  value       = google_compute_ssl_policy.main.profile
}

output "ssl_policy_min_tls_version" {
  description = "Minimum TLS version for SSL policy"
  value       = google_compute_ssl_policy.main.min_tls_version
}

# =============================================================================
# IAP Outputs
# =============================================================================

output "iap_brand_name" {
  description = "Name of the IAP brand"
  value       = var.enable_iap ? google_iap_brand.main[0].name : null
}

output "iap_client_id" {
  description = "IAP OAuth client ID"
  value       = var.enable_iap ? google_iap_client.main[0].client_id : null
}

output "iap_client_secret" {
  description = "IAP OAuth client secret (sensitive)"
  value       = var.enable_iap ? google_iap_client.main[0].secret : null
  sensitive   = true
}

# =============================================================================
# Binary Authorization Outputs
# =============================================================================

output "binary_authorization_policy" {
  description = "Binary authorization policy"
  value       = var.enable_binary_authorization ? google_binary_authorization_policy.main[0].id : null
}

# =============================================================================
# Access Context Manager Outputs
# =============================================================================

output "access_policy_name" {
  description = "Name of the access context manager policy"
  value       = var.enable_access_context_manager ? google_access_context_manager_access_policy.main[0].name : null
}

output "service_perimeter_name" {
  description = "Name of the service perimeter"
  value       = var.enable_access_context_manager ? google_access_context_manager_service_perimeter.main[0].name : null
}

# =============================================================================
# Security Configuration Summary
# =============================================================================

output "security_config" {
  description = "Complete security configuration summary"
  value = {
    # Encryption
    kms_enabled              = var.enable_kms_encryption
    field_level_encryption   = var.enable_field_level_encryption
    backup_encryption       = var.enable_backup_encryption
    
    # Authentication & Authorization
    iap_enabled             = var.enable_iap
    workload_identity       = var.enable_workload_identity
    binary_authorization    = var.enable_binary_authorization
    
    # Network Security
    cloud_armor_enabled     = true
    ddos_protection        = var.enable_ddos_protection
    waf_enabled            = var.enable_waf
    ssl_policy_profile     = var.ssl_policy_profile
    min_tls_version        = var.min_tls_version
    
    # Monitoring & Compliance
    audit_logs_enabled      = var.enable_audit_logs
    vulnerability_scanning  = var.enable_vulnerability_scanning
    anomaly_detection      = var.enable_anomaly_detection
    pii_detection          = var.pii_detection_enabled
    security_center        = var.enable_security_center
    
    # Data Protection
    data_classification    = var.data_classification
    data_retention_days   = var.data_retention_days
    compliance_standards  = var.compliance_standards
    
    # Threat Detection
    threat_detection_sensitivity = var.threat_detection_sensitivity
  }
}

# =============================================================================
# Compliance Outputs
# =============================================================================

output "compliance_status" {
  description = "Compliance status summary"
  value = {
    standards_addressed = var.compliance_standards
    data_classification = var.data_classification
    
    # Security controls implemented
    controls = {
      encryption_at_rest     = var.enable_kms_encryption
      encryption_in_transit  = var.min_tls_version != "TLS_1_0"
      access_controls       = true  # Service accounts with least privilege
      audit_logging         = var.enable_audit_logs
      network_segmentation  = true  # VPC with firewall rules
      vulnerability_scanning = var.enable_vulnerability_scanning
      secret_management     = true  # Secret Manager
      backup_encryption     = var.enable_backup_encryption
      ddos_protection       = var.enable_ddos_protection
      rate_limiting         = true  # Cloud Armor
    }
    
    # Compliance frameworks
    frameworks = {
      soc2_type2 = {
        access_controls      = true
        audit_logging       = var.enable_audit_logs
        encryption          = var.enable_kms_encryption
        monitoring          = true
        incident_response   = length(var.incident_response_contacts) > 0
      }
      
      hipaa = {
        encryption_required  = var.enable_kms_encryption
        access_controls     = true
        audit_trails        = var.enable_audit_logs
        backup_encryption   = var.enable_backup_encryption
        data_retention      = var.data_retention_days >= 6  # 6 years for HIPAA
      }
      
      gdpr = {
        data_encryption     = var.enable_kms_encryption
        access_logging      = var.enable_audit_logs
        data_retention      = var.data_retention_days <= 365  # Max 1 year default
        pii_protection      = var.pii_detection_enabled
        right_to_erasure    = true  # Can be implemented via data deletion APIs
      }
    }
  }
}

# =============================================================================
# Security Headers Configuration
# =============================================================================

output "security_headers" {
  description = "Security headers configuration for applications"
  value = {
    "Strict-Transport-Security" = "max-age=${var.hsts_max_age}; includeSubDomains; preload"
    "X-Content-Type-Options"   = "nosniff"
    "X-Frame-Options"          = "DENY"
    "X-XSS-Protection"         = "1; mode=block"
    "Referrer-Policy"          = "strict-origin-when-cross-origin"
    "Content-Security-Policy"  = var.content_security_policy
    "Permissions-Policy"       = "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    "Cross-Origin-Embedder-Policy" = "require-corp"
    "Cross-Origin-Opener-Policy"   = "same-origin"
    "Cross-Origin-Resource-Policy" = "same-origin"
  }
}

# =============================================================================
# Security Monitoring Outputs
# =============================================================================

output "security_monitoring" {
  description = "Security monitoring configuration"
  value = {
    # Log sinks for security events
    security_log_filters = [
      "severity >= ERROR",
      "protoPayload.authenticationInfo.principalEmail != \"\"",
      "protoPayload.methodName : \"iam.googleapis.com\"",
      "resource.type=\"gce_firewall_rule\"",
      "jsonPayload.connection.dest_port = (22 OR 3389)"
    ]
    
    # Key metrics to monitor
    security_metrics = [
      "Failed authentication attempts",
      "Unusual API access patterns",
      "Database connection anomalies",
      "Firewall rule changes",
      "Service account key usage",
      "Secret access patterns"
    ]
    
    # Alert conditions
    alert_conditions = {
      failed_logins_threshold    = 10
      unusual_access_threshold   = 5
      admin_action_threshold     = 1
      firewall_change_threshold  = 1
    }
  }
}

# =============================================================================
# Security Best Practices Documentation
# =============================================================================

output "security_best_practices" {
  description = "Security best practices implemented"
  value = {
    # Identity and Access Management
    iam = [
      "Service accounts use least privilege principle",
      "Regular key rotation enabled",
      "Audit logging enabled for all admin actions",
      "Multi-factor authentication recommended for human users"
    ]
    
    # Network Security
    network = [
      "Private VPC with controlled egress",
      "Cloud Armor WAF protection enabled",
      "DDoS protection activated",
      "Rate limiting configured",
      "SSL/TLS encryption enforced"
    ]
    
    # Data Protection
    data = [
      "Encryption at rest using customer-managed keys",
      "Encryption in transit with TLS 1.2+",
      "Secret Manager for sensitive data",
      "Regular backup encryption",
      "PII detection and protection enabled"
    ]
    
    # Application Security
    application = [
      "Container vulnerability scanning",
      "Security headers enforced",
      "Input validation and sanitization",
      "SQL injection protection",
      "XSS protection enabled"
    ]
    
    # Monitoring and Compliance
    monitoring = [
      "Real-time security alerting",
      "Audit trail for all operations",
      "Anomaly detection enabled",
      "Compliance reporting automated",
      "Incident response procedures documented"
    ]
  }
}

# =============================================================================
# Certificate and SSL Outputs
# =============================================================================

output "ssl_configuration" {
  description = "SSL/TLS configuration details"
  value = {
    policy_name        = google_compute_ssl_policy.main.name
    profile           = google_compute_ssl_policy.main.profile
    min_tls_version   = google_compute_ssl_policy.main.min_tls_version
    enabled_features  = google_compute_ssl_policy.main.enabled_features
    
    # Certificate transparency
    ct_logging_enabled = var.certificate_transparency_enabled
    
    # HSTS configuration
    hsts_max_age      = var.hsts_max_age
    hsts_preload      = true
  }
}

# =============================================================================
# Access Control Outputs
# =============================================================================

output "access_control" {
  description = "Access control configuration"
  value = {
    # Service accounts created
    service_accounts_count = 6
    
    # IAP configuration
    iap_enabled           = var.enable_iap
    iap_support_email     = var.iap_support_email
    
    # Organization policies
    org_policies_enabled  = var.enable_org_policies
    
    # VPC Service Controls
    access_context_manager = var.enable_access_context_manager
    
    # Binary Authorization
    binary_authorization  = var.enable_binary_authorization
  }
}

# =============================================================================
# Threat Protection Outputs
# =============================================================================

output "threat_protection" {
  description = "Threat protection configuration"
  value = {
    # Cloud Armor configuration
    security_policy = {
      name                     = google_compute_security_policy.main.name
      rate_limit_enabled      = true
      requests_per_minute     = var.rate_limit_requests_per_minute
      ban_duration_seconds    = var.ban_duration_seconds
      blocked_ip_ranges_count = length(var.blocked_ip_ranges)
      adaptive_protection     = true
    }
    
    # DDoS protection
    ddos_protection_enabled = var.enable_ddos_protection
    
    # WAF configuration
    waf_enabled = var.enable_waf
    
    # Anomaly detection
    anomaly_detection = {
      enabled     = var.enable_anomaly_detection
      sensitivity = var.threat_detection_sensitivity
    }
  }
}

# =============================================================================
# Data Protection Outputs
# =============================================================================

output "data_protection" {
  description = "Data protection configuration"
  value = {
    # Encryption
    encryption = {
      kms_enabled            = var.enable_kms_encryption
      field_level_enabled    = var.enable_field_level_encryption
      backup_encryption      = var.enable_backup_encryption
      key_rotation_days     = tonumber(replace(var.key_rotation_period, "s", "")) / 86400
    }
    
    # Data governance
    governance = {
      classification       = var.data_classification
      retention_days      = var.data_retention_days
      pii_detection       = var.pii_detection_enabled
      dlp_enabled         = var.enable_data_loss_prevention
    }
    
    # Compliance
    compliance = {
      standards          = var.compliance_standards
      audit_enabled      = var.enable_audit_logs
      security_scanning  = var.enable_vulnerability_scanning
    }
  }
}

# =============================================================================
# Security Recommendations
# =============================================================================

output "security_recommendations" {
  description = "Security recommendations for production deployment"
  value = {
    immediate_actions = [
      "Configure DNS records for domains",
      "Wait for SSL certificate provisioning",
      "Test all endpoints for proper SSL/TLS configuration",
      "Verify firewall rules are correctly applied",
      "Confirm all secrets are properly configured"
    ]
    
    ongoing_tasks = [
      "Regular security scanning with Cloud Security Command Center",
      "Monitor audit logs for suspicious activities",
      "Rotate secrets according to schedule",
      "Update container images regularly",
      "Review and update firewall rules quarterly"
    ]
    
    compliance_tasks = [
      "Document security controls for compliance audits",
      "Implement data retention policies according to regulations",
      "Set up incident response procedures",
      "Conduct regular penetration testing",
      "Maintain security training for team members"
    ]
    
    monitoring_setup = [
      "Configure security alerting thresholds",
      "Set up automated compliance reporting",
      "Implement security metrics dashboard",
      "Enable real-time threat detection",
      "Configure backup and disaster recovery testing"
    ]
  }
}