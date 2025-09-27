# =============================================================================
# DARWIN Backend Module - Outputs
# =============================================================================

# =============================================================================
# Service Account Outputs
# =============================================================================

output "service_account_email" {
  description = "Email of the backend service account"
  value       = google_service_account.backend.email
}

output "service_account_id" {
  description = "ID of the backend service account"
  value       = google_service_account.backend.id
}

output "service_account_unique_id" {
  description = "Unique ID of the backend service account"
  value       = google_service_account.backend.unique_id
}

# =============================================================================
# Database Outputs
# =============================================================================

output "database_instance_name" {
  description = "Name of the Cloud SQL instance"
  value       = google_sql_database_instance.main.name
}

output "database_instance_id" {
  description = "ID of the Cloud SQL instance"
  value       = google_sql_database_instance.main.id
}

output "database_connection_name" {
  description = "Connection name of the Cloud SQL instance"
  value       = google_sql_database_instance.main.connection_name
}

output "database_private_ip" {
  description = "Private IP address of the database"
  value       = google_sql_database_instance.main.private_ip_address
}

output "database_public_ip" {
  description = "Public IP address of the database"
  value       = google_sql_database_instance.main.public_ip_address
}

output "database_name" {
  description = "Name of the main database"
  value       = google_sql_database.main.name
}

output "database_user" {
  description = "Database user name"
  value       = google_sql_user.backend_user.name
}

output "database_url" {
  description = "Database connection URL (sensitive)"
  value       = "postgresql://${google_sql_user.backend_user.name}:${random_password.database_password.result}@${google_sql_database_instance.main.private_ip_address}:5432/${google_sql_database.main.name}"
  sensitive   = true
}

output "database_ssl_cert" {
  description = "Database SSL certificate"
  value       = google_sql_database_instance.main.server_ca_cert
}

# =============================================================================
# Redis Outputs
# =============================================================================

output "redis_instance_id" {
  description = "ID of the Redis instance"
  value       = google_redis_instance.main.id
}

output "redis_instance_name" {
  description = "Name of the Redis instance"
  value       = google_redis_instance.main.name
}

output "redis_host" {
  description = "Redis host address"
  value       = google_redis_instance.main.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.main.port
}

output "redis_url" {
  description = "Redis connection URL (sensitive)"
  value       = "redis://:${google_redis_instance.main.auth_string}@${google_redis_instance.main.host}:${google_redis_instance.main.port}"
  sensitive   = true
}

output "redis_auth_string" {
  description = "Redis auth string (sensitive)"
  value       = google_redis_instance.main.auth_string
  sensitive   = true
}

output "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  value       = google_redis_instance.main.memory_size_gb
}

output "redis_version" {
  description = "Redis version"
  value       = google_redis_instance.main.redis_version
}

# =============================================================================
# Cloud Storage Outputs
# =============================================================================

output "storage_buckets" {
  description = "Map of storage bucket names"
  value = {
    for k, v in google_storage_bucket.buckets : k => v.name
  }
}

output "storage_bucket_urls" {
  description = "Map of storage bucket URLs"
  value = {
    for k, v in google_storage_bucket.buckets : k => v.url
  }
}

output "uploads_bucket" {
  description = "Name of the uploads bucket"
  value       = google_storage_bucket.buckets["uploads"].name
}

output "documents_bucket" {
  description = "Name of the documents bucket"
  value       = google_storage_bucket.buckets["documents"].name
}

output "models_bucket" {
  description = "Name of the models bucket"
  value       = google_storage_bucket.buckets["models"].name
}

output "backups_bucket" {
  description = "Name of the backups bucket"
  value       = google_storage_bucket.buckets["backups"].name
}

output "logs_bucket" {
  description = "Name of the logs bucket"
  value       = google_storage_bucket.buckets["logs"].name
}

# =============================================================================
# Cloud Run Service Outputs
# =============================================================================

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.name
}

output "service_id" {
  description = "ID of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.id
}

output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.uri
}

output "service_location" {
  description = "Location of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.location
}

output "api_url" {
  description = "Custom domain URL for the API"
  value       = "https://${var.api_domain}"
}

output "service_status" {
  description = "Status of the Cloud Run service"
  value       = google_cloud_run_v2_service.backend.terminal_condition
}

# =============================================================================
# Secret Manager Outputs
# =============================================================================

output "database_secret_name" {
  description = "Name of the database URL secret"
  value       = google_secret_manager_secret.database_url.secret_id
}

output "redis_secret_name" {
  description = "Name of the Redis URL secret"
  value       = google_secret_manager_secret.redis_url.secret_id
}

output "jwt_secret_name" {
  description = "Name of the JWT secret"
  value       = google_secret_manager_secret.jwt_secret.secret_id
}

output "secrets_list" {
  description = "List of all secret names"
  value = [
    google_secret_manager_secret.database_url.secret_id,
    google_secret_manager_secret.redis_url.secret_id,
    google_secret_manager_secret.jwt_secret.secret_id
  ]
}

# =============================================================================
# Domain Mapping Outputs
# =============================================================================

output "domain_mapping_name" {
  description = "Name of the domain mapping"
  value       = google_cloud_run_domain_mapping.api_domain.name
}

output "domain_mapping_status" {
  description = "Status of the domain mapping"
  value       = google_cloud_run_domain_mapping.api_domain.status
}

# =============================================================================
# Configuration Summary Outputs
# =============================================================================

output "backend_config" {
  description = "Backend configuration summary"
  value = {
    service_name        = google_cloud_run_v2_service.backend.name
    service_url         = google_cloud_run_v2_service.backend.uri
    api_domain         = var.api_domain
    database_tier      = var.database_tier
    database_version   = google_sql_database_instance.main.database_version
    redis_memory_size  = var.redis_memory_size
    redis_version      = var.redis_version
    min_instances      = var.min_instances
    max_instances      = var.max_instances
    cpu_limit         = var.cpu_limit
    memory_limit      = var.memory_limit
    environment       = var.environment
    region           = var.region
  }
}

output "storage_config" {
  description = "Storage configuration summary"
  value = {
    buckets = {
      for k, v in google_storage_bucket.buckets : k => {
        name          = v.name
        location      = v.location
        storage_class = v.storage_class
        url          = v.url
      }
    }
    total_buckets = length(google_storage_bucket.buckets)
    versioning_enabled = var.enable_versioning
    lifecycle_enabled = var.enable_lifecycle
  }
}

output "database_config" {
  description = "Database configuration summary"
  value = {
    instance_name       = google_sql_database_instance.main.name
    database_version    = google_sql_database_instance.main.database_version
    tier               = google_sql_database_instance.main.settings[0].tier
    disk_size          = google_sql_database_instance.main.settings[0].disk_size
    availability_type  = google_sql_database_instance.main.settings[0].availability_type
    backup_enabled     = google_sql_database_instance.main.settings[0].backup_configuration[0].enabled
    private_network    = google_sql_database_instance.main.settings[0].ip_configuration[0].private_network
  }
}

output "redis_config" {
  description = "Redis configuration summary"
  value = {
    instance_id        = google_redis_instance.main.id
    memory_size_gb     = google_redis_instance.main.memory_size_gb
    redis_version      = google_redis_instance.main.redis_version
    auth_enabled       = google_redis_instance.main.auth_enabled
    transit_encryption = google_redis_instance.main.transit_encryption_mode
    authorized_network = google_redis_instance.main.authorized_network
    connect_mode      = google_redis_instance.main.connect_mode
  }
}

# =============================================================================
# Health Check and Monitoring Outputs
# =============================================================================

output "health_endpoints" {
  description = "Health check endpoints"
  value = {
    health_check = "https://${var.api_domain}/health"
    ready_check  = "https://${var.api_domain}/health/ready"
    metrics     = "https://${var.api_domain}/metrics"
    status      = "https://${var.api_domain}/status"
  }
}

output "monitoring_config" {
  description = "Monitoring configuration"
  value = {
    cloud_logging_enabled   = var.enable_cloud_logging
    cloud_monitoring_enabled = var.enable_cloud_monitoring
    cloud_trace_enabled     = var.enable_cloud_trace
    cloud_debugger_enabled  = var.enable_cloud_debugger
    cloud_profiler_enabled  = var.enable_cloud_profiler
    log_level              = var.log_level
  }
}

# =============================================================================
# Security Outputs
# =============================================================================

output "security_config" {
  description = "Security configuration summary"
  value = {
    service_account_email    = google_service_account.backend.email
    database_ssl_required    = var.enable_database_ssl
    redis_auth_enabled      = var.enable_redis_auth
    redis_ssl_enabled       = var.enable_redis_ssl
    secrets_count          = length([
      google_secret_manager_secret.database_url.secret_id,
      google_secret_manager_secret.redis_url.secret_id,
      google_secret_manager_secret.jwt_secret.secret_id
    ])
    kms_encryption_enabled = var.kms_key_name != null
  }
}

# =============================================================================
# Performance Outputs
# =============================================================================

output "performance_config" {
  description = "Performance configuration summary"
  value = {
    connection_pool_size       = var.connection_pool_size
    redis_connection_pool_size = var.redis_connection_pool_size
    max_workers               = var.max_workers
    worker_timeout            = var.worker_timeout
    request_timeout           = var.request_timeout
    connection_pooling_enabled = var.enable_connection_pooling
    caching_enabled           = var.enable_caching
    async_processing_enabled  = var.enable_async_processing
  }
}

# =============================================================================
# Feature Flags Outputs
# =============================================================================

output "feature_flags" {
  description = "Enabled feature flags"
  value = {
    vector_search_enabled     = var.enable_vector_search
    full_text_search_enabled = var.enable_full_text_search
    caching_enabled          = var.enable_caching
    async_processing_enabled = var.enable_async_processing
    file_uploads_enabled     = var.enable_file_uploads
    gpu_enabled             = var.enable_gpu
  }
}

# =============================================================================
# Environment Variables for Applications
# =============================================================================

output "environment_variables" {
  description = "Environment variables for the application (sensitive)"
  value = {
    ENVIRONMENT         = var.environment
    PROJECT_ID         = var.project_id
    REGION            = var.region
    DATABASE_URL      = "postgresql://${google_sql_user.backend_user.name}:${random_password.database_password.result}@${google_sql_database_instance.main.private_ip_address}:5432/${google_sql_database.main.name}"
    REDIS_URL         = "redis://:${google_redis_instance.main.auth_string}@${google_redis_instance.main.host}:${google_redis_instance.main.port}"
    UPLOADS_BUCKET    = google_storage_bucket.buckets["uploads"].name
    DOCUMENTS_BUCKET  = google_storage_bucket.buckets["documents"].name
    MODELS_BUCKET     = google_storage_bucket.buckets["models"].name
    BACKUPS_BUCKET    = google_storage_bucket.buckets["backups"].name
    LOGS_BUCKET       = google_storage_bucket.buckets["logs"].name
    MAX_WORKERS       = var.max_workers
    WORKER_TIMEOUT    = var.worker_timeout
  }
  sensitive = true
}

# =============================================================================
# Connection Strings (for external services)
# =============================================================================

output "connection_strings" {
  description = "Connection strings for external integrations (sensitive)"
  value = {
    database = {
      host     = google_sql_database_instance.main.private_ip_address
      port     = 5432
      database = google_sql_database.main.name
      user     = google_sql_user.backend_user.name
      password = random_password.database_password.result
      ssl_mode = "require"
    }
    redis = {
      host     = google_redis_instance.main.host
      port     = google_redis_instance.main.port
      password = google_redis_instance.main.auth_string
      ssl      = var.enable_redis_ssl
    }
  }
  sensitive = true
}