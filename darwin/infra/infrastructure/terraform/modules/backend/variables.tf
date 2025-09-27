# =============================================================================
# DARWIN Backend Module - Variables
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

variable "vpc_name" {
  description = "Name of the VPC network"
  type        = string
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
}

variable "vpc_connector" {
  description = "VPC connector for Cloud Run"
  type        = string
}

variable "api_domain" {
  description = "Domain for API service"
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
# Database Configuration
# =============================================================================

variable "database_tier" {
  description = "Cloud SQL database tier"
  type        = string
  default     = "db-f1-micro"
  
  validation {
    condition = contains([
      "db-f1-micro", 
      "db-g1-small", 
      "db-n1-standard-1", 
      "db-n1-standard-2", 
      "db-n1-standard-4",
      "db-n1-standard-8",
      "db-n1-highmem-2",
      "db-n1-highmem-4",
      "db-custom-1-3840",
      "db-custom-2-7680",
      "db-custom-4-15360"
    ], var.database_tier)
    error_message = "Database tier must be a valid Cloud SQL tier."
  }
}

variable "database_disk_size" {
  description = "Database disk size in GB"
  type        = number
  default     = 20
  
  validation {
    condition     = var.database_disk_size >= 10 && var.database_disk_size <= 10000
    error_message = "Database disk size must be between 10 and 10000 GB."
  }
}

variable "database_disk_max_size" {
  description = "Maximum database disk size for autoresize in GB"
  type        = number
  default     = 100
  
  validation {
    condition     = var.database_disk_max_size >= 10 && var.database_disk_max_size <= 10000
    error_message = "Database disk max size must be between 10 and 10000 GB."
  }
}

variable "database_availability_type" {
  description = "Database availability type"
  type        = string
  default     = "ZONAL"
  
  validation {
    condition     = contains(["ZONAL", "REGIONAL"], var.database_availability_type)
    error_message = "Database availability type must be ZONAL or REGIONAL."
  }
}

variable "max_database_connections" {
  description = "Maximum number of database connections"
  type        = string
  default     = "100"
}

variable "deletion_protection" {
  description = "Enable deletion protection for database"
  type        = bool
  default     = true
}

variable "vpc_connector_ip_range" {
  description = "IP range for VPC connector access to database"
  type        = string
  default     = "10.8.0.0/28"
}

# =============================================================================
# Redis Configuration
# =============================================================================

variable "redis_memory_size" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
  
  validation {
    condition     = var.redis_memory_size >= 1 && var.redis_memory_size <= 300
    error_message = "Redis memory size must be between 1 and 300 GB."
  }
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_6_X"
  
  validation {
    condition     = contains(["REDIS_5_0", "REDIS_6_X", "REDIS_7_0"], var.redis_version)
    error_message = "Redis version must be REDIS_5_0, REDIS_6_X, or REDIS_7_0."
  }
}

# =============================================================================
# Cloud Run Configuration
# =============================================================================

variable "backend_image" {
  description = "Backend container image"
  type        = string
  default     = "gcr.io/PROJECT_ID/darwin-backend:latest"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
  
  validation {
    condition     = var.min_instances >= 0 && var.min_instances <= 100
    error_message = "Min instances must be between 0 and 100."
  }
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
  
  validation {
    condition     = var.max_instances >= 1 && var.max_instances <= 1000
    error_message = "Max instances must be between 1 and 1000."
  }
}

variable "cpu_limit" {
  description = "CPU limit for containers"
  type        = string
  default     = "2000m"
  
  validation {
    condition = can(regex("^[0-9]+m?$", var.cpu_limit))
    error_message = "CPU limit must be in format like '1000m' or '2'."
  }
}

variable "memory_limit" {
  description = "Memory limit for containers"
  type        = string
  default     = "4Gi"
  
  validation {
    condition = can(regex("^[0-9]+[GM]i?$", var.memory_limit))
    error_message = "Memory limit must be in format like '4Gi' or '2G'."
  }
}

variable "request_timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
  
  validation {
    condition     = var.request_timeout >= 1 && var.request_timeout <= 3600
    error_message = "Request timeout must be between 1 and 3600 seconds."
  }
}

variable "max_workers" {
  description = "Maximum number of worker processes"
  type        = string
  default     = "4"
}

variable "worker_timeout" {
  description = "Worker timeout in seconds"
  type        = string
  default     = "30"
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "storage_location" {
  description = "Location for storage buckets"
  type        = string
  default     = "US"
}

variable "storage_class" {
  description = "Storage class for buckets"
  type        = string
  default     = "STANDARD"
  
  validation {
    condition = contains([
      "STANDARD", 
      "NEARLINE", 
      "COLDLINE", 
      "ARCHIVE"
    ], var.storage_class)
    error_message = "Storage class must be STANDARD, NEARLINE, COLDLINE, or ARCHIVE."
  }
}

variable "enable_versioning" {
  description = "Enable versioning for storage buckets"
  type        = bool
  default     = true
}

variable "enable_lifecycle" {
  description = "Enable lifecycle management for storage buckets"
  type        = bool
  default     = true
}

variable "kms_key_name" {
  description = "KMS key for bucket encryption (null for Google-managed encryption)"
  type        = string
  default     = null
}

variable "cors_origins" {
  description = "CORS origins for uploads bucket"
  type        = list(string)
  default     = ["*"]
}

# =============================================================================
# Security Configuration
# =============================================================================

variable "enable_database_ssl" {
  description = "Require SSL for database connections"
  type        = bool
  default     = true
}

variable "enable_redis_auth" {
  description = "Enable Redis authentication"
  type        = bool
  default     = true
}

variable "enable_redis_ssl" {
  description = "Enable SSL for Redis connections"
  type        = bool
  default     = true
}

variable "allowed_ingress_sources" {
  description = "Allowed ingress sources for Cloud Run"
  type        = list(string)
  default     = ["all"]
}

# =============================================================================
# JAX and AI Configuration
# =============================================================================

variable "enable_gpu" {
  description = "Enable GPU support for JAX workloads"
  type        = bool
  default     = false
}

variable "gpu_type" {
  description = "GPU type for JAX workloads"
  type        = string
  default     = "nvidia-tesla-t4"
  
  validation {
    condition = contains([
      "nvidia-tesla-t4",
      "nvidia-tesla-v100",
      "nvidia-tesla-p100",
      "nvidia-tesla-k80"
    ], var.gpu_type)
    error_message = "GPU type must be a valid Google Cloud GPU type."
  }
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 0
  
  validation {
    condition     = var.gpu_count >= 0 && var.gpu_count <= 8
    error_message = "GPU count must be between 0 and 8."
  }
}

# =============================================================================
# Monitoring and Logging
# =============================================================================

variable "enable_cloud_logging" {
  description = "Enable Cloud Logging"
  type        = bool
  default     = true
}

variable "enable_cloud_monitoring" {
  description = "Enable Cloud Monitoring"
  type        = bool
  default     = true
}

variable "enable_cloud_trace" {
  description = "Enable Cloud Trace"
  type        = bool
  default     = true
}

variable "enable_cloud_debugger" {
  description = "Enable Cloud Debugger"
  type        = bool
  default     = false
}

variable "enable_cloud_profiler" {
  description = "Enable Cloud Profiler"
  type        = bool
  default     = false
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "INFO"
  
  validation {
    condition = contains([
      "DEBUG", 
      "INFO", 
      "WARNING", 
      "ERROR", 
      "CRITICAL"
    ], var.log_level)
    error_message = "Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
  }
}

# =============================================================================
# Performance Tuning
# =============================================================================

variable "connection_pool_size" {
  description = "Database connection pool size"
  type        = number
  default     = 20
  
  validation {
    condition     = var.connection_pool_size >= 1 && var.connection_pool_size <= 100
    error_message = "Connection pool size must be between 1 and 100."
  }
}

variable "redis_connection_pool_size" {
  description = "Redis connection pool size"
  type        = number
  default     = 10
  
  validation {
    condition     = var.redis_connection_pool_size >= 1 && var.redis_connection_pool_size <= 50
    error_message = "Redis connection pool size must be between 1 and 50."
  }
}

variable "enable_connection_pooling" {
  description = "Enable database connection pooling"
  type        = bool
  default     = true
}

# =============================================================================
# Backup and Recovery
# =============================================================================

variable "backup_retention_days" {
  description = "Database backup retention in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for database"
  type        = bool
  default     = true
}

variable "backup_start_time" {
  description = "Backup start time (HH:MM format)"
  type        = string
  default     = "03:00"
  
  validation {
    condition     = can(regex("^[0-2][0-9]:[0-5][0-9]$", var.backup_start_time))
    error_message = "Backup start time must be in HH:MM format (24-hour)."
  }
}

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_vector_search" {
  description = "Enable vector search capabilities"
  type        = bool
  default     = true
}

variable "enable_full_text_search" {
  description = "Enable full-text search capabilities"
  type        = bool
  default     = true
}

variable "enable_caching" {
  description = "Enable Redis caching"
  type        = bool
  default     = true
}

variable "enable_async_processing" {
  description = "Enable asynchronous task processing"
  type        = bool
  default     = true
}

variable "enable_file_uploads" {
  description = "Enable file upload functionality"
  type        = bool
  default     = true
}

# =============================================================================
# Service Account Configuration
# =============================================================================

variable "service_account" {
  description = "Service account email for VPC access"
  type        = string
  default     = ""
}