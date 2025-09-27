# =============================================================================
# DARWIN Infrastructure - Variables Configuration
# =============================================================================

# =============================================================================
# Core Project Variables
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "darwin"
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "billing_account_id" {
  description = "GCP Billing Account ID"
  type        = string
  sensitive   = true
}

# =============================================================================
# Domain Configuration
# =============================================================================

variable "api_domain" {
  description = "Domain for API service"
  type        = string
  default     = "api.agourakis.med.br"
}

variable "frontend_domain" {
  description = "Domain for frontend service"
  type        = string
  default     = "darwin.agourakis.med.br"
}

# =============================================================================
# Networking Variables
# =============================================================================

variable "subnet_cidr" {
  description = "CIDR range for subnet"
  type        = string
  default     = "10.0.0.0/24"
}

# =============================================================================
# Database Variables
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
    condition     = var.database_disk_size >= 10 && var.database_disk_size <= 1000
    error_message = "Database disk size must be between 10 and 1000 GB."
  }
}

# =============================================================================
# Redis Variables
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
# Backend Cloud Run Variables
# =============================================================================

variable "backend_image" {
  description = "Backend container image"
  type        = string
  default     = "gcr.io/PROJECT_ID/darwin-backend:latest"
}

variable "backend_min_instances" {
  description = "Minimum number of backend instances"
  type        = number
  default     = 1
}

variable "backend_max_instances" {
  description = "Maximum number of backend instances"
  type        = number
  default     = 10
}

variable "backend_cpu_limit" {
  description = "Backend CPU limit"
  type        = string
  default     = "2000m"
}

variable "backend_memory_limit" {
  description = "Backend memory limit"
  type        = string
  default     = "4Gi"
}

# =============================================================================
# Frontend Cloud Run Variables
# =============================================================================

variable "frontend_image" {
  description = "Frontend container image"
  type        = string
  default     = "gcr.io/PROJECT_ID/darwin-frontend:latest"
}

variable "frontend_min_instances" {
  description = "Minimum number of frontend instances"
  type        = number
  default     = 1
}

variable "frontend_max_instances" {
  description = "Maximum number of frontend instances"
  type        = number
  default     = 5
}

variable "frontend_cpu_limit" {
  description = "Frontend CPU limit"
  type        = string
  default     = "1000m"
}

variable "frontend_memory_limit" {
  description = "Frontend memory limit"
  type        = string
  default     = "2Gi"
}

# =============================================================================
# Monitoring Variables
# =============================================================================

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 500
  
  validation {
    condition     = var.budget_amount > 0
    error_message = "Budget amount must be greater than 0."
  }
}

# =============================================================================
# Security Variables
# =============================================================================

variable "allowed_cidrs" {
  description = "List of CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Open to all - adjust for production
}

variable "ssl_certificate_domains" {
  description = "List of domains for SSL certificate"
  type        = list(string)
  default     = ["api.agourakis.med.br", "darwin.agourakis.med.br"]
}

# =============================================================================
# Storage Variables
# =============================================================================

variable "storage_bucket_location" {
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

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_cdn" {
  description = "Enable Cloud CDN for frontend"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable advanced monitoring"
  type        = bool
  default     = true
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for services"
  type        = bool
  default     = true
}

# =============================================================================
# Performance Variables
# =============================================================================

variable "connection_pool_size" {
  description = "Database connection pool size"
  type        = number
  default     = 20
}

variable "request_timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 30
}

variable "cache_ttl" {
  description = "Cache TTL in seconds"
  type        = number
  default     = 3600
}