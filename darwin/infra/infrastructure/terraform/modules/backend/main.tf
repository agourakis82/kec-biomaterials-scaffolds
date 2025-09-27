# =============================================================================
# DARWIN Backend Module - Main Configuration
# Cloud Run, Cloud SQL PostgreSQL + pgvector, Redis, Storage, Service Accounts
# =============================================================================

# =============================================================================
# Local Values
# =============================================================================

locals {
  service_name = "${var.project_name}-${var.environment}-backend"
  
  # Storage bucket names
  bucket_names = {
    uploads    = "${var.project_name}-${var.environment}-uploads"
    documents  = "${var.project_name}-${var.environment}-documents"
    models     = "${var.project_name}-${var.environment}-models"
    backups    = "${var.project_name}-${var.environment}-backups"
    logs       = "${var.project_name}-${var.environment}-logs"
  }
  
  # Database configuration
  database_name = "${var.project_name}_${var.environment}"
  database_instance_name = "${var.project_name}-${var.environment}-db"
  
  # Redis configuration
  redis_instance_id = "${var.project_name}-${var.environment}-redis"
  
  # Service account names
  service_account_id = "${var.project_name}-${var.environment}-backend-sa"
}

# =============================================================================
# Random Password Generation
# =============================================================================

resource "random_password" "database_password" {
  length  = 32
  special = true
}

resource "random_id" "database_user_suffix" {
  byte_length = 4
}

# =============================================================================
# Service Account for Backend
# =============================================================================

resource "google_service_account" "backend" {
  account_id   = local.service_account_id
  display_name = "DARWIN Backend Service Account"
  description  = "Service account for ${var.project_name} backend services in ${var.environment}"
  project      = var.project_id
}

# IAM roles for the service account
resource "google_project_iam_member" "backend_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/redis.editor", 
    "roles/storage.objectAdmin",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent",
    "roles/clouddebugger.agent",
    "roles/cloudprofiler.agent",
    "roles/aiplatform.user"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# =============================================================================
# Cloud SQL PostgreSQL Instance with Vector Extensions
# =============================================================================

resource "google_sql_database_instance" "main" {
  name             = local.database_instance_name
  project          = var.project_id
  region           = var.region
  database_version = "POSTGRES_15"
  deletion_protection = var.deletion_protection
  
  settings {
    tier                        = var.database_tier
    availability_type           = var.database_availability_type
    disk_size                  = var.database_disk_size
    disk_type                  = "PD_SSD"
    disk_autoresize           = true
    disk_autoresize_limit     = var.database_disk_max_size
    
    # Backup configuration
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }
    
    # Maintenance window
    maintenance_window {
      day          = 7  # Sunday
      hour         = 4  # 4 AM
      update_track = "stable"
    }
    
    # IP configuration
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                              = "projects/${var.project_id}/global/networks/${var.vpc_name}"
      enable_private_path_for_google_cloud_services = true
      ssl_mode                                      = "ENCRYPTED_ONLY"
    }
    
    # Database flags (minimal configuration for compatibility)
    database_flags {
      name  = "max_connections"
      value = var.max_database_connections
    }
    
    # Enable query insights
    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute = 5
      query_string_length    = 1024
      record_application_tags = true
      record_client_address  = true
    }
  }
  
  depends_on = [google_service_account.backend]
}

# Database user
resource "google_sql_user" "backend_user" {
  name     = "darwin_user_${random_id.database_user_suffix.hex}"
  instance = google_sql_database_instance.main.name
  password = random_password.database_password.result
  project  = var.project_id
}

# Main application database
resource "google_sql_database" "main" {
  name     = local.database_name
  instance = google_sql_database_instance.main.name
  project  = var.project_id
  
  depends_on = [google_sql_database_instance.main]
}

# =============================================================================
# Redis Memorystore Instance
# =============================================================================

resource "google_redis_instance" "main" {
  name               = local.redis_instance_id
  project            = var.project_id
  region             = var.region
  memory_size_gb     = var.redis_memory_size
  redis_version      = var.redis_version
  display_name       = "${var.project_name} ${var.environment} Redis"
  
  # Network configuration
  authorized_network = "projects/${var.project_id}/global/networks/${var.vpc_name}"
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  # Redis configuration
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
    timeout = "300"
  }
  
  # Maintenance policy
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 4
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  # Auth and SSL
  auth_enabled            = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  
  labels = var.common_labels
}

# =============================================================================
# Cloud Storage Buckets
# =============================================================================

resource "google_storage_bucket" "buckets" {
  for_each = local.bucket_names
  
  name          = each.value
  project       = var.project_id
  location      = var.storage_location
  storage_class = var.storage_class
  
  # Uniform bucket-level access
  uniform_bucket_level_access = true
  
  # Versioning
  versioning {
    enabled = var.enable_versioning
  }
  
  # Lifecycle management
  dynamic "lifecycle_rule" {
    for_each = var.enable_lifecycle ? [1] : []
    content {
      condition {
        age = 90
      }
      action {
        type = "Delete"
      }
    }
  }
  
  dynamic "lifecycle_rule" {
    for_each = var.enable_lifecycle ? [1] : []
    content {
      condition {
        num_newer_versions = 3
      }
      action {
        type = "Delete"
      }
    }
  }
  
  # CORS configuration for uploads bucket
  dynamic "cors" {
    for_each = each.key == "uploads" ? [1] : []
    content {
      origin          = var.cors_origins
      method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
      response_header = ["*"]
      max_age_seconds = 3600
    }
  }
  
  # Encryption (only if KMS key is provided)
  dynamic "encryption" {
    for_each = var.kms_key_name != null ? [1] : []
    content {
      default_kms_key_name = var.kms_key_name
    }
  }
  
  labels = var.common_labels
  
  # Prevent accidental deletion for critical buckets
  lifecycle {
    prevent_destroy = false
  }
}

# Bucket IAM for service account
resource "google_storage_bucket_iam_member" "backend_bucket_access" {
  for_each = local.bucket_names
  
  bucket = google_storage_bucket.buckets[each.key].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.backend.email}"
}

# =============================================================================
# Secret Manager Secrets
# =============================================================================

resource "google_secret_manager_secret" "database_url" {
  secret_id = "${var.project_name}-${var.environment}-database-url"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "database_url" {
  secret = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${google_sql_user.backend_user.name}:${random_password.database_password.result}@${google_sql_database_instance.main.private_ip_address}:5432/${google_sql_database.main.name}"
}

resource "google_secret_manager_secret" "redis_url" {
  secret_id = "${var.project_name}-${var.environment}-redis-url"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret = google_secret_manager_secret.redis_url.id
  secret_data = "redis://:${google_redis_instance.main.auth_string}@${google_redis_instance.main.host}:${google_redis_instance.main.port}"
}

# JWT secret
resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "${var.project_name}-${var.environment}-jwt-secret"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = var.common_labels
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret = google_secret_manager_secret.jwt_secret.id
  secret_data = random_password.jwt_secret.result
}

# =============================================================================
# Cloud Run Service
# =============================================================================

resource "google_cloud_run_v2_service" "backend" {
  name     = local.service_name
  project  = var.project_id
  location = var.region
  
  ingress = "INGRESS_TRAFFIC_ALL"
  
  template {
    # Scaling configuration
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    # VPC Access
    vpc_access {
      connector = var.vpc_connector
      egress    = "PRIVATE_RANGES_ONLY"
    }
    
    # Service account
    service_account = google_service_account.backend.email
    
    containers {
      name  = "backend"
      image = var.backend_image
      
      # Resource limits
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle = true
        startup_cpu_boost = true
      }
      
      # Environment variables
      env {
        name = "ENVIRONMENT"
        value = var.environment
      }
      
      env {
        name = "PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name = "REGION"
        value = var.region
      }
      
      # Database URL from Secret Manager
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      
      # Redis URL from Secret Manager
      env {
        name = "REDIS_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.redis_url.secret_id
            version = "latest"
          }
        }
      }
      
      # JWT Secret from Secret Manager
      env {
        name = "JWT_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.jwt_secret.secret_id
            version = "latest"
          }
        }
      }
      
      # Storage bucket configurations
      env {
        name = "UPLOADS_BUCKET"
        value = google_storage_bucket.buckets["uploads"].name
      }
      
      env {
        name = "DOCUMENTS_BUCKET" 
        value = google_storage_bucket.buckets["documents"].name
      }
      
      env {
        name = "MODELS_BUCKET"
        value = google_storage_bucket.buckets["models"].name
      }
      
      env {
        name = "BACKUPS_BUCKET"
        value = google_storage_bucket.buckets["backups"].name
      }
      
      # Performance configurations
      env {
        name = "MAX_WORKERS"
        value = var.max_workers
      }
      
      env {
        name = "WORKER_TIMEOUT"
        value = var.worker_timeout
      }
      
      # Health check port
      ports {
        name           = "http1"
        container_port = 8080
      }
      
      # Startup probe
      startup_probe {
        initial_delay_seconds = 10
        timeout_seconds      = 5
        period_seconds       = 10
        failure_threshold    = 5
        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
    
    # Session affinity
    session_affinity = false
    
    # Execution environment
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    
    # Timeout
    timeout = "${var.request_timeout}s"
    
    labels = var.common_labels
  }
  
  # Traffic configuration
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  
  depends_on = [
    google_sql_database_instance.main,
    google_redis_instance.main,
    google_storage_bucket.buckets,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_version.redis_url,
    google_secret_manager_secret_version.jwt_secret
  ]
}

# =============================================================================
# Domain Mapping for API
# =============================================================================

resource "google_cloud_run_domain_mapping" "api_domain" {
  location = var.region
  name     = var.api_domain
  project  = var.project_id
  
  metadata {
    namespace = var.project_id
  }
  
  spec {
    route_name = google_cloud_run_v2_service.backend.name
  }
}

# =============================================================================
# IAM Policy for Cloud Run Service
# =============================================================================

resource "google_cloud_run_service_iam_policy" "noauth" {
  location = google_cloud_run_v2_service.backend.location
  project  = google_cloud_run_v2_service.backend.project
  service  = google_cloud_run_v2_service.backend.name
  
  policy_data = data.google_iam_policy.noauth.policy_data
}

data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}
resource "google_project_iam_member" "vpc_access_user" {
  role    = "roles/vpcaccess.user"
  member  = "serviceAccount:${google_service_account.backend.email}"
  project = var.project_id
}
