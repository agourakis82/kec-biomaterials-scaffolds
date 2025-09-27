# =============================================================================
# DARWIN Terraform Variables - Production Environment
# Configuração de variáveis para deployment de produção
# =============================================================================

# =============================================================================
# Core Project Configuration
# =============================================================================

project_id = "pcs-helio"
project_name = "darwin"
region = "us-central1"
environment = "production"

# =============================================================================
# Domain Configuration
# =============================================================================

api_domain = "api.agourakis.med.br"
frontend_domain = "darwin.agourakis.med.br"

ssl_certificate_domains = [
  "api.agourakis.med.br",
  "darwin.agourakis.med.br"
]

# =============================================================================
# Networking Configuration
# =============================================================================

subnet_cidr = "10.0.0.0/24"
allowed_cidrs = ["0.0.0.0/0"]  # Adjust for production security

# =============================================================================
# Database Configuration (Production-Ready)
# =============================================================================

database_tier = "db-n1-standard-2"  # 2 vCPUs, 7.5GB RAM
database_disk_size = 100
database_disk_max_size = 500
database_availability_type = "REGIONAL"  # High availability
max_database_connections = "200"
deletion_protection = true

# Backup Configuration
backup_retention_days = 30
enable_point_in_time_recovery = true
backup_start_time = "03:00"

# =============================================================================
# Redis Configuration
# =============================================================================

redis_memory_size = 2  # 2GB
redis_version = "REDIS_6_X"

# =============================================================================
# Backend Cloud Run Configuration
# =============================================================================

backend_image = "gcr.io/pcs-helio/darwin-backend:latest"
backend_min_instances = 2  # Always warm for production
backend_max_instances = 20
backend_cpu_limit = "2000m"  # 2 vCPUs
backend_memory_limit = "4Gi"  # 4GB RAM

# Backend Application Settings
max_workers = "4"
worker_timeout = "30"
connection_pool_size = 20
redis_connection_pool_size = 10

# =============================================================================
# Frontend Cloud Run Configuration
# =============================================================================

frontend_image = "gcr.io/pcs-helio/darwin-frontend:latest"
frontend_min_instances = 1
frontend_max_instances = 10
frontend_cpu_limit = "1000m"  # 1 vCPU
frontend_memory_limit = "2Gi"  # 2GB RAM

# =============================================================================
# Storage Configuration
# =============================================================================

storage_bucket_location = "US"
storage_class = "STANDARD"

# =============================================================================
# Monitoring Configuration
# =============================================================================

# Email addresses for alerts (configure as needed)
notification_channels = []
email_addresses = [
  "admin@agourakis.med.br"
  # Add more email addresses as needed
]

# Budget configuration
budget_amount = 500  # USD per month

# Alert thresholds
error_rate_threshold = 0.05  # 5%
latency_threshold_ms = 5000  # 5 seconds
cpu_threshold_percent = 80
memory_threshold_percent = 85
database_connections_threshold = 160  # 80% of max connections
redis_memory_threshold_percent = 85

# SLO configuration
api_availability_slo = 0.995  # 99.5%
api_latency_slo = 2000  # 2 seconds
slo_rolling_period_days = 30

# =============================================================================
# Feature Flags
# =============================================================================

enable_cdn = true
enable_backup = true
enable_monitoring = true
enable_auto_scaling = true

# Vector search and AI features
enable_vector_search = true
enable_full_text_search = true
enable_caching = true
enable_async_processing = true
enable_file_uploads = true

# Security features
enable_database_ssl = true
enable_redis_auth = true
enable_redis_ssl = true

# Performance features
enable_connection_pooling = true
enable_compression = true
enable_http2 = true

# PWA features
enable_pwa = true
enable_service_worker = true
enable_offline_support = true

# Development features (disabled in production)
enable_debug_mode = false
enable_profiling = false
enable_debugging = false

# =============================================================================
# Performance Tuning
# =============================================================================

request_timeout = 300  # 5 minutes
cache_ttl = 3600  # 1 hour

# CDN Configuration
cdn_default_ttl = 3600  # 1 hour
cdn_max_ttl = 86400  # 24 hours
cdn_client_ttl = 3600  # 1 hour

# =============================================================================
# Security Configuration
# =============================================================================

# SSL Configuration
ssl_policy = "MODERN"
enable_hsts = true
hsts_max_age = 31536000  # 1 year

# Rate limiting
rate_limit_requests_per_minute = 1000

# Content Security Policy
content_security_policy = "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:"

# =============================================================================
# Integration Configuration
# =============================================================================

# GitHub configuration (for Cloud Build triggers)
github_owner = ""  # Configure if using GitHub integration
github_repo = ""   # Configure if using GitHub integration
github_branch = "main"

# Third-party integrations
analytics_id = ""  # Configure Google Analytics if needed
sentry_dsn = ""    # Configure Sentry for error tracking
intercom_app_id = ""  # Configure Intercom for customer support

# =============================================================================
# Internationalization
# =============================================================================

default_locale = "en"
supported_locales = ["en", "pt", "es"]
enable_i18n = false  # Disable initially, enable when needed

# =============================================================================
# Advanced Configuration
# =============================================================================

# JAX and GPU configuration
enable_gpu = false  # Enable when GPU workloads are needed
gpu_type = "nvidia-tesla-t4"
gpu_count = 0

# Logging configuration
log_level = "INFO"
enable_cloud_logging = true
enable_cloud_monitoring = true
enable_cloud_trace = true

# Asset optimization
enable_asset_optimization = true
image_optimization_quality = 85

# Node.js configuration
node_version = "18"
npm_registry = "https://registry.npmjs.org/"

# Build configuration
build_timeout = "1200s"  # 20 minutes

# =============================================================================
# Environment-Specific Overrides
# =============================================================================

# Note: This file represents production configuration
# For other environments, copy this file and modify values:
# 
# For staging:
#   - database_tier = "db-f1-micro"
#   - backend_min_instances = 0
#   - budget_amount = 200
#   - enable_backup = false (optional)
#
# For development:
#   - database_tier = "db-f1-micro"
#   - backend_min_instances = 0
#   - frontend_min_instances = 0
#   - budget_amount = 100
#   - enable_backup = false
#   - enable_monitoring = false (optional)