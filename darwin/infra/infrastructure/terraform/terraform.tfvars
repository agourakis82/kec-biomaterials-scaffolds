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
billing_account_id = "01D1AC-D3254D-FC28CF"

# =============================================================================
# Domain Configuration
# =============================================================================

api_domain = "api.agourakis.med.br"
frontend_domain = "darwin.agourakis.med.br"

# =============================================================================
# Networking Configuration
# =============================================================================

subnet_cidr = "10.0.0.0/24"
allowed_cidrs = ["0.0.0.0/0"]  # Adjust for production security

# =============================================================================
# Database Configuration (Production-Ready)
# =============================================================================

database_tier = "db-custom-2-7680"  # 2 vCPUs, 7.5GB RAM (PostgreSQL compatible)
database_disk_size = 100

# =============================================================================
# Redis Configuration
# =============================================================================

redis_memory_size = 2  # 2GB
redis_version = "REDIS_6_X"

# =============================================================================
# Backend Cloud Run Configuration
# =============================================================================

backend_image = "gcr.io/pcs-helio/darwin-backend:fixed2"
backend_min_instances = 2  # Always warm for production
backend_max_instances = 20
backend_cpu_limit = "2000m"  # 2 vCPUs
backend_memory_limit = "4Gi"  # 4GB RAM

# =============================================================================
# Frontend Cloud Run Configuration
# =============================================================================

frontend_image = "gcr.io/pcs-helio/darwin-frontend:test"
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

# Notification channels for alerts
notification_channels = []

# Budget configuration
budget_amount = 500  # USD per month

# =============================================================================
# Feature Flags
# =============================================================================

enable_cdn = true
enable_backup = true
enable_monitoring = true
enable_auto_scaling = true
enable_asset_optimization = false