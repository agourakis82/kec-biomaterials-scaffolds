# =============================================================================
# DARWIN Infrastructure - Main Configuration
# Production-ready deployment for api.agourakis.med.br + darwin.agourakis.med.br
# =============================================================================

terraform {
  required_version = ">= 1.0"
  
  # backend "gcs" {
  #   bucket = "darwin-terraform-state-bucket"
  #   prefix = "terraform/state"
  # }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Local Values
# =============================================================================

locals {
  common_labels = {
    project     = "darwin"
    environment = var.environment
    managed_by  = "terraform"
    cost_center = "research"
    team        = "kec-biomaterials"
  }

  domains = {
    api      = var.api_domain
    frontend = var.frontend_domain
  }
}

# =============================================================================
# Networking Module
# =============================================================================

module "networking" {
  source = "./modules/networking"

  project_id   = var.project_id
  project_name = var.project_name
  region       = var.region
  environment  = var.environment
  
  vpc_name     = "${var.project_name}-vpc"
  subnet_cidr  = var.subnet_cidr
  
  common_labels = local.common_labels
}

# =============================================================================
# Backend Infrastructure Module
# =============================================================================

module "backend" {
  source = "./modules/backend"
  
  project_id    = var.project_id
  region        = var.region
  environment   = var.environment
  project_name  = var.project_name
  
  # Networking
  vpc_name       = module.networking.vpc_name
  subnet_name    = module.networking.subnet_name
  vpc_connector  = module.networking.vpc_connector_self_link
  
  # Database
  database_tier       = var.database_tier
  database_disk_size  = var.database_disk_size
  
  # Redis
  redis_memory_size   = var.redis_memory_size
  redis_version       = var.redis_version
  
  # Cloud Run
  backend_image       = var.backend_image
  min_instances       = var.backend_min_instances
  max_instances       = var.backend_max_instances
  cpu_limit          = var.backend_cpu_limit
  memory_limit       = var.backend_memory_limit
  
  # Domains
  api_domain = local.domains.api
  
  common_labels = local.common_labels
  
  depends_on = [module.networking]
}

# =============================================================================
# Frontend Infrastructure Module
# =============================================================================

module "frontend" {
  source = "./modules/frontend"
  
  project_id    = var.project_id
  region        = var.region
  environment   = var.environment
  project_name  = var.project_name
  
  # Networking
  vpc_name       = module.networking.vpc_name
  subnet_name    = module.networking.subnet_name
  vpc_connector  = module.networking.vpc_connector_self_link
  
  # Cloud Run
  frontend_image     = var.frontend_image
  min_instances      = var.frontend_min_instances
  max_instances      = var.frontend_max_instances
  cpu_limit         = var.frontend_cpu_limit
  memory_limit      = var.frontend_memory_limit
  
  # Backend integration
  api_url = module.backend.api_url
  
  # Domains
  frontend_domain = local.domains.frontend
  
  # SSL Certificate
  ssl_certificate_id = google_compute_managed_ssl_certificate.darwin_ssl.id
  
  # Load Balancer IP
  load_balancer_ip = module.networking.load_balancer_ip
  
  common_labels = local.common_labels
  
  depends_on = [module.networking, module.backend]
}

# =============================================================================
# Monitoring Infrastructure Module
# =============================================================================

# =============================================================================
# Monitoring Infrastructure Module (Commented out for now)
# =============================================================================

# module "monitoring" {
#   source = "./modules/monitoring"
#
#   project_id    = var.project_id
#   region        = var.region
#   environment   = var.environment
#   project_name  = var.project_name
#
#   # Resources to monitor
#   backend_service_name  = module.backend.service_name
#   frontend_service_name = module.frontend.service_name
#   database_instance_id  = module.backend.database_instance_id
#   redis_instance_id     = module.backend.redis_instance_id
#
#   # Alerting
#   notification_channels = var.notification_channels
#   budget_amount        = var.budget_amount
#
#   # Domains for uptime checks
#   api_domain      = local.domains.api
#   frontend_domain = local.domains.frontend
#
#   # Billing account for budget alerts
#   billing_account_id = var.billing_account_id
#
#   common_labels = local.common_labels
#
#   depends_on = [module.backend, module.frontend]
# }

# =============================================================================
# Security Configuration
# =============================================================================

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "run.googleapis.com",
    "sql-component.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "artifactregistry.googleapis.com",
    "certificatemanager.googleapis.com",
    "dns.googleapis.com"
  ])
  
  project = var.project_id
  service = each.key
  
  disable_on_destroy = false
}

# Global SSL Certificate for managed domains
resource "google_compute_managed_ssl_certificate" "darwin_ssl" {
  name = "${var.project_name}-ssl-cert"
  
  managed {
    domains = [
      local.domains.api,
      local.domains.frontend
    ]
  }
  
  lifecycle {
    create_before_destroy = true
  }
}