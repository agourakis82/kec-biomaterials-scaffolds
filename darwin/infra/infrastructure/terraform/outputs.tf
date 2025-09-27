# =============================================================================
# DARWIN Infrastructure - Outputs Configuration
# =============================================================================

# =============================================================================
# Project Information
# =============================================================================

output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "environment" {
  description = "Environment"
  value       = var.environment
}

# =============================================================================
# Networking Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "vpc_name" {
  description = "VPC name"
  value       = module.networking.vpc_name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = module.networking.subnet_id
}

output "subnet_name" {
  description = "Subnet name"
  value       = module.networking.subnet_name
}

output "vpc_connector_name" {
  description = "VPC Connector name"
  value       = module.networking.vpc_connector_name
}

output "load_balancer_ip" {
  description = "Load Balancer IP address"
  value       = module.networking.load_balancer_ip
}

# =============================================================================
# Backend Infrastructure Outputs
# =============================================================================

output "backend_service_name" {
  description = "Backend Cloud Run service name"
  value       = module.backend.service_name
}

output "backend_service_url" {
  description = "Backend service URL"
  value       = module.backend.service_url
}

output "api_url" {
  description = "API URL with custom domain"
  value       = module.backend.api_url
}

output "database_instance_name" {
  description = "Database instance name"
  value       = module.backend.database_instance_name
}

output "database_instance_id" {
  description = "Database instance ID"
  value       = module.backend.database_instance_id
}

output "database_connection_name" {
  description = "Database connection name"
  value       = module.backend.database_connection_name
}

output "redis_instance_id" {
  description = "Redis instance ID"
  value       = module.backend.redis_instance_id
}

output "redis_host" {
  description = "Redis host"
  value       = module.backend.redis_host
}

output "redis_port" {
  description = "Redis port"
  value       = module.backend.redis_port
}

output "storage_buckets" {
  description = "Storage bucket names"
  value       = module.backend.storage_buckets
}

# =============================================================================
# Frontend Infrastructure Outputs
# =============================================================================

output "frontend_service_name" {
  description = "Frontend Cloud Run service name"
  value       = module.frontend.service_name
}

output "frontend_service_url" {
  description = "Frontend service URL"
  value       = module.frontend.service_url
}

output "frontend_url" {
  description = "Frontend URL with custom domain"
  value       = module.frontend.frontend_url
}

output "cdn_url" {
  description = "CDN URL"
  value       = module.frontend.cdn_url
}

# =============================================================================
# Security Outputs
# =============================================================================

output "ssl_certificate_id" {
  description = "SSL certificate ID"
  value       = google_compute_managed_ssl_certificate.darwin_ssl.id
}

output "ssl_certificate_status" {
  description = "SSL certificate status"
  value       = google_compute_managed_ssl_certificate.darwin_ssl.managed[0]
}

output "service_accounts" {
  description = "Service accounts created"
  value = {
    backend  = module.backend.service_account_email
    frontend = module.frontend.service_account_email
  }
}

# =============================================================================
# Monitoring Outputs
# =============================================================================
/*
output "monitoring_dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = module.monitoring.dashboard_url
}

output "alerting_policies" {
  description = "Alerting policies"
  value       = module.monitoring.alerting_policies
}

output "uptime_checks" {
  description = "Uptime check IDs"
  value       = module.monitoring.uptime_checks
}
*/
# =============================================================================
# Cost Management Outputs
# =============================================================================
/*
output "budget_name" {
  description = "Budget name"
  value       = module.monitoring.budget_name
}
*/

output "estimated_monthly_cost" {
  description = "Estimated monthly cost"
  value       = "$${var.budget_amount}"
}

# =============================================================================
# Deployment Information
# =============================================================================

output "deployment_timestamp" {
  description = "Deployment timestamp"
  value       = timestamp()
}

output "terraform_version" {
  description = "Terraform version used"
  value       = "1.0+"
}

# =============================================================================
# Connection Strings and Configuration
# =============================================================================

output "database_url" {
  description = "Database connection URL (sensitive)"
  value       = module.backend.database_url
  sensitive   = true
}

output "redis_url" {
  description = "Redis connection URL (sensitive)"
  value       = module.backend.redis_url
  sensitive   = true
}

output "environment_variables" {
  description = "Environment variables for applications"
  value = {
    ENVIRONMENT = var.environment
    PROJECT_ID  = var.project_id
    REGION      = var.region
    API_URL     = module.backend.api_url
    FRONTEND_URL = module.frontend.frontend_url
    DATABASE_URL = module.backend.database_url
    REDIS_URL   = module.backend.redis_url
  }
  sensitive = true
}

# =============================================================================
# Health Check URLs
# =============================================================================

output "health_check_urls" {
  description = "Health check URLs for services"
  value = {
    api_health      = "${module.backend.api_url}/health"
    api_metrics     = "${module.backend.api_url}/metrics"
    frontend_health = module.frontend.frontend_url
  }
}

# =============================================================================
# DNS Configuration
# =============================================================================

output "dns_configuration" {
  description = "DNS configuration required"
  value = {
    api_domain = {
      domain = var.api_domain
      type   = "A"
      value  = module.networking.load_balancer_ip
    }
    frontend_domain = {
      domain = var.frontend_domain
      type   = "A" 
      value  = module.networking.load_balancer_ip
    }
  }
}

# =============================================================================
# Backup Information
# =============================================================================

output "backup_configuration" {
  description = "Backup configuration information"
  value = {
    database_backup_enabled = var.enable_backup
    storage_backup_enabled  = var.enable_backup
    backup_retention_days   = 7
  }
}

# =============================================================================
# Performance Configuration
# =============================================================================

output "performance_configuration" {
  description = "Performance configuration settings"
  value = {
    auto_scaling_enabled = var.enable_auto_scaling
    cdn_enabled         = var.enable_cdn
    connection_pool_size = var.connection_pool_size
    cache_ttl           = var.cache_ttl
  }
}