# =============================================================================
# DARWIN Frontend Module - Outputs
# =============================================================================

# =============================================================================
# Service Account Outputs
# =============================================================================

output "service_account_email" {
  description = "Email of the frontend service account"
  value       = google_service_account.frontend.email
}

output "service_account_id" {
  description = "ID of the frontend service account"
  value       = google_service_account.frontend.id
}

output "service_account_unique_id" {
  description = "Unique ID of the frontend service account"
  value       = google_service_account.frontend.unique_id
}

# =============================================================================
# Cloud Run Service Outputs
# =============================================================================

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.frontend.name
}

output "service_id" {
  description = "ID of the Cloud Run service"
  value       = google_cloud_run_v2_service.frontend.id
}

output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.frontend.uri
}

output "service_location" {
  description = "Location of the Cloud Run service"
  value       = google_cloud_run_v2_service.frontend.location
}

output "frontend_url" {
  description = "Custom domain URL for the frontend"
  value       = "https://${var.frontend_domain}"
}

output "service_status" {
  description = "Status of the Cloud Run service"
  value       = google_cloud_run_v2_service.frontend.terminal_condition
}

# =============================================================================
# Storage Outputs
# =============================================================================

output "cdn_bucket_name" {
  description = "Name of the CDN assets bucket"
  value       = google_storage_bucket.cdn_assets.name
}

output "cdn_bucket_url" {
  description = "URL of the CDN assets bucket"
  value       = google_storage_bucket.cdn_assets.url
}

output "static_bucket_name" {
  description = "Name of the static assets bucket"
  value       = google_storage_bucket.static_assets.name
}

output "static_bucket_url" {
  description = "URL of the static assets bucket"
  value       = google_storage_bucket.static_assets.url
}

output "storage_buckets" {
  description = "Map of storage bucket names"
  value = {
    cdn_assets    = google_storage_bucket.cdn_assets.name
    static_assets = google_storage_bucket.static_assets.name
  }
}

# =============================================================================
# CDN and Load Balancer Outputs
# =============================================================================

output "cdn_backend_bucket_name" {
  description = "Name of the CDN backend bucket"
  value       = google_compute_backend_bucket.cdn_backend.name
}

output "static_backend_bucket_name" {
  description = "Name of the static backend bucket"
  value       = google_compute_backend_bucket.static_backend.name
}

output "backend_service_name" {
  description = "Name of the backend service"
  value       = google_compute_backend_service.frontend.name
}

output "backend_service_id" {
  description = "ID of the backend service"
  value       = google_compute_backend_service.frontend.id
}

output "url_map_name" {
  description = "Name of the URL map"
  value       = google_compute_url_map.frontend.name
}

output "url_map_id" {
  description = "ID of the URL map"
  value       = google_compute_url_map.frontend.id
}

output "https_proxy_name" {
  description = "Name of the HTTPS proxy"
  value       = google_compute_target_https_proxy.frontend.name
}

output "https_proxy_id" {
  description = "ID of the HTTPS proxy"
  value       = google_compute_target_https_proxy.frontend.id
}

output "https_forwarding_rule_name" {
  description = "Name of the HTTPS forwarding rule"
  value       = google_compute_global_forwarding_rule.frontend_https.name
}

output "https_forwarding_rule_id" {
  description = "ID of the HTTPS forwarding rule"
  value       = google_compute_global_forwarding_rule.frontend_https.id
}

output "cdn_url" {
  description = "CDN URL for assets"
  value       = var.enable_cdn ? "https://cdn.${var.frontend_domain}" : google_storage_bucket.cdn_assets.url
}

# =============================================================================
# Domain Mapping Outputs
# =============================================================================

output "domain_mapping_name" {
  description = "Name of the domain mapping"
  value       = google_cloud_run_domain_mapping.frontend_domain.name
}

output "domain_mapping_status" {
  description = "Status of the domain mapping"
  value       = google_cloud_run_domain_mapping.frontend_domain.status
}

# =============================================================================
# Network Endpoint Group Outputs
# =============================================================================

output "neg_name" {
  description = "Name of the network endpoint group"
  value       = google_compute_region_network_endpoint_group.frontend_neg.name
}

output "neg_id" {
  description = "ID of the network endpoint group"
  value       = google_compute_region_network_endpoint_group.frontend_neg.id
}

# =============================================================================
# PWA and Assets Outputs
# =============================================================================

output "pwa_manifest_url" {
  description = "URL of the PWA manifest"
  value       = var.enable_pwa ? "https://${var.frontend_domain}/manifest.json" : null
}

output "service_worker_url" {
  description = "URL of the service worker"
  value       = var.enable_pwa ? "https://${var.frontend_domain}/sw.js" : null
}

output "pwa_enabled" {
  description = "Whether PWA features are enabled"
  value       = var.enable_pwa
}



# =============================================================================
# Configuration Summary Outputs
# =============================================================================

output "frontend_config" {
  description = "Frontend configuration summary"
  value = {
    service_name        = google_cloud_run_v2_service.frontend.name
    service_url         = google_cloud_run_v2_service.frontend.uri
    frontend_domain     = var.frontend_domain
    frontend_url        = "https://${var.frontend_domain}"
    cdn_enabled         = var.enable_cdn
    pwa_enabled         = var.enable_pwa
    min_instances       = var.min_instances
    max_instances       = var.max_instances
    cpu_limit          = var.cpu_limit
    memory_limit       = var.memory_limit
    environment        = var.environment
    region            = var.region
  }
}

output "cdn_config" {
  description = "CDN configuration summary"
  value = {
    enabled          = var.enable_cdn
    default_ttl      = var.cdn_default_ttl
    max_ttl         = var.cdn_max_ttl
    client_ttl      = var.cdn_client_ttl
    cdn_bucket      = google_storage_bucket.cdn_assets.name
    static_bucket   = google_storage_bucket.static_assets.name
  }
}

output "storage_config" {
  description = "Storage configuration summary"
  value = {
    buckets = {
      cdn_assets = {
        name     = google_storage_bucket.cdn_assets.name
        location = google_storage_bucket.cdn_assets.location
        url      = google_storage_bucket.cdn_assets.url
      }
      static_assets = {
        name     = google_storage_bucket.static_assets.name
        location = google_storage_bucket.static_assets.location
        url      = google_storage_bucket.static_assets.url
      }
    }
    cors_enabled = length(var.cors_origins) > 0
  }
}

# =============================================================================
# Performance Outputs
# =============================================================================

output "performance_config" {
  description = "Performance configuration summary"
  value = {
    cdn_enabled              = var.enable_cdn
    compression_enabled      = var.enable_compression
    http2_enabled           = var.enable_http2
    asset_optimization      = var.enable_asset_optimization
    connection_draining_timeout = var.connection_draining_timeout
    static_assets_max_age   = var.static_assets_max_age
    html_max_age           = var.html_max_age
  }
}

# =============================================================================
# Security Outputs
# =============================================================================

output "security_config" {
  description = "Security configuration summary"
  value = {
    hsts_enabled            = var.enable_hsts
    hsts_max_age           = var.hsts_max_age
    csrf_protection        = var.enable_csrf_protection
    rate_limiting          = var.enable_rate_limiting
    content_security_policy = var.content_security_policy
    service_account_email   = google_service_account.frontend.email
  }
}

# =============================================================================
# Feature Flags Outputs
# =============================================================================

output "feature_flags" {
  description = "Enabled feature flags"
  value = {
    pwa_enabled              = var.enable_pwa
    analytics_enabled        = var.enable_analytics
    real_time_enabled        = var.enable_real_time
    offline_support_enabled  = var.enable_offline_support
    asset_optimization      = var.enable_asset_optimization
    service_worker_enabled  = var.enable_service_worker
    i18n_enabled           = var.enable_i18n
  }
}

# =============================================================================
# Environment Variables for Applications
# =============================================================================

output "environment_variables" {
  description = "Environment variables for the frontend application"
  value = {
    NODE_ENV                    = var.environment == "production" ? "production" : "development"
    ENVIRONMENT                = var.environment
    PROJECT_ID                 = var.project_id
    REGION                     = var.region
    NEXT_PUBLIC_API_URL        = var.api_url
    NEXT_PUBLIC_FRONTEND_URL   = "https://${var.frontend_domain}"
    NEXT_PUBLIC_CDN_URL        = var.enable_cdn ? "https://cdn.${var.frontend_domain}" : google_storage_bucket.cdn_assets.url
    NEXT_PUBLIC_STATIC_URL     = google_storage_bucket.static_assets.url
    NEXT_PUBLIC_ENABLE_PWA     = var.enable_pwa
    NEXT_PUBLIC_ENABLE_ANALYTICS = var.enable_analytics
    NEXT_PUBLIC_ENABLE_REAL_TIME = var.enable_real_time
    NEXT_PUBLIC_ENABLE_OFFLINE = var.enable_offline_support
  }
}

# =============================================================================
# Health Check Outputs
# =============================================================================

output "health_endpoints" {
  description = "Health check endpoints"
  value = {
    health_check = "https://${var.frontend_domain}/api/health"
    ready_check  = "https://${var.frontend_domain}/api/ready"
    status      = "https://${var.frontend_domain}/api/status"
  }
}

# =============================================================================
# Monitoring Outputs
# =============================================================================

output "monitoring_config" {
  description = "Monitoring configuration"
  value = {
    access_logs_enabled     = var.enable_access_logs
    error_reporting_enabled = var.enable_error_reporting
    log_sample_rate        = var.log_sample_rate
    analytics_id           = var.analytics_id
    sentry_dsn            = var.sentry_dsn != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
  }
}

# =============================================================================
# Build Configuration Outputs
# =============================================================================

output "build_config" {
  description = "Build configuration summary"
  value = {
    github_owner           = var.github_owner
    github_repo           = var.github_repo
    github_branch         = var.github_branch
    node_version          = var.node_version
    build_command         = var.nextjs_build_command
    start_command         = var.nextjs_start_command
    build_timeout         = var.build_timeout
    optimization_enabled  = var.enable_asset_optimization
  }
}

# =============================================================================
# URLs and Endpoints
# =============================================================================

output "urls" {
  description = "Important URLs and endpoints"
  value = {
    frontend_url          = "https://${var.frontend_domain}"
    cdn_url              = var.enable_cdn ? "https://cdn.${var.frontend_domain}" : google_storage_bucket.cdn_assets.url
    static_assets_url    = google_storage_bucket.static_assets.url
    pwa_manifest_url     = var.enable_pwa ? "https://${var.frontend_domain}/manifest.json" : null
    service_worker_url   = var.enable_pwa ? "https://${var.frontend_domain}/sw.js" : null
  }
}

# =============================================================================
# Internationalization Outputs
# =============================================================================

output "i18n_config" {
  description = "Internationalization configuration"
  value = {
    enabled           = var.enable_i18n
    default_locale    = var.default_locale
    supported_locales = var.supported_locales
  }
}

# =============================================================================
# Asset Management Outputs
# =============================================================================

output "asset_config" {
  description = "Asset management configuration"
  value = {
    image_optimization_quality = var.image_optimization_quality
    static_assets_max_age     = var.static_assets_max_age
    html_max_age             = var.html_max_age
    api_max_age              = var.api_max_age
    minification_enabled     = var.minification
    tree_shaking_enabled     = var.tree_shaking
  }
}

# =============================================================================
# Third-party Integration Outputs
# =============================================================================

output "integrations" {
  description = "Third-party integrations status"
  value = {
    analytics = {
      enabled = var.enable_analytics == "true"
      id      = var.analytics_id != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
    }
    sentry = {
      enabled = var.sentry_dsn != ""
      dsn     = var.sentry_dsn != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
    }
    intercom = {
      enabled = var.intercom_app_id != ""
      app_id  = var.intercom_app_id != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
    }
  }
}