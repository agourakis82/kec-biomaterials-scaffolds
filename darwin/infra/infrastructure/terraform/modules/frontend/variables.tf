# =============================================================================
# DARWIN Frontend Module - Variables
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

variable "frontend_domain" {
  description = "Domain for frontend service"
  type        = string
}

variable "api_url" {
  description = "Backend API URL"
  type        = string
}

variable "ssl_certificate_id" {
  description = "SSL certificate ID for HTTPS"
  type        = string
}

variable "load_balancer_ip" {
  description = "Load balancer IP address"
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
# Cloud Run Configuration
# =============================================================================

variable "frontend_image" {
  description = "Frontend container image"
  type        = string
  default     = "gcr.io/PROJECT_ID/darwin-frontend:latest"
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
  default     = 5
  
  validation {
    condition     = var.max_instances >= 1 && var.max_instances <= 1000
    error_message = "Max instances must be between 1 and 1000."
  }
}

variable "cpu_limit" {
  description = "CPU limit for containers"
  type        = string
  default     = "1000m"
  
  validation {
    condition = can(regex("^[0-9]+m?$", var.cpu_limit))
    error_message = "CPU limit must be in format like '1000m' or '1'."
  }
}

variable "memory_limit" {
  description = "Memory limit for containers"
  type        = string
  default     = "2Gi"
  
  validation {
    condition = can(regex("^[0-9]+[GM]i?$", var.memory_limit))
    error_message = "Memory limit must be in format like '2Gi' or '1G'."
  }
}

variable "request_timeout" {
  description = "Request timeout in seconds"
  type        = number
  default     = 60
  
  validation {
    condition     = var.request_timeout >= 1 && var.request_timeout <= 3600
    error_message = "Request timeout must be between 1 and 3600 seconds."
  }
}

# =============================================================================
# Storage Configuration
# =============================================================================

variable "storage_location" {
  description = "Location for storage buckets"
  type        = string
  default     = "US"
}

variable "cors_origins" {
  description = "CORS origins for assets"
  type        = list(string)
  default     = ["*"]
}

# =============================================================================
# CDN Configuration
# =============================================================================

variable "enable_cdn" {
  description = "Enable Cloud CDN"
  type        = bool
  default     = true
}

variable "cdn_default_ttl" {
  description = "Default TTL for CDN cache in seconds"
  type        = number
  default     = 3600
  
  validation {
    condition     = var.cdn_default_ttl >= 0 && var.cdn_default_ttl <= 31536000
    error_message = "CDN default TTL must be between 0 and 31536000 seconds (1 year)."
  }
}

variable "cdn_max_ttl" {
  description = "Maximum TTL for CDN cache in seconds"
  type        = number
  default     = 86400
  
  validation {
    condition     = var.cdn_max_ttl >= 0 && var.cdn_max_ttl <= 31536000
    error_message = "CDN max TTL must be between 0 and 31536000 seconds (1 year)."
  }
}

variable "cdn_client_ttl" {
  description = "Client TTL for CDN cache in seconds"
  type        = number
  default     = 3600
  
  validation {
    condition     = var.cdn_client_ttl >= 0 && var.cdn_client_ttl <= 31536000
    error_message = "CDN client TTL must be between 0 and 31536000 seconds (1 year)."
  }
}

# =============================================================================
# Progressive Web App Configuration
# =============================================================================

variable "enable_pwa" {
  description = "Enable Progressive Web App features"
  type        = string
  default     = "true"
}

variable "pwa_name" {
  description = "PWA application name"
  type        = string
  default     = "DARWIN Platform"
}

variable "pwa_short_name" {
  description = "PWA short name"
  type        = string
  default     = "DARWIN"
}

variable "pwa_theme_color" {
  description = "PWA theme color"
  type        = string
  default     = "#000000"
}

variable "pwa_background_color" {
  description = "PWA background color"
  type        = string
  default     = "#ffffff"
}

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_analytics" {
  description = "Enable analytics tracking"
  type        = string
  default     = "false"
}

variable "enable_real_time" {
  description = "Enable real-time features"
  type        = string
  default     = "true"
}

variable "enable_offline_support" {
  description = "Enable offline support"
  type        = string
  default     = "true"
}

variable "enable_asset_optimization" {
  description = "Enable automatic asset optimization"
  type        = bool
  default     = true
}

variable "enable_service_worker" {
  description = "Enable service worker for caching"
  type        = bool
  default     = true
}

# =============================================================================
# Domain Configuration
# =============================================================================

variable "redirect_www" {
  description = "Redirect www subdomain to apex domain"
  type        = bool
  default     = true
}

variable "enable_hsts" {
  description = "Enable HTTP Strict Transport Security"
  type        = bool
  default     = true
}

variable "hsts_max_age" {
  description = "HSTS max age in seconds"
  type        = number
  default     = 31536000  # 1 year
  
  validation {
    condition     = var.hsts_max_age >= 0 && var.hsts_max_age <= 63072000
    error_message = "HSTS max age must be between 0 and 63072000 seconds (2 years)."
  }
}

# =============================================================================
# Performance Configuration
# =============================================================================

variable "enable_compression" {
  description = "Enable gzip compression"
  type        = bool
  default     = true
}

variable "enable_http2" {
  description = "Enable HTTP/2"
  type        = bool
  default     = true
}

variable "connection_draining_timeout" {
  description = "Connection draining timeout in seconds"
  type        = number
  default     = 300
  
  validation {
    condition     = var.connection_draining_timeout >= 0 && var.connection_draining_timeout <= 3600
    error_message = "Connection draining timeout must be between 0 and 3600 seconds."
  }
}

# =============================================================================
# Build and Deployment Configuration
# =============================================================================

variable "github_owner" {
  description = "GitHub repository owner"
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "GitHub branch for builds"
  type        = string
  default     = "main"
}

variable "build_timeout" {
  description = "Build timeout in seconds"
  type        = string
  default     = "1200s"
}

# =============================================================================
# Next.js Specific Configuration
# =============================================================================

variable "nextjs_build_command" {
  description = "Next.js build command"
  type        = string
  default     = "npm run build"
}

variable "nextjs_start_command" {
  description = "Next.js start command"
  type        = string
  default     = "npm start"
}

variable "node_version" {
  description = "Node.js version"
  type        = string
  default     = "18"
  
  validation {
    condition     = contains(["16", "18", "20"], var.node_version)
    error_message = "Node version must be 16, 18, or 20."
  }
}

variable "npm_registry" {
  description = "NPM registry URL"
  type        = string
  default     = "https://registry.npmjs.org/"
}

# =============================================================================
# Security Configuration
# =============================================================================

variable "enable_csrf_protection" {
  description = "Enable CSRF protection"
  type        = bool
  default     = true
}

variable "enable_rate_limiting" {
  description = "Enable rate limiting"
  type        = bool
  default     = true
}

variable "rate_limit_requests_per_minute" {
  description = "Rate limit requests per minute"
  type        = number
  default     = 100
  
  validation {
    condition     = var.rate_limit_requests_per_minute >= 1 && var.rate_limit_requests_per_minute <= 10000
    error_message = "Rate limit must be between 1 and 10000 requests per minute."
  }
}

variable "content_security_policy" {
  description = "Content Security Policy header"
  type        = string
  default     = "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:"
}

# =============================================================================
# Monitoring and Logging
# =============================================================================

variable "enable_access_logs" {
  description = "Enable access logging"
  type        = bool
  default     = true
}

variable "log_sample_rate" {
  description = "Log sampling rate (0.0 to 1.0)"
  type        = number
  default     = 1.0
  
  validation {
    condition     = var.log_sample_rate >= 0.0 && var.log_sample_rate <= 1.0
    error_message = "Log sample rate must be between 0.0 and 1.0."
  }
}

variable "enable_error_reporting" {
  description = "Enable error reporting"
  type        = bool
  default     = true
}

# =============================================================================
# Asset Management
# =============================================================================

variable "static_assets_max_age" {
  description = "Max age for static assets in seconds"
  type        = number
  default     = 31536000  # 1 year
}

variable "html_max_age" {
  description = "Max age for HTML files in seconds"
  type        = number
  default     = 0  # No cache for HTML
}

variable "api_max_age" {
  description = "Max age for API responses in seconds"
  type        = number
  default     = 0  # No cache for API
}

variable "image_optimization_quality" {
  description = "Image optimization quality (1-100)"
  type        = number
  default     = 85
  
  validation {
    condition     = var.image_optimization_quality >= 1 && var.image_optimization_quality <= 100
    error_message = "Image optimization quality must be between 1 and 100."
  }
}

# =============================================================================
# Environment Specific Configuration
# =============================================================================

variable "debug_mode" {
  description = "Enable debug mode (development only)"
  type        = bool
  default     = false
}

variable "hot_reload" {
  description = "Enable hot reload (development only)"
  type        = bool
  default     = false
}

variable "source_maps" {
  description = "Enable source maps"
  type        = bool
  default     = false
}

# =============================================================================
# Third-party Integrations
# =============================================================================

variable "analytics_id" {
  description = "Google Analytics tracking ID"
  type        = string
  default     = ""
}

variable "sentry_dsn" {
  description = "Sentry DSN for error tracking"
  type        = string
  default     = ""
}

variable "intercom_app_id" {
  description = "Intercom application ID"
  type        = string
  default     = ""
}

# =============================================================================
# Internationalization
# =============================================================================

variable "default_locale" {
  description = "Default locale"
  type        = string
  default     = "en"
}

variable "supported_locales" {
  description = "List of supported locales"
  type        = list(string)
  default     = ["en", "pt", "es"]
}

variable "enable_i18n" {
  description = "Enable internationalization"
  type        = bool
  default     = false
}

# =============================================================================
# Resource Optimization
# =============================================================================

variable "bundle_analyzer" {
  description = "Enable bundle analyzer"
  type        = bool
  default     = false
}

variable "tree_shaking" {
  description = "Enable tree shaking"
  type        = bool
  default     = true
}

variable "minification" {
  description = "Enable minification"
  type        = bool
  default     = true
}