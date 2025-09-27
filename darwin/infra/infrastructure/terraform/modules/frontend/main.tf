# =============================================================================
# DARWIN Frontend Module - Main Configuration
# Cloud Run, Cloud CDN, Static Assets, Custom Domain
# =============================================================================

# =============================================================================
# Local Values
# =============================================================================

locals {
  service_name = "${var.project_name}-${var.environment}-frontend"
  
  # CDN bucket names
  cdn_bucket_name = "${var.project_name}-${var.environment}-cdn-assets"
  static_bucket_name = "${var.project_name}-${var.environment}-static-assets"
  
  # Service account names
  service_account_id = "${var.project_name}-${var.environment}-frontend-sa"
}

# =============================================================================
# Service Account for Frontend
# =============================================================================

resource "google_service_account" "frontend" {
  account_id   = local.service_account_id
  display_name = "DARWIN Frontend Service Account"
  description  = "Service account for ${var.project_name} frontend services in ${var.environment}"
  project      = var.project_id
}

# IAM roles for the service account
resource "google_project_iam_member" "frontend_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

# =============================================================================
# Cloud Storage for Static Assets
# =============================================================================

resource "google_storage_bucket" "cdn_assets" {
  name          = local.cdn_bucket_name
  project       = var.project_id
  location      = var.storage_location
  storage_class = "STANDARD"
  
  # Uniform bucket-level access
  uniform_bucket_level_access = true
  
  # Website configuration
  website {
    main_page_suffix = "index.html"
    not_found_page   = "404.html"
  }
  
  # CORS configuration
  cors {
    origin          = var.cors_origins
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  # Lifecycle management for optimized storage costs
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  labels = var.common_labels
}

resource "google_storage_bucket" "static_assets" {
  name          = local.static_bucket_name
  project       = var.project_id
  location      = var.storage_location
  storage_class = "STANDARD"
  
  # Uniform bucket-level access
  uniform_bucket_level_access = true
  
  # CORS configuration
  cors {
    origin          = var.cors_origins
    method          = ["GET", "HEAD"]
    response_header = ["*"]
    max_age_seconds = 86400  # 24 hours
  }
  
  labels = var.common_labels
}

# Make CDN bucket publicly readable
resource "google_storage_bucket_iam_member" "cdn_public_read" {
  bucket = google_storage_bucket.cdn_assets.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Make static assets bucket publicly readable
resource "google_storage_bucket_iam_member" "static_public_read" {
  bucket = google_storage_bucket.static_assets.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Frontend service account access to buckets
resource "google_storage_bucket_iam_member" "frontend_cdn_access" {
  bucket = google_storage_bucket.cdn_assets.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_storage_bucket_iam_member" "frontend_static_access" {
  bucket = google_storage_bucket.static_assets.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.frontend.email}"
}

# =============================================================================
# Cloud CDN Backend Buckets
# =============================================================================

resource "google_compute_backend_bucket" "cdn_backend" {
  name        = "${var.project_name}-${var.environment}-cdn-backend"
  project     = var.project_id
  bucket_name = google_storage_bucket.cdn_assets.name
  enable_cdn  = var.enable_cdn
  
  description = "Backend bucket for CDN assets"
  
  dynamic "cdn_policy" {
    for_each = var.enable_cdn ? [1] : []
    content {
      cache_mode                   = "CACHE_ALL_STATIC"
      default_ttl                 = var.cdn_default_ttl
      max_ttl                     = var.cdn_max_ttl
      client_ttl                  = var.cdn_client_ttl
      negative_caching            = true
      negative_caching_policy {
        code = 404
        ttl  = 120
      }
      serve_while_stale           = 86400
      signed_url_cache_max_age_sec = 7200
    }
  }
}

resource "google_compute_backend_bucket" "static_backend" {
  name        = "${var.project_name}-${var.environment}-static-backend"
  project     = var.project_id
  bucket_name = google_storage_bucket.static_assets.name
  enable_cdn  = var.enable_cdn
  
  description = "Backend bucket for static assets"
  
  dynamic "cdn_policy" {
    for_each = var.enable_cdn ? [1] : []
    content {
      cache_mode                   = "CACHE_ALL_STATIC"
      default_ttl                 = 86400  # 24 hours for static assets
      max_ttl                     = 604800  # 7 days
      client_ttl                  = 86400
      negative_caching            = true
      serve_while_stale           = 86400
      signed_url_cache_max_age_sec = 7200
    }
  }
}

# =============================================================================
# Cloud Run Service for Frontend
# =============================================================================

resource "google_cloud_run_v2_service" "frontend" {
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
    service_account = google_service_account.frontend.email
    
    containers {
      name  = "frontend"
      image = var.frontend_image
      
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
        name = "NODE_ENV"
        value = var.environment == "production" ? "production" : "development"
      }
      
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
      
      # API URL for backend communication
      env {
        name = "NEXT_PUBLIC_API_URL"
        value = var.api_url
      }
      
      env {
        name = "API_URL"
        value = var.api_url
      }
      
      # CDN URLs for assets
      env {
        name = "NEXT_PUBLIC_CDN_URL"
        value = var.enable_cdn ? "https://cdn.${var.frontend_domain}" : google_storage_bucket.cdn_assets.url
      }
      
      env {
        name = "NEXT_PUBLIC_STATIC_URL"
        value = google_storage_bucket.static_assets.url
      }
      
      # Frontend domain
      env {
        name = "NEXT_PUBLIC_FRONTEND_URL"
        value = "https://${var.frontend_domain}"
      }
      
      # Performance configurations
      env {
        name = "NEXT_PUBLIC_ENABLE_PWA"
        value = var.enable_pwa
      }
      
      env {
        name = "NEXT_PUBLIC_ENABLE_ANALYTICS"
        value = var.enable_analytics
      }
      
      # Feature flags
      env {
        name = "NEXT_PUBLIC_ENABLE_REAL_TIME"
        value = var.enable_real_time
      }
      
      env {
        name = "NEXT_PUBLIC_ENABLE_OFFLINE"
        value = var.enable_offline_support
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
        tcp_socket {
          port = 8080
        }
      }
      
      # Liveness probe
      liveness_probe {
        initial_delay_seconds = 30
        timeout_seconds      = 5
        period_seconds       = 30
        failure_threshold    = 3
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
    google_storage_bucket.cdn_assets,
    google_storage_bucket.static_assets
  ]
}

# =============================================================================
# Cloud Run Backend Service for Load Balancer
# =============================================================================

resource "google_compute_region_network_endpoint_group" "frontend_neg" {
  name                  = "${var.project_name}-${var.environment}-frontend-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  project               = var.project_id
  
  cloud_run {
    service = google_cloud_run_v2_service.frontend.name
  }
}

resource "google_compute_backend_service" "frontend" {
  name        = "${var.project_name}-${var.environment}-frontend-backend"
  project     = var.project_id
  protocol    = "HTTP"
  port_name   = "http"
  
  backend {
    group = google_compute_region_network_endpoint_group.frontend_neg.id
  }
  
  # CDN configuration
  dynamic "cdn_policy" {
    for_each = var.enable_cdn ? [1] : []
    content {
      cache_mode                   = "USE_ORIGIN_HEADERS"
      default_ttl                 = var.cdn_default_ttl
      max_ttl                     = var.cdn_max_ttl
      client_ttl                  = var.cdn_client_ttl
      negative_caching            = true
      serve_while_stale           = 86400
      signed_url_cache_max_age_sec = 7200
      
      cache_key_policy {
        include_host         = true
        include_protocol     = true
        include_query_string = false
      }
    }
  }
  
  enable_cdn = var.enable_cdn
  
  # Health check - will use the one from networking module
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# =============================================================================
# URL Map for Load Balancer
# =============================================================================

resource "google_compute_url_map" "frontend" {
  name    = "${var.project_name}-${var.environment}-frontend-urlmap"
  project = var.project_id
  
  description = "URL map for frontend load balancer"
  
  # Default service (frontend)
  default_service = google_compute_backend_service.frontend.id
  
  # Route for CDN assets
  path_matcher {
    name            = "cdn-matcher"
    default_service = google_compute_backend_service.frontend.id
    
    path_rule {
      paths   = ["/cdn/*", "/assets/*", "/_next/static/*"]
      service = google_compute_backend_bucket.cdn_backend.id
    }
    
    path_rule {
      paths   = ["/static/*", "/images/*", "/icons/*"]
      service = google_compute_backend_bucket.static_backend.id
    }
  }
  
  # Host rules for different domains
  host_rule {
    hosts        = [var.frontend_domain]
    path_matcher = "cdn-matcher"
  }
  
  # Redirect www to non-www if applicable
  dynamic "host_rule" {
    for_each = var.redirect_www ? [1] : []
    content {
      hosts        = ["www.${var.frontend_domain}"]
      path_matcher = "www-redirect"
    }
  }
  
  dynamic "path_matcher" {
    for_each = var.redirect_www ? [1] : []
    content {
      name = "www-redirect"
      default_url_redirect {
        host_redirect          = var.frontend_domain
        redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
        strip_query           = false
      }
    }
  }
}

# =============================================================================
# HTTPS Target Proxy
# =============================================================================

resource "google_compute_target_https_proxy" "frontend" {
  name    = "${var.project_name}-${var.environment}-frontend-https-proxy"
  project = var.project_id
  url_map = google_compute_url_map.frontend.id
  ssl_certificates = [
    var.ssl_certificate_id
  ]
  
  description = "HTTPS proxy for frontend"
}

# =============================================================================
# Global Forwarding Rule for HTTPS
# =============================================================================

resource "google_compute_global_forwarding_rule" "frontend_https" {
  name       = "${var.project_name}-${var.environment}-frontend-https-forwarding-rule"
  project    = var.project_id
  target     = google_compute_target_https_proxy.frontend.id
  port_range = "443"
  ip_address = var.load_balancer_ip
  
  description = "Forward HTTPS traffic for frontend"
}

# =============================================================================
# Domain Mapping for Frontend
# =============================================================================

resource "google_cloud_run_domain_mapping" "frontend_domain" {
  location = var.region
  name     = var.frontend_domain
  project  = var.project_id
  
  metadata {
    namespace = var.project_id
  }
  
  spec {
    route_name = google_cloud_run_v2_service.frontend.name
  }
}

# =============================================================================
# IAM Policy for Cloud Run Service
# =============================================================================

resource "google_cloud_run_service_iam_policy" "noauth" {
  location = google_cloud_run_v2_service.frontend.location
  project  = google_cloud_run_v2_service.frontend.project
  service  = google_cloud_run_v2_service.frontend.name
  
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

# =============================================================================
# Progressive Web App Manifest Upload
# =============================================================================

resource "google_storage_bucket_object" "pwa_manifest" {
  count  = var.enable_pwa ? 1 : 0
  name   = "manifest.json"
  bucket = google_storage_bucket.static_assets.name
  content = jsonencode({
    name             = "DARWIN Platform"
    short_name       = "DARWIN"
    description      = "DARWIN Biomaterials Research Platform"
    start_url        = "/"
    display          = "standalone"
    theme_color      = "#000000"
    background_color = "#ffffff"
    icons = [
      {
        src   = "/icons/icon-192x192.png"
        sizes = "192x192"
        type  = "image/png"
      },
      {
        src   = "/icons/icon-512x512.png"
        sizes = "512x512"
        type  = "image/png"
      }
    ]
  })
  
  content_type = "application/json"
  
  metadata = {
    Cache-Control = "public, max-age=3600"
  }
}

# Service worker for PWA
resource "google_storage_bucket_object" "service_worker" {
  count  = var.enable_pwa ? 1 : 0
  name   = "sw.js"
  bucket = google_storage_bucket.static_assets.name
  source = "${path.module}/assets/sw.js"
  
  content_type = "application/javascript"
  
  metadata = {
    Cache-Control = "public, max-age=0, must-revalidate"
  }
}

