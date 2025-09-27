# =============================================================================
# DARWIN Networking Module - Main Configuration
# VPC, Subnets, Firewall, Load Balancer, Cloud NAT
# =============================================================================

# =============================================================================
# Local Values
# =============================================================================

locals {
  network_name = "${var.project_name}-${var.environment}-vpc"
  subnet_name  = "${var.project_name}-${var.environment}-subnet"
  
  # Common firewall tags
  firewall_tags = {
    backend  = "${var.project_name}-backend"
    frontend = "${var.project_name}-frontend"
    database = "${var.project_name}-database"
    redis    = "${var.project_name}-redis"
  }
}

# =============================================================================
# VPC Network
# =============================================================================

resource "google_compute_network" "main" {
  name                    = local.network_name
  project                = var.project_id
  auto_create_subnetworks = false
  mtu                    = 1460
  routing_mode           = "GLOBAL"
  
  description = "Main VPC network for ${var.project_name} ${var.environment} environment"
  
  # Delete default routes on creation
  delete_default_routes_on_create = false
}

# =============================================================================
# Subnet Configuration
# =============================================================================

resource "google_compute_subnetwork" "main" {
  name          = local.subnet_name
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = var.subnet_cidr
  
  description = "Main subnet for ${var.project_name} ${var.environment}"
  
  # Enable private Google access
  private_ip_google_access = true
  
  # Secondary IP ranges for services
  secondary_ip_range {
    range_name    = "services-range"
    ip_cidr_range = "192.168.0.0/20"
  }
  
  secondary_ip_range {
    range_name    = "pods-range"  
    ip_cidr_range = "192.168.16.0/20"
  }
  
  # Log configuration
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# =============================================================================
# Private Service Access for Managed Services (Redis, Cloud SQL)
# =============================================================================

# Reserve IP range for private services
resource "google_compute_global_address" "private_service_access" {
  name          = "${var.project_name}-${var.environment}-private-services"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
  
  description = "IP range reserved for private service access"
}

# Create private service connection
resource "google_service_networking_connection" "private_service_access" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_access.name]
}

# =============================================================================
# Cloud Router for NAT
# =============================================================================

resource "google_compute_router" "main" {
  name    = "${var.project_name}-${var.environment}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.main.id
  
  description = "Cloud Router for NAT gateway"
  
  bgp {
    asn = 64514
  }
}

# =============================================================================
# Cloud NAT Gateway
# =============================================================================

resource "google_compute_router_nat" "main" {
  name                               = "${var.project_name}-${var.environment}-nat"
  project                           = var.project_id
  router                            = google_compute_router.main.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  # Log configuration
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
  
  # Minimum ports per VM
  min_ports_per_vm = 64
  
  # UDP idle timeout
  udp_idle_timeout_sec = 30
  
  # ICMP idle timeout  
  icmp_idle_timeout_sec = 30
  
  # TCP established idle timeout
  tcp_established_idle_timeout_sec = 1200
  
  # TCP transitory idle timeout
  tcp_transitory_idle_timeout_sec = 30
}

# =============================================================================
# VPC Connector for Cloud Run
# =============================================================================

resource "google_vpc_access_connector" "main" {
  name          = "${var.project_name}-${var.environment}-conn"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.name
  ip_cidr_range = "10.8.0.0/28"
  
  min_throughput = 200
  max_throughput = 1000
  
  depends_on = [google_compute_subnetwork.main]
}

# =============================================================================
# Firewall Rules
# =============================================================================

# Allow HTTP traffic
resource "google_compute_firewall" "allow_http" {
  name    = "${var.project_name}-${var.environment}-allow-http"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow HTTP traffic"
  
  allow {
    protocol = "tcp"
    ports    = ["80"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = [local.firewall_tags.frontend, local.firewall_tags.backend]
}

# Allow HTTPS traffic
resource "google_compute_firewall" "allow_https" {
  name    = "${var.project_name}-${var.environment}-allow-https"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow HTTPS traffic"
  
  allow {
    protocol = "tcp"
    ports    = ["443"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = [local.firewall_tags.frontend, local.firewall_tags.backend]
}

# Allow SSH (restricted)
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_name}-${var.environment}-allow-ssh"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow SSH access for debugging"
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = var.ssh_source_ranges
  target_tags   = ["ssh-allowed"]
}

# Allow Cloud Run to Database
resource "google_compute_firewall" "allow_database" {
  name    = "${var.project_name}-${var.environment}-allow-database"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow Cloud Run to access database"
  
  allow {
    protocol = "tcp"
    ports    = ["5432"]  # PostgreSQL
  }
  
  source_tags = [local.firewall_tags.backend]
  target_tags = [local.firewall_tags.database]
}

# Allow Cloud Run to Redis
resource "google_compute_firewall" "allow_redis" {
  name    = "${var.project_name}-${var.environment}-allow-redis"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow Cloud Run to access Redis"
  
  allow {
    protocol = "tcp"
    ports    = ["6379"]  # Redis
  }
  
  source_tags = [local.firewall_tags.backend]
  target_tags = [local.firewall_tags.redis]
}

# Allow health check probes
resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.project_name}-${var.environment}-allow-health-check"
  project = var.project_id
  network = google_compute_network.main.name
  
  description = "Allow Google Cloud health check probes"
  
  allow {
    protocol = "tcp"
    ports    = ["8080", "8000", "80", "443"]
  }
  
  # Google Cloud health check IP ranges
  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]
  
  target_tags = [local.firewall_tags.backend, local.firewall_tags.frontend]
}

# Deny all other traffic (implicit)
resource "google_compute_firewall" "deny_all" {
  name    = "${var.project_name}-${var.environment}-deny-all"
  project = var.project_id
  network = google_compute_network.main.name
  priority = 65534
  
  description = "Deny all other traffic (explicit)"
  
  deny {
    protocol = "all"
  }
  
  source_ranges = ["0.0.0.0/0"]
  
  # Exclude specific tags that should have access
  target_tags = ["deny-all"]
}

# =============================================================================
# Global Load Balancer Components
# =============================================================================

# Global IP address for load balancer
resource "google_compute_global_address" "main" {
  name         = "${var.project_name}-${var.environment}-global-ip"
  project      = var.project_id
  address_type = "EXTERNAL"
  
  description = "Global IP address for load balancer"
}

# HTTP to HTTPS redirect
resource "google_compute_url_map" "https_redirect" {
  name    = "${var.project_name}-${var.environment}-https-redirect"
  project = var.project_id
  
  description = "Redirect HTTP to HTTPS"
  
  default_url_redirect {
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    https_redirect        = true
    strip_query           = false
  }
}

# Target HTTP proxy for redirect
resource "google_compute_target_http_proxy" "https_redirect" {
  name    = "${var.project_name}-${var.environment}-http-proxy"
  project = var.project_id
  url_map = google_compute_url_map.https_redirect.id
  
  description = "HTTP proxy for HTTPS redirect"
}

# Global forwarding rule for HTTP (redirect)
resource "google_compute_global_forwarding_rule" "http" {
  name       = "${var.project_name}-${var.environment}-http-forwarding-rule"
  project    = var.project_id
  target     = google_compute_target_http_proxy.https_redirect.id
  port_range = "80"
  ip_address = google_compute_global_address.main.address
  
  description = "Forward HTTP traffic for redirect"
}

# =============================================================================
# SSL Certificate (Managed)
# =============================================================================

resource "google_compute_managed_ssl_certificate" "main" {
  name    = "${var.project_name}-${var.environment}-ssl-cert"
  project = var.project_id
  
  managed {
    domains = var.ssl_domains
  }
  
  description = "Managed SSL certificate for ${var.project_name}"
  
  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# Backend Services (will be configured by other modules)
# =============================================================================

# Health check for backend services
resource "google_compute_health_check" "backend" {
  name    = "${var.project_name}-${var.environment}-backend-health"
  project = var.project_id
  
  description = "Health check for backend services"
  
  http_health_check {
    port               = 8080
    request_path       = "/health"
  }
  check_interval_sec = 30
  timeout_sec        = 10
  healthy_threshold  = 2
  unhealthy_threshold = 3
  
  log_config {
    enable = true
  }
}

resource "google_compute_health_check" "frontend" {
  name    = "${var.project_name}-${var.environment}-frontend-health"
  project = var.project_id
  
  description = "Health check for frontend services"
  
  http_health_check {
    port               = 3000
    request_path       = "/"
  }
  check_interval_sec = 30
  timeout_sec        = 10
  healthy_threshold  = 2
  unhealthy_threshold = 3
  
  log_config {
    enable = true
  }
}