# =============================================================================
# DARWIN Networking Module - Outputs
# =============================================================================

# =============================================================================
# VPC Network Outputs
# =============================================================================

output "vpc_id" {
  description = "The ID of the VPC network"
  value       = google_compute_network.main.id
}

output "vpc_name" {
  description = "The name of the VPC network"
  value       = google_compute_network.main.name
}

output "vpc_self_link" {
  description = "The self link of the VPC network"
  value       = google_compute_network.main.self_link
}

output "vpc_routing_mode" {
  description = "The routing mode of the VPC network"
  value       = google_compute_network.main.routing_mode
}

# =============================================================================
# Subnet Outputs
# =============================================================================

output "subnet_id" {
  description = "The ID of the main subnet"
  value       = google_compute_subnetwork.main.id
}

output "subnet_name" {
  description = "The name of the main subnet"
  value       = google_compute_subnetwork.main.name
}

output "subnet_self_link" {
  description = "The self link of the main subnet"
  value       = google_compute_subnetwork.main.self_link
}

output "subnet_cidr" {
  description = "The CIDR range of the main subnet"
  value       = google_compute_subnetwork.main.ip_cidr_range
}

output "subnet_gateway_address" {
  description = "The gateway address of the subnet"
  value       = google_compute_subnetwork.main.gateway_address
}

output "services_secondary_range_name" {
  description = "The name of the services secondary IP range"
  value       = google_compute_subnetwork.main.secondary_ip_range[0].range_name
}

output "services_secondary_cidr" {
  description = "The CIDR of the services secondary IP range"
  value       = google_compute_subnetwork.main.secondary_ip_range[0].ip_cidr_range
}

output "pods_secondary_range_name" {
  description = "The name of the pods secondary IP range"
  value       = google_compute_subnetwork.main.secondary_ip_range[1].range_name
}

output "pods_secondary_cidr" {
  description = "The CIDR of the pods secondary IP range"
  value       = google_compute_subnetwork.main.secondary_ip_range[1].ip_cidr_range
}

# =============================================================================
# Cloud Router and NAT Outputs
# =============================================================================

output "cloud_router_name" {
  description = "The name of the Cloud Router"
  value       = google_compute_router.main.name
}

output "cloud_router_self_link" {
  description = "The self link of the Cloud Router"
  value       = google_compute_router.main.self_link
}

output "nat_gateway_name" {
  description = "The name of the Cloud NAT gateway"
  value       = google_compute_router_nat.main.name
}

# =============================================================================
# VPC Connector Outputs
# =============================================================================

output "vpc_connector_name" {
  description = "The name of the VPC connector for Cloud Run"
  value       = google_vpc_access_connector.main.name
}

output "vpc_connector_id" {
  description = "The ID of the VPC connector"
  value       = google_vpc_access_connector.main.id
}

output "vpc_connector_self_link" {
  description = "The self link of the VPC connector"
  value       = google_vpc_access_connector.main.self_link
}

output "vpc_connector_state" {
  description = "The state of the VPC connector"
  value       = google_vpc_access_connector.main.state
}

# =============================================================================
# Global Load Balancer Outputs
# =============================================================================

output "load_balancer_ip" {
  description = "The global IP address of the load balancer"
  value       = google_compute_global_address.main.address
}

output "load_balancer_ip_id" {
  description = "The ID of the global IP address"
  value       = google_compute_global_address.main.id
}

output "load_balancer_ip_self_link" {
  description = "The self link of the global IP address"
  value       = google_compute_global_address.main.self_link
}

# =============================================================================
# SSL Certificate Outputs
# =============================================================================

output "ssl_certificate_name" {
  description = "The name of the managed SSL certificate"
  value       = google_compute_managed_ssl_certificate.main.name
}

output "ssl_certificate_id" {
  description = "The ID of the managed SSL certificate"
  value       = google_compute_managed_ssl_certificate.main.id
}

output "ssl_certificate_self_link" {
  description = "The self link of the managed SSL certificate"
  value       = google_compute_managed_ssl_certificate.main.self_link
}

output "ssl_certificate_domains" {
  description = "The domains covered by the SSL certificate"
  value       = google_compute_managed_ssl_certificate.main.managed[0].domains
}

# output "ssl_certificate_status" {
#   description = "The status of the managed SSL certificate"
#   value       = google_compute_managed_ssl_certificate.main.managed[0].status
# }

# =============================================================================
# Health Check Outputs
# =============================================================================

output "backend_health_check_name" {
  description = "The name of the backend health check"
  value       = google_compute_health_check.backend.name
}

output "backend_health_check_id" {
  description = "The ID of the backend health check"
  value       = google_compute_health_check.backend.id
}

output "backend_health_check_self_link" {
  description = "The self link of the backend health check"
  value       = google_compute_health_check.backend.self_link
}

output "frontend_health_check_name" {
  description = "The name of the frontend health check"
  value       = google_compute_health_check.frontend.name
}

output "frontend_health_check_id" {
  description = "The ID of the frontend health check"
  value       = google_compute_health_check.frontend.id
}

output "frontend_health_check_self_link" {
  description = "The self link of the frontend health check"
  value       = google_compute_health_check.frontend.self_link
}

# =============================================================================
# Firewall Rules Outputs
# =============================================================================

output "firewall_rules" {
  description = "Map of firewall rule names and their IDs"
  value = {
    allow_http         = google_compute_firewall.allow_http.name
    allow_https        = google_compute_firewall.allow_https.name
    allow_ssh          = google_compute_firewall.allow_ssh.name
    allow_database     = google_compute_firewall.allow_database.name
    allow_redis        = google_compute_firewall.allow_redis.name
    allow_health_check = google_compute_firewall.allow_health_check.name
    deny_all          = google_compute_firewall.deny_all.name
  }
}

output "firewall_tags" {
  description = "Map of firewall tags for different service types"
  value = {
    backend  = "${var.project_name}-backend"
    frontend = "${var.project_name}-frontend"
    database = "${var.project_name}-database"
    redis    = "${var.project_name}-redis"
  }
}

# =============================================================================
# HTTP/HTTPS Redirect Outputs
# =============================================================================

output "https_redirect_url_map_name" {
  description = "The name of the HTTPS redirect URL map"
  value       = google_compute_url_map.https_redirect.name
}

output "https_redirect_url_map_id" {
  description = "The ID of the HTTPS redirect URL map"
  value       = google_compute_url_map.https_redirect.id
}

output "http_proxy_name" {
  description = "The name of the HTTP target proxy"
  value       = google_compute_target_http_proxy.https_redirect.name
}

output "http_proxy_id" {
  description = "The ID of the HTTP target proxy"
  value       = google_compute_target_http_proxy.https_redirect.id
}

output "http_forwarding_rule_name" {
  description = "The name of the HTTP forwarding rule"
  value       = google_compute_global_forwarding_rule.http.name
}

output "http_forwarding_rule_id" {
  description = "The ID of the HTTP forwarding rule"
  value       = google_compute_global_forwarding_rule.http.id
}

# =============================================================================
# Network Configuration Summary
# =============================================================================

output "network_config" {
  description = "Summary of network configuration"
  value = {
    vpc_name              = google_compute_network.main.name
    subnet_name           = google_compute_subnetwork.main.name
    subnet_cidr           = google_compute_subnetwork.main.ip_cidr_range
    load_balancer_ip      = google_compute_global_address.main.address
    ssl_certificate_name  = google_compute_managed_ssl_certificate.main.name
    vpc_connector_name    = google_vpc_access_connector.main.name
    nat_gateway_name      = google_compute_router_nat.main.name
    region               = var.region
    environment          = var.environment
  }
}

# =============================================================================
# DNS Configuration Requirements
# =============================================================================

output "dns_configuration" {
  description = "DNS configuration required for domains"
  value = {
    domains = var.ssl_domains
    ip_address = google_compute_global_address.main.address
    record_type = "A"
    ttl = 300
  }
}

# =============================================================================
# Security Configuration
# =============================================================================

output "security_config" {
  description = "Security configuration summary"
  value = {
    private_google_access = google_compute_subnetwork.main.private_ip_google_access
    flow_logs_enabled    = google_compute_subnetwork.main.log_config != null
    nat_gateway_enabled  = google_compute_router_nat.main.name != null
    firewall_rules_count = length([
      google_compute_firewall.allow_http.name,
      google_compute_firewall.allow_https.name,
      google_compute_firewall.allow_ssh.name,
      google_compute_firewall.allow_database.name,
      google_compute_firewall.allow_redis.name,
      google_compute_firewall.allow_health_check.name,
      google_compute_firewall.deny_all.name
    ])
#    ssl_certificate_status = google_compute_managed_ssl_certificate.main.managed[0].status
  }
}