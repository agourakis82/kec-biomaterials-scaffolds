# =============================================================================
# DARWIN Networking Module - Variables
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
  description = "GCP region for regional resources"
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

variable "subnet_cidr" {
  description = "CIDR range for the main subnet"
  type        = string
  default     = "10.0.0.0/24"
  
  validation {
    condition     = can(cidrhost(var.subnet_cidr, 0))
    error_message = "Subnet CIDR must be a valid CIDR block."
  }
}

# =============================================================================
# Optional Variables
# =============================================================================

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "ssl_domains" {
  description = "List of domains for SSL certificate"
  type        = list(string)
  default     = ["api.agourakis.med.br", "darwin.agourakis.med.br"]
}

variable "ssh_source_ranges" {
  description = "Source IP ranges allowed for SSH access"
  type        = list(string)
  default     = ["35.235.240.0/20"]  # Google Cloud Shell IPs
}

variable "allowed_source_ranges" {
  description = "Source IP ranges allowed for HTTP/HTTPS access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# =============================================================================
# VPC Configuration
# =============================================================================

variable "enable_private_google_access" {
  description = "Enable private Google access for the subnet"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "flow_logs_sampling" {
  description = "Sampling rate for VPC Flow Logs (0.0 to 1.0)"
  type        = number
  default     = 0.5
  
  validation {
    condition     = var.flow_logs_sampling >= 0.0 && var.flow_logs_sampling <= 1.0
    error_message = "Flow logs sampling must be between 0.0 and 1.0."
  }
}

# =============================================================================
# NAT Configuration
# =============================================================================

variable "enable_nat_gateway" {
  description = "Enable Cloud NAT gateway"
  type        = bool
  default     = true
}

variable "nat_min_ports_per_vm" {
  description = "Minimum number of ports allocated to a VM from NAT gateway"
  type        = number
  default     = 64
  
  validation {
    condition     = var.nat_min_ports_per_vm >= 2 && var.nat_min_ports_per_vm <= 65536
    error_message = "NAT min ports per VM must be between 2 and 65536."
  }
}

variable "nat_udp_idle_timeout" {
  description = "UDP idle timeout for NAT gateway (seconds)"
  type        = number
  default     = 30
  
  validation {
    condition     = var.nat_udp_idle_timeout >= 30 && var.nat_udp_idle_timeout <= 7200
    error_message = "NAT UDP idle timeout must be between 30 and 7200 seconds."
  }
}

variable "nat_tcp_established_idle_timeout" {
  description = "TCP established idle timeout for NAT gateway (seconds)"
  type        = number
  default     = 1200
  
  validation {
    condition     = var.nat_tcp_established_idle_timeout >= 30 && var.nat_tcp_established_idle_timeout <= 7200
    error_message = "NAT TCP established idle timeout must be between 30 and 7200 seconds."
  }
}

# =============================================================================
# VPC Connector Configuration
# =============================================================================

variable "vpc_connector_cidr" {
  description = "CIDR range for VPC connector"
  type        = string
  default     = "10.8.0.0/28"
  
  validation {
    condition     = can(cidrhost(var.vpc_connector_cidr, 0))
    error_message = "VPC connector CIDR must be a valid CIDR block."
  }
}

variable "vpc_connector_min_throughput" {
  description = "Minimum throughput for VPC connector (Mbps)"
  type        = number
  default     = 200
  
  validation {
    condition     = var.vpc_connector_min_throughput >= 200 && var.vpc_connector_min_throughput <= 1000
    error_message = "VPC connector min throughput must be between 200 and 1000 Mbps."
  }
}

variable "vpc_connector_max_throughput" {
  description = "Maximum throughput for VPC connector (Mbps)"
  type        = number
  default     = 1000
  
  validation {
    condition     = var.vpc_connector_max_throughput >= 200 && var.vpc_connector_max_throughput <= 1000
    error_message = "VPC connector max throughput must be between 200 and 1000 Mbps."
  }
}

# =============================================================================
# Load Balancer Configuration
# =============================================================================

variable "enable_https_redirect" {
  description = "Enable HTTP to HTTPS redirect"
  type        = bool
  default     = true
}

variable "enable_cdn" {
  description = "Enable Cloud CDN on the load balancer"
  type        = bool
  default     = true
}

variable "ssl_policy" {
  description = "SSL policy for the load balancer"
  type        = string
  default     = "MODERN"
  
  validation {
    condition     = contains(["COMPATIBLE", "MODERN", "RESTRICTED"], var.ssl_policy)
    error_message = "SSL policy must be COMPATIBLE, MODERN, or RESTRICTED."
  }
}

# =============================================================================
# Health Check Configuration
# =============================================================================

variable "backend_health_check_path" {
  description = "Path for backend health check"
  type        = string
  default     = "/health"
}

variable "backend_health_check_port" {
  description = "Port for backend health check"
  type        = number
  default     = 8080
}

variable "frontend_health_check_path" {
  description = "Path for frontend health check"
  type        = string
  default     = "/"
}

variable "frontend_health_check_port" {
  description = "Port for frontend health check"
  type        = number
  default     = 3000
}

variable "health_check_interval" {
  description = "Health check interval in seconds"
  type        = number
  default     = 30
  
  validation {
    condition     = var.health_check_interval >= 1 && var.health_check_interval <= 300
    error_message = "Health check interval must be between 1 and 300 seconds."
  }
}

variable "health_check_timeout" {
  description = "Health check timeout in seconds"
  type        = number
  default     = 10
  
  validation {
    condition     = var.health_check_timeout >= 1 && var.health_check_timeout <= 300
    error_message = "Health check timeout must be between 1 and 300 seconds."
  }
}

variable "health_check_healthy_threshold" {
  description = "Number of consecutive successful checks before marking healthy"
  type        = number
  default     = 2
  
  validation {
    condition     = var.health_check_healthy_threshold >= 1 && var.health_check_healthy_threshold <= 10
    error_message = "Health check healthy threshold must be between 1 and 10."
  }
}

variable "health_check_unhealthy_threshold" {
  description = "Number of consecutive failed checks before marking unhealthy"
  type        = number
  default     = 3
  
  validation {
    condition     = var.health_check_unhealthy_threshold >= 1 && var.health_check_unhealthy_threshold <= 10
    error_message = "Health check unhealthy threshold must be between 1 and 10."
  }
}

# =============================================================================
# Secondary IP Ranges
# =============================================================================

variable "services_secondary_cidr" {
  description = "Secondary CIDR range for services"
  type        = string
  default     = "192.168.0.0/20"
  
  validation {
    condition     = can(cidrhost(var.services_secondary_cidr, 0))
    error_message = "Services secondary CIDR must be a valid CIDR block."
  }
}

variable "pods_secondary_cidr" {
  description = "Secondary CIDR range for pods (if using GKE)"
  type        = string
  default     = "192.168.16.0/20"
  
  validation {
    condition     = can(cidrhost(var.pods_secondary_cidr, 0))
    error_message = "Pods secondary CIDR must be a valid CIDR block."
  }
}

# =============================================================================
# Firewall Configuration
# =============================================================================

variable "firewall_log_config" {
  description = "Enable firewall logging"
  type        = bool
  default     = true
}

variable "deny_all_priority" {
  description = "Priority for deny-all firewall rule"
  type        = number
  default     = 65534
  
  validation {
    condition     = var.deny_all_priority >= 0 && var.deny_all_priority <= 65534
    error_message = "Firewall priority must be between 0 and 65534."
  }
}