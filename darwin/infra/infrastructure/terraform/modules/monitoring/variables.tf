# =============================================================================
# DARWIN Monitoring Module - Variables
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

variable "backend_service_name" {
  description = "Name of the backend Cloud Run service to monitor"
  type        = string
}

variable "frontend_service_name" {
  description = "Name of the frontend Cloud Run service to monitor"
  type        = string
}

variable "database_instance_id" {
  description = "ID of the database instance to monitor"
  type        = string
}

variable "redis_instance_id" {
  description = "ID of the Redis instance to monitor"
  type        = string
}

variable "api_domain" {
  description = "API domain for uptime checks"
  type        = string
}

variable "frontend_domain" {
  description = "Frontend domain for uptime checks"
  type        = string
}

variable "budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
}

variable "billing_account_id" {
  description = "Billing account ID for budget alerts"
  type        = string
}

# =============================================================================
# Notification Channels
# =============================================================================

variable "notification_channels" {
  description = "List of existing notification channel IDs"
  type        = list(string)
  default     = []
}

variable "email_addresses" {
  description = "List of email addresses for notifications"
  type        = list(string)
  default     = []
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "slack_channel_name" {
  description = "Slack channel name for notifications"
  type        = string
  default     = "#alerts"
}

variable "pagerduty_service_key" {
  description = "PagerDuty service integration key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "sms_number" {
  description = "SMS phone number for critical alerts"
  type        = string
  default     = ""
}

# =============================================================================
# Alert Thresholds
# =============================================================================

variable "error_rate_threshold" {
  description = "Error rate threshold (0-1) for alerting"
  type        = number
  default     = 0.05  # 5%
  
  validation {
    condition     = var.error_rate_threshold >= 0 && var.error_rate_threshold <= 1
    error_message = "Error rate threshold must be between 0 and 1."
  }
}

variable "latency_threshold_ms" {
  description = "Latency threshold in milliseconds for alerting"
  type        = number
  default     = 5000  # 5 seconds
  
  validation {
    condition     = var.latency_threshold_ms > 0
    error_message = "Latency threshold must be greater than 0."
  }
}

variable "cpu_threshold_percent" {
  description = "CPU utilization threshold percentage for alerting"
  type        = number
  default     = 80
  
  validation {
    condition     = var.cpu_threshold_percent >= 0 && var.cpu_threshold_percent <= 100
    error_message = "CPU threshold must be between 0 and 100."
  }
}

variable "memory_threshold_percent" {
  description = "Memory utilization threshold percentage for alerting"
  type        = number
  default     = 85
  
  validation {
    condition     = var.memory_threshold_percent >= 0 && var.memory_threshold_percent <= 100
    error_message = "Memory threshold must be between 0 and 100."
  }
}

variable "database_connections_threshold" {
  description = "Database connections threshold for alerting"
  type        = number
  default     = 80
  
  validation {
    condition     = var.database_connections_threshold > 0
    error_message = "Database connections threshold must be greater than 0."
  }
}

variable "redis_memory_threshold_percent" {
  description = "Redis memory usage threshold percentage for alerting"
  type        = number
  default     = 85
  
  validation {
    condition     = var.redis_memory_threshold_percent >= 0 && var.redis_memory_threshold_percent <= 100
    error_message = "Redis memory threshold must be between 0 and 100."
  }
}

# =============================================================================
# Uptime Check Configuration
# =============================================================================

variable "uptime_check_timeout" {
  description = "Timeout for uptime checks in seconds"
  type        = string
  default     = "10s"
}

variable "uptime_check_period" {
  description = "Period between uptime checks"
  type        = string
  default     = "300s"
}

variable "uptime_check_regions" {
  description = "List of regions for uptime checks"
  type        = list(string)
  default = [
    "USA_OREGON",
    "USA_VIRGINIA",
    "EUROPE_IRELAND",
    "ASIA_SINGAPORE"
  ]
}

variable "api_health_path" {
  description = "Health check path for API service"
  type        = string
  default     = "/health"
}

variable "frontend_health_path" {
  description = "Health check path for frontend service"
  type        = string
  default     = "/"
}

# =============================================================================
# Dashboard Configuration
# =============================================================================

variable "dashboard_title" {
  description = "Title for the main dashboard"
  type        = string
  default     = ""
}

variable "dashboard_columns" {
  description = "Number of columns in dashboard layout"
  type        = number
  default     = 12
  
  validation {
    condition     = var.dashboard_columns >= 1 && var.dashboard_columns <= 12
    error_message = "Dashboard columns must be between 1 and 12."
  }
}

variable "enable_custom_dashboard" {
  description = "Enable creation of custom dashboard"
  type        = bool
  default     = true
}

# =============================================================================
# SLO Configuration
# =============================================================================

variable "api_availability_slo" {
  description = "API availability SLO target (0-1)"
  type        = number
  default     = 0.995  # 99.5%
  
  validation {
    condition     = var.api_availability_slo >= 0.9 && var.api_availability_slo <= 1.0
    error_message = "API availability SLO must be between 0.9 and 1.0."
  }
}

variable "api_latency_slo" {
  description = "API latency SLO target in milliseconds"
  type        = number
  default     = 2000  # 2 seconds
  
  validation {
    condition     = var.api_latency_slo > 0
    error_message = "API latency SLO must be greater than 0."
  }
}

variable "slo_rolling_period_days" {
  description = "Rolling period for SLO calculation in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.slo_rolling_period_days >= 1 && var.slo_rolling_period_days <= 365
    error_message = "SLO rolling period must be between 1 and 365 days."
  }
}

# =============================================================================
# Budget Alert Configuration
# =============================================================================

variable "budget_alert_thresholds" {
  description = "Budget alert threshold percentages"
  type        = list(number)
  default     = [0.5, 0.8, 1.0, 1.2]  # 50%, 80%, 100%, 120%
}

variable "enable_budget_alerts" {
  description = "Enable budget alerting"
  type        = bool
  default     = true
}

variable "budget_services" {
  description = "List of service IDs to include in budget monitoring"
  type        = list(string)
  default = [
    "6F81-5844-456A",  # Compute Engine
    "95FF-2EF5-5EA1",  # Cloud SQL
    "24E6-581D-38E5",  # Cloud Storage
    "C654-D5A5-5EC1",  # Cloud Run
    "58CD-8F1E-C991",  # Cloud Load Balancing
    "F25E-97C4-4556",  # Cloud CDN
    "95FF-2EF5-5EA1"   # Cloud Monitoring
  ]
}

# =============================================================================
# Log-based Metrics Configuration
# =============================================================================

variable "enable_log_metrics" {
  description = "Enable creation of log-based metrics"
  type        = bool
  default     = true
}

variable "error_log_filter" {
  description = "Filter for error log entries"
  type        = string
  default     = "severity >= ERROR"
}

variable "slow_query_threshold_ms" {
  description = "Threshold for slow query detection in milliseconds"
  type        = number
  default     = 5000
  
  validation {
    condition     = var.slow_query_threshold_ms > 0
    error_message = "Slow query threshold must be greater than 0."
  }
}

# =============================================================================
# Alert Policy Configuration
# =============================================================================

variable "alert_auto_close_duration" {
  description = "Auto-close duration for alert policies"
  type        = string
  default     = "1800s"  # 30 minutes
}

variable "alert_documentation_format" {
  description = "Format for alert documentation"
  type        = string
  default     = "text/markdown"
}

variable "enable_uptime_alerts" {
  description = "Enable uptime check alerts"
  type        = bool
  default     = true
}

variable "enable_performance_alerts" {
  description = "Enable performance-related alerts"
  type        = bool
  default     = true
}

variable "enable_resource_alerts" {
  description = "Enable resource usage alerts"
  type        = bool
  default     = true
}

variable "critical_alert_channels" {
  description = "Notification channels for critical alerts"
  type        = list(string)
  default     = []
}

# =============================================================================
# Monitoring Filters
# =============================================================================

variable "service_filter" {
  description = "Filter for services to monitor"
  type        = string
  default     = ""
}

variable "resource_filter" {
  description = "Filter for resources to monitor"
  type        = string
  default     = ""
}

# =============================================================================
# Advanced Configuration
# =============================================================================

variable "enable_distributed_tracing" {
  description = "Enable distributed tracing"
  type        = bool
  default     = true
}

variable "enable_profiling" {
  description = "Enable application profiling"
  type        = bool
  default     = false
}

variable "enable_debugging" {
  description = "Enable cloud debugging"
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.log_retention_days >= 1 && var.log_retention_days <= 3653
    error_message = "Log retention days must be between 1 and 3653 days."
  }
}

variable "metrics_retention_days" {
  description = "Metrics retention period in days"
  type        = number
  default     = 90
  
  validation {
    condition     = var.metrics_retention_days >= 1 && var.metrics_retention_days <= 3653
    error_message = "Metrics retention days must be between 1 and 3653 days."
  }
}

# =============================================================================
# Performance Monitoring
# =============================================================================

variable "enable_synthetic_monitoring" {
  description = "Enable synthetic monitoring"
  type        = bool
  default     = true
}

variable "synthetic_check_frequency" {
  description = "Frequency for synthetic checks in seconds"
  type        = number
  default     = 300
  
  validation {
    condition     = var.synthetic_check_frequency >= 60
    error_message = "Synthetic check frequency must be at least 60 seconds."
  }
}

variable "apm_sample_rate" {
  description = "APM sampling rate (0-1)"
  type        = number
  default     = 0.1
  
  validation {
    condition     = var.apm_sample_rate >= 0 && var.apm_sample_rate <= 1
    error_message = "APM sample rate must be between 0 and 1."
  }
}

# =============================================================================
# Security Monitoring
# =============================================================================

variable "enable_security_monitoring" {
  description = "Enable security-related monitoring"
  type        = bool
  default     = true
}

variable "suspicious_activity_threshold" {
  description = "Threshold for suspicious activity alerts"
  type        = number
  default     = 100
}

variable "failed_login_threshold" {
  description = "Threshold for failed login attempts"
  type        = number
  default     = 10
}

# =============================================================================
# Custom Metrics Configuration
# =============================================================================

variable "custom_metrics" {
  description = "List of custom metrics to create"
  type = list(object({
    name        = string
    filter      = string
    description = string
    metric_kind = string
    value_type  = string
  }))
  default = []
}

# =============================================================================
# Integration Configuration
# =============================================================================

variable "enable_stackdriver_integration" {
  description = "Enable Stackdriver integration"
  type        = bool
  default     = true
}

variable "enable_grafana_integration" {
  description = "Enable Grafana integration"
  type        = bool
  default     = false
}

variable "grafana_workspace_id" {
  description = "Grafana workspace ID"
  type        = string
  default     = ""
}

variable "enable_datadog_integration" {
  description = "Enable Datadog integration"
  type        = bool
  default     = false
}

variable "datadog_api_key" {
  description = "Datadog API key"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# Alerting Rules
# =============================================================================

variable "custom_alerting_rules" {
  description = "List of custom alerting rules"
  type = list(object({
    name        = string
    condition   = string
    threshold   = number
    duration    = string
    severity    = string
    description = string
  }))
  default = []
}

# =============================================================================
# Common Labels
# =============================================================================

variable "common_labels" {
  description = "Common labels to apply to all monitoring resources"
  type        = map(string)
  default     = {}
}