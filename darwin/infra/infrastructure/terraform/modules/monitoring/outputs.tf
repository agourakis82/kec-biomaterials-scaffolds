# =============================================================================
# DARWIN Monitoring Module - Outputs
# =============================================================================

# =============================================================================
# Notification Channels Outputs
# =============================================================================

output "email_notification_channels" {
  description = "Map of email notification channels"
  value = {
    for email, channel in google_monitoring_notification_channel.email :
    email => {
      name         = channel.name
      id          = channel.id
      display_name = channel.display_name
      type        = channel.type
    }
  }
}

output "slack_notification_channel" {
  description = "Slack notification channel details"
  value = var.slack_webhook_url != "" ? {
    name         = google_monitoring_notification_channel.slack[0].name
    id          = google_monitoring_notification_channel.slack[0].id
    display_name = google_monitoring_notification_channel.slack[0].display_name
    type        = google_monitoring_notification_channel.slack[0].type
  } : null
}

output "pagerduty_notification_channel" {
  description = "PagerDuty notification channel details"
  value = var.pagerduty_service_key != "" ? {
    name         = google_monitoring_notification_channel.pagerduty[0].name
    id          = google_monitoring_notification_channel.pagerduty[0].id
    display_name = google_monitoring_notification_channel.pagerduty[0].display_name
    type        = google_monitoring_notification_channel.pagerduty[0].type
  } : null
}

output "all_notification_channels" {
  description = "List of all notification channel names"
  value = concat(
    [for channel in google_monitoring_notification_channel.email : channel.name],
    var.slack_webhook_url != "" ? [google_monitoring_notification_channel.slack[0].name] : [],
    var.pagerduty_service_key != "" ? [google_monitoring_notification_channel.pagerduty[0].name] : []
  )
}

# =============================================================================
# Uptime Checks Outputs
# =============================================================================

output "api_uptime_check" {
  description = "API uptime check configuration"
  value = {
    name         = google_monitoring_uptime_check_config.api_uptime.name
    id          = google_monitoring_uptime_check_config.api_uptime.uptime_check_id
    display_name = google_monitoring_uptime_check_config.api_uptime.display_name
    timeout     = google_monitoring_uptime_check_config.api_uptime.timeout
    period      = google_monitoring_uptime_check_config.api_uptime.period
    regions     = google_monitoring_uptime_check_config.api_uptime.selected_regions
  }
}

output "frontend_uptime_check" {
  description = "Frontend uptime check configuration"
  value = {
    name         = google_monitoring_uptime_check_config.frontend_uptime.name
    id          = google_monitoring_uptime_check_config.frontend_uptime.uptime_check_id
    display_name = google_monitoring_uptime_check_config.frontend_uptime.display_name
    timeout     = google_monitoring_uptime_check_config.frontend_uptime.timeout
    period      = google_monitoring_uptime_check_config.frontend_uptime.period
    regions     = google_monitoring_uptime_check_config.frontend_uptime.selected_regions
  }
}

output "uptime_checks" {
  description = "Map of all uptime checks"
  value = {
    api = {
      name         = google_monitoring_uptime_check_config.api_uptime.name
      id          = google_monitoring_uptime_check_config.api_uptime.uptime_check_id
      display_name = google_monitoring_uptime_check_config.api_uptime.display_name
    }
    frontend = {
      name         = google_monitoring_uptime_check_config.frontend_uptime.name
      id          = google_monitoring_uptime_check_config.frontend_uptime.uptime_check_id
      display_name = google_monitoring_uptime_check_config.frontend_uptime.display_name
    }
  }
}

# =============================================================================
# Alert Policies Outputs
# =============================================================================

output "api_down_alert" {
  description = "API service down alert policy"
  value = {
    name         = google_monitoring_alert_policy.api_down.name
    display_name = google_monitoring_alert_policy.api_down.display_name
    enabled     = google_monitoring_alert_policy.api_down.enabled
  }
}

output "frontend_down_alert" {
  description = "Frontend service down alert policy"
  value = {
    name         = google_monitoring_alert_policy.frontend_down.name
    display_name = google_monitoring_alert_policy.frontend_down.display_name
    enabled     = google_monitoring_alert_policy.frontend_down.enabled
  }
}

output "high_error_rate_alert" {
  description = "High error rate alert policy"
  value = {
    name         = google_monitoring_alert_policy.high_error_rate.name
    display_name = google_monitoring_alert_policy.high_error_rate.display_name
    enabled     = google_monitoring_alert_policy.high_error_rate.enabled
  }
}

output "high_latency_alert" {
  description = "High latency alert policy"
  value = {
    name         = google_monitoring_alert_policy.high_latency.name
    display_name = google_monitoring_alert_policy.high_latency.display_name
    enabled     = google_monitoring_alert_policy.high_latency.enabled
  }
}

output "database_connections_alert" {
  description = "Database connections alert policy"
  value = {
    name         = google_monitoring_alert_policy.database_connections.name
    display_name = google_monitoring_alert_policy.database_connections.display_name
    enabled     = google_monitoring_alert_policy.database_connections.enabled
  }
}

output "redis_memory_alert" {
  description = "Redis memory usage alert policy"
  value = {
    name         = google_monitoring_alert_policy.redis_memory.name
    display_name = google_monitoring_alert_policy.redis_memory.display_name
    enabled     = google_monitoring_alert_policy.redis_memory.enabled
  }
}

output "alerting_policies" {
  description = "Map of all alerting policies"
  value = {
    api_down              = google_monitoring_alert_policy.api_down.name
    frontend_down         = google_monitoring_alert_policy.frontend_down.name
    high_error_rate      = google_monitoring_alert_policy.high_error_rate.name
    high_latency         = google_monitoring_alert_policy.high_latency.name
    database_connections = google_monitoring_alert_policy.database_connections.name
    redis_memory         = google_monitoring_alert_policy.redis_memory.name
  }
}

# =============================================================================
# Dashboard Outputs
# =============================================================================

output "dashboard_url" {
  description = "URL to access the main monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.main.id}?project=${var.project_id}"
}

output "dashboard_id" {
  description = "ID of the main monitoring dashboard"
  value       = google_monitoring_dashboard.main.id
}

output "dashboard_name" {
  description = "Name of the main monitoring dashboard"
  value       = "${var.project_name} ${var.environment} - DARWIN Platform Dashboard"
}

# =============================================================================
# Budget Outputs
# =============================================================================

output "budget_name" {
  description = "Name of the budget"
  value       = google_billing_budget.budget.display_name
}

output "budget_amount" {
  description = "Budget amount in USD"
  value       = var.budget_amount
}

output "budget_thresholds" {
  description = "Budget alert thresholds"
  value       = var.budget_alert_thresholds
}

# =============================================================================
# Log-based Metrics Outputs
# =============================================================================

output "error_rate_metric" {
  description = "Error rate log-based metric"
  value = {
    name   = google_logging_metric.error_rate.name
    filter = google_logging_metric.error_rate.filter
    id     = google_logging_metric.error_rate.id
  }
}

output "slow_queries_metric" {
  description = "Slow queries log-based metric"
  value = {
    name   = google_logging_metric.slow_queries.name
    filter = google_logging_metric.slow_queries.filter
    id     = google_logging_metric.slow_queries.id
  }
}

output "log_metrics" {
  description = "Map of all log-based metrics"
  value = {
    error_rate    = google_logging_metric.error_rate.name
    slow_queries = google_logging_metric.slow_queries.name
  }
}

# =============================================================================
# SLO Outputs
# =============================================================================

output "api_availability_slo" {
  description = "API availability SLO configuration"
  value = {
    name    = google_monitoring_slo.api_availability.name
    slo_id  = google_monitoring_slo.api_availability.slo_id
    goal    = google_monitoring_slo.api_availability.goal
    service = google_monitoring_slo.api_availability.service
  }
}

output "api_service" {
  description = "API monitoring service configuration"
  value = {
    name       = google_monitoring_service.api_service.name
    service_id = google_monitoring_service.api_service.service_id
    id        = google_monitoring_service.api_service.id
  }
}

# =============================================================================
# Monitoring Configuration Summary
# =============================================================================

output "monitoring_config" {
  description = "Complete monitoring configuration summary"
  value = {
    project_id   = var.project_id
    environment  = var.environment
    region      = var.region
    
    # Services being monitored
    monitored_services = {
      backend_service  = var.backend_service_name
      frontend_service = var.frontend_service_name
      database        = var.database_instance_id
      redis          = var.redis_instance_id
    }
    
    # Domains being monitored
    monitored_domains = {
      api      = var.api_domain
      frontend = var.frontend_domain
    }
    
    # Alert thresholds
    thresholds = {
      error_rate_percent    = var.error_rate_threshold * 100
      latency_ms           = var.latency_threshold_ms
      cpu_percent         = var.cpu_threshold_percent
      memory_percent      = var.memory_threshold_percent
      db_connections      = var.database_connections_threshold
      redis_memory_percent = var.redis_memory_threshold_percent
    }
    
    # SLO targets
    slo_targets = {
      api_availability = var.api_availability_slo
      api_latency_ms   = var.api_latency_slo
    }
    
    # Budget configuration
    budget = {
      amount      = var.budget_amount
      currency    = "USD"
      thresholds  = var.budget_alert_thresholds
    }
  }
}

# =============================================================================
# Notification Configuration
# =============================================================================

output "notification_config" {
  description = "Notification configuration summary"
  value = {
    email_addresses_count = length(var.email_addresses)
    slack_enabled        = var.slack_webhook_url != ""
    pagerduty_enabled    = var.pagerduty_service_key != ""
    sms_enabled         = var.sms_number != ""
    
    channels = {
      email     = length(google_monitoring_notification_channel.email)
      slack     = var.slack_webhook_url != "" ? 1 : 0
      pagerduty = var.pagerduty_service_key != "" ? 1 : 0
    }
  }
}

# =============================================================================
# Uptime Monitoring Configuration
# =============================================================================

output "uptime_monitoring_config" {
  description = "Uptime monitoring configuration summary"
  value = {
    api_uptime_check = {
      enabled     = true
      path        = var.api_health_path
      timeout     = var.uptime_check_timeout
      period      = var.uptime_check_period
      regions     = var.uptime_check_regions
    }
    
    frontend_uptime_check = {
      enabled     = true
      path        = var.frontend_health_path
      timeout     = var.uptime_check_timeout
      period      = var.uptime_check_period
      regions     = var.uptime_check_regions
    }
    
    total_checks = 2
  }
}

# =============================================================================
# Alert Configuration
# =============================================================================

output "alert_config" {
  description = "Alert configuration summary"
  value = {
    total_policies = 6
    
    policies = {
      uptime_alerts = {
        api_down      = google_monitoring_alert_policy.api_down.enabled
        frontend_down = google_monitoring_alert_policy.frontend_down.enabled
      }
      
      performance_alerts = {
        high_error_rate = google_monitoring_alert_policy.high_error_rate.enabled
        high_latency    = google_monitoring_alert_policy.high_latency.enabled
      }
      
      resource_alerts = {
        database_connections = google_monitoring_alert_policy.database_connections.enabled
        redis_memory        = google_monitoring_alert_policy.redis_memory.enabled
      }
    }
    
    auto_close_duration = var.alert_auto_close_duration
  }
}

# =============================================================================
# Monitoring URLs and Links
# =============================================================================

output "monitoring_urls" {
  description = "Important monitoring URLs"
  value = {
    dashboard = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.main.id}?project=${var.project_id}"
    metrics   = "https://console.cloud.google.com/monitoring/metrics-explorer?project=${var.project_id}"
    logs      = "https://console.cloud.google.com/logs/query?project=${var.project_id}"
    alerts    = "https://console.cloud.google.com/monitoring/alerting?project=${var.project_id}"
    uptime    = "https://console.cloud.google.com/monitoring/uptime?project=${var.project_id}"
    slo       = "https://console.cloud.google.com/monitoring/slo?project=${var.project_id}"
  }
}

# =============================================================================
# Feature Flags
# =============================================================================

output "monitoring_features" {
  description = "Enabled monitoring features"
  value = {
    custom_dashboard      = var.enable_custom_dashboard
    budget_alerts        = var.enable_budget_alerts
    log_metrics          = var.enable_log_metrics
    uptime_alerts        = var.enable_uptime_alerts
    performance_alerts   = var.enable_performance_alerts
    resource_alerts      = var.enable_resource_alerts
    synthetic_monitoring = var.enable_synthetic_monitoring
    security_monitoring  = var.enable_security_monitoring
    distributed_tracing  = var.enable_distributed_tracing
    profiling           = var.enable_profiling
    debugging           = var.enable_debugging
  }
}

# =============================================================================
# Health Check Endpoints
# =============================================================================

output "health_check_endpoints" {
  description = "Health check endpoints being monitored"
  value = {
    api = {
      url    = "https://${var.api_domain}${var.api_health_path}"
      domain = var.api_domain
      path   = var.api_health_path
    }
    frontend = {
      url    = "https://${var.frontend_domain}${var.frontend_health_path}"
      domain = var.frontend_domain
      path   = var.frontend_health_path
    }
  }
}

# =============================================================================
# Retention Configuration
# =============================================================================

output "retention_config" {
  description = "Data retention configuration"
  value = {
    logs_retention_days    = var.log_retention_days
    metrics_retention_days = var.metrics_retention_days
    slo_rolling_period_days = var.slo_rolling_period_days
  }
}

# =============================================================================
# Integration Status
# =============================================================================

output "integration_status" {
  description = "Status of third-party integrations"
  value = {
    stackdriver = var.enable_stackdriver_integration
    grafana = {
      enabled      = var.enable_grafana_integration
      workspace_id = var.grafana_workspace_id != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
    }
    datadog = {
      enabled = var.enable_datadog_integration
      api_key = var.datadog_api_key != "" ? "[CONFIGURED]" : "[NOT_CONFIGURED]"
    }
  }
}