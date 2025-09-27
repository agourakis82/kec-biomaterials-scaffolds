# =============================================================================
# DARWIN Monitoring Module - Main Configuration
# Cloud Monitoring, Alerting, Dashboards, Uptime Checks, Budget Alerts
# =============================================================================

# =============================================================================
# Local Values
# =============================================================================

locals {
  notification_channels = var.notification_channels
  
  # Alert thresholds
  error_rate_threshold = 0.05  # 5% error rate
  latency_threshold    = 5000  # 5 seconds
  cpu_threshold        = 80    # 80% CPU
  memory_threshold     = 85    # 85% memory
  disk_threshold       = 90    # 90% disk
  
  # Dashboard configuration
  dashboard_config = {
    displayName = "${var.project_name} ${var.environment} - DARWIN Platform Dashboard"
    mosaicLayout = {
      columns = 12
    }
  }
}

# =============================================================================
# Notification Channels
# =============================================================================

resource "google_monitoring_notification_channel" "email" {
  for_each = toset(var.email_addresses)
  
  display_name = "Email - ${each.value}"
  type         = "email"
  project      = var.project_id
  
  labels = {
    email_address = each.value
  }
  
  description = "Email notifications for ${var.project_name} ${var.environment}"
}

resource "google_monitoring_notification_channel" "slack" {
  count = var.slack_webhook_url != "" ? 1 : 0

  display_name = "Slack - ${var.project_name}"
  type         = "slack"
  project      = var.project_id

  labels = {
    channel_name = var.slack_channel_name
  }

  sensitive_labels {
    auth_token = var.slack_webhook_url
  }

  description = "Slack notifications for ${var.project_name} ${var.environment}"
}

resource "google_monitoring_notification_channel" "pagerduty" {
  count = var.pagerduty_service_key != "" ? 1 : 0
  
  display_name = "PagerDuty - ${var.project_name}"
  type         = "pagerduty"
  project      = var.project_id
  
  sensitive_labels {
    service_key = var.pagerduty_service_key
  }
  
  description = "PagerDuty notifications for ${var.project_name} ${var.environment}"
}

# =============================================================================
# Uptime Checks
# =============================================================================

resource "google_monitoring_uptime_check_config" "api_uptime" {
  display_name = "${var.project_name} API Uptime Check"
  project      = var.project_id
  timeout      = "10s"
  period       = "300s"
  
  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
    
    accepted_response_status_codes {
      status_class = "STATUS_CLASS_2XX"
    }
    
    headers = {
      "User-Agent" = "Google-Cloud-Uptime-Check"
    }
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = var.api_domain
    }
  }
  
  checker_type = "STATIC_IP_CHECKERS"
  
  selected_regions = [
    "USA_OREGON",
    "USA_VIRGINIA", 
    "EUROPE_IRELAND",
    "ASIA_SINGAPORE"
  ]
}

resource "google_monitoring_uptime_check_config" "frontend_uptime" {
  display_name = "${var.project_name} Frontend Uptime Check"
  project      = var.project_id
  timeout      = "10s"
  period       = "300s"
  
  http_check {
    path         = "/"
    port         = 443
    use_ssl      = true
    validate_ssl = true
    
    accepted_response_status_codes {
      status_class = "STATUS_CLASS_2XX"
    }
    
    headers = {
      "User-Agent" = "Google-Cloud-Uptime-Check"
    }
  }
  
  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = var.frontend_domain
    }
  }
  
  checker_type = "STATIC_IP_CHECKERS"
  
  selected_regions = [
    "USA_OREGON",
    "USA_VIRGINIA",
    "EUROPE_IRELAND", 
    "ASIA_SINGAPORE"
  ]
}

# =============================================================================
# Alerting Policies
# =============================================================================

# API Service Down Alert
resource "google_monitoring_alert_policy" "api_down" {
  display_name = "${var.project_name} - API Service Down"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "API Uptime Check Failed"
    
    condition_threshold {
      filter         = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\""
      duration       = "300s"
      comparison     = "COMPARISON_LT"
      threshold_value = 1
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.project_id", "resource.label.host"]
      }
    }
  }
  
  notification_channels = concat(
    [for channel in google_monitoring_notification_channel.email : channel.name],
    var.slack_webhook_url != "" ? [google_monitoring_notification_channel.slack[0].name] : [],
    var.pagerduty_service_key != "" ? [google_monitoring_notification_channel.pagerduty[0].name] : []
  )
  
  alert_strategy {
    auto_close = "1800s"  # 30 minutes
  }
  
  documentation {
    content = "API service is down. Check Cloud Run service health and logs."
    mime_type = "text/markdown"
  }
}

# Frontend Service Down Alert
resource "google_monitoring_alert_policy" "frontend_down" {
  display_name = "${var.project_name} - Frontend Service Down"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Frontend Uptime Check Failed"
    
    condition_threshold {
      filter         = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\""
      duration       = "300s"
      comparison     = "COMPARISON_LT"
      threshold_value = 1
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.project_id", "resource.label.host"]
      }
    }
  }
  
  notification_channels = concat(
    [for channel in google_monitoring_notification_channel.email : channel.name],
    var.slack_webhook_url != "" ? [google_monitoring_notification_channel.slack[0].name] : []
  )
  
  alert_strategy {
    auto_close = "1800s"
  }
  
  documentation {
    content = "Frontend service is down. Check Cloud Run service health and logs."
    mime_type = "text/markdown"
  }
}

# High Error Rate Alert
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "${var.project_name} - High Error Rate"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Error Rate > 5%"
    
    condition_threshold {
      filter         = "metric.type=\"run.googleapis.com/request_count\" AND resource.type=\"cloud_run_revision\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = local.error_rate_threshold
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields    = ["resource.label.service_name", "metric.label.response_code_class"]
      }
    }
  }
  
  notification_channels = concat(
    [for channel in google_monitoring_notification_channel.email : channel.name],
    var.slack_webhook_url != "" ? [google_monitoring_notification_channel.slack[0].name] : []
  )
  
  documentation {
    content = "High error rate detected. Check application logs for errors."
    mime_type = "text/markdown"
  }
}

# High Latency Alert
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "${var.project_name} - High Latency"
  project      = var.project_id
  combiner     = "OR" 
  enabled      = true
  
  conditions {
    display_name = "Response Latency > 5s"
    
    condition_threshold {
      filter         = "metric.type=\"run.googleapis.com/request_latencies\" AND resource.type=\"cloud_run_revision\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = local.latency_threshold
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }
  
  notification_channels = [for channel in google_monitoring_notification_channel.email : channel.name]
  
  documentation {
    content = "High latency detected. Check application performance and database queries."
    mime_type = "text/markdown"
  }
}

# Database High Connections Alert
resource "google_monitoring_alert_policy" "database_connections" {
  display_name = "${var.project_name} - Database High Connections"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Database Connections > 80%"
    
    condition_threshold {
      filter         = "metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends\" AND resource.type=\"cloudsql_database\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = 80
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.database_id"]
      }
    }
  }
  
  notification_channels = [for channel in google_monitoring_notification_channel.email : channel.name]
  
  documentation {
    content = "Database connection pool is nearly exhausted. Check application connection usage."
    mime_type = "text/markdown"
  }
}

# Redis Memory Usage Alert
resource "google_monitoring_alert_policy" "redis_memory" {
  display_name = "${var.project_name} - Redis High Memory Usage"
  project      = var.project_id
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Redis Memory Usage > 85%"
    
    condition_threshold {
      filter         = "metric.type=\"redis.googleapis.com/stats/memory/usage_ratio\" AND resource.type=\"redis_instance\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = 0.85
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields    = ["resource.label.instance_id"]
      }
    }
  }
  
  notification_channels = [for channel in google_monitoring_notification_channel.email : channel.name]
  
  documentation {
    content = "Redis memory usage is high. Consider increasing memory or reviewing cache policies."
    mime_type = "text/markdown"
  }
}

# =============================================================================
# Custom Dashboard
# =============================================================================

resource "google_monitoring_dashboard" "main" {
  dashboard_json = jsonencode({
    displayName = local.dashboard_config.displayName
    
    mosaicLayout = {
      columns = local.dashboard_config.mosaicLayout.columns
      
      tiles = [
        # API Request Rate
        {
          width = 6
          height = 4
          widget = {
            title = "API Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/request_count\" AND resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.backend_service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields = ["metric.label.response_code_class"]
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Requests/sec"
                scale = "LINEAR"
              }
            }
          }
        },
        
        # API Response Latency
        {
          width = 6
          height = 4
          widget = {
            title = "API Response Latency (95th percentile)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/request_latencies\" AND resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.backend_service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Latency (ms)"
                scale = "LINEAR"
              }
            }
          }
        },
        
        # Cloud Run Instances
        {
          width = 6
          height = 4
          widget = {
            title = "Cloud Run Instance Count"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"run.googleapis.com/container/instance_count\" AND resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.backend_service_name}\""
                      aggregation = {
                        alignmentPeriod = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_SUM"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "Backend"
                },
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"run.googleapis.com/container/instance_count\" AND resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${var.frontend_service_name}\""
                      aggregation = {
                        alignmentPeriod = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_SUM"
                      }
                    }
                  }
                  plotType = "LINE"
                  legendTemplate = "Frontend"
                }
              ]
              yAxis = {
                label = "Instances"
                scale = "LINEAR"
              }
            }
          }
        },
        
        # Database Connections
        {
          width = 6
          height = 4
          widget = {
            title = "Database Connections"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends\" AND resource.type=\"cloudsql_database\" AND resource.label.database_id=\"${var.database_instance_id}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Active Connections"
                scale = "LINEAR"
              }
            }
          }
        },
        
        # Redis Memory Usage
        {
          width = 6
          height = 4
          widget = {
            title = "Redis Memory Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"redis.googleapis.com/stats/memory/usage_ratio\" AND resource.type=\"redis_instance\" AND resource.label.instance_id=\"${var.redis_instance_id}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Memory Usage Ratio"
                scale = "LINEAR"
              }
            }
          }
        },
        
        # Error Log Entries
        {
          width = 6
          height = 4
          widget = {
            title = "Error Log Entries"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/log_entry_count\" AND resource.type=\"cloud_run_revision\" AND metric.label.severity=\"ERROR\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
              }]
              yAxis = {
                label = "Errors/sec"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
  
  project = var.project_id
}

# =============================================================================
# Budget Alert
# =============================================================================

data "google_billing_account" "account" {
  billing_account = var.billing_account_id
}

resource "google_billing_budget" "budget" {
  billing_account = data.google_billing_account.account.id
  display_name    = "${var.project_name} ${var.environment} Budget"
  
  budget_filter {
    projects = ["projects/${var.project_id}"]
    
    services = [
      "services/6F81-5844-456A",  # Compute Engine
      "services/95FF-2EF5-5EA1",  # Cloud SQL
      "services/24E6-581D-38E5",  # Cloud Storage
      "services/C654-D5A5-5EC1",  # Cloud Run
      "services/58CD-8F1E-C991",  # Cloud Load Balancing
      "services/F25E-97C4-4556",  # Cloud CDN
      "services/95FF-2EF5-5EA1"   # Cloud Monitoring
    ]
  }
  
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }
  
  threshold_rules {
    threshold_percent = 0.5  # 50%
    spend_basis      = "CURRENT_SPEND"
  }
  
  threshold_rules {
    threshold_percent = 0.8  # 80%
    spend_basis      = "CURRENT_SPEND"
  }
  
  threshold_rules {
    threshold_percent = 1.0  # 100%
    spend_basis      = "CURRENT_SPEND"
  }
  
  threshold_rules {
    threshold_percent = 1.2  # 120%
    spend_basis      = "FORECASTED_SPEND"
  }
  
  all_updates_rule {
    monitoring_notification_channels = [for channel in google_monitoring_notification_channel.email : channel.name]
    disable_default_iam_recipients   = false
  }
}

# =============================================================================
# Log-based Metrics
# =============================================================================

resource "google_logging_metric" "error_rate" {
  name   = "${var.project_name}_${var.environment}_error_rate"
  filter = "severity >= ERROR"
  
  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "Error Rate"
  }
  
  label_extractors = {
    "service_name" = "EXTRACT(resource.labels.service_name)"
    "severity"     = "EXTRACT(severity)"
  }
  
  project = var.project_id
}

resource "google_logging_metric" "slow_queries" {
  name   = "${var.project_name}_${var.environment}_slow_queries"
  filter = "resource.type=\"cloudsql_database\" AND textPayload=~\"duration: [5-9][0-9]{3}ms|duration: [1-9][0-9]{4}ms\""
  
  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "Slow Database Queries"
  }
  
  project = var.project_id
}

# =============================================================================
# SLI/SLO Configuration
# =============================================================================

resource "google_monitoring_slo" "api_availability" {
  service      = google_monitoring_service.api_service.service_id
  display_name = "API Availability SLO"
  slo_id       = "api-availability"
  
  request_based_sli {
    good_total_ratio {
      total_service_filter = "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\" resource.label.service_name=\"${var.backend_service_name}\""
      good_service_filter  = "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\" resource.label.service_name=\"${var.backend_service_name}\" metric.label.response_code_class=\"2xx\""
    }
  }
  
  goal = 0.995  # 99.5% availability
  
  rolling_period_days = 30
}

resource "google_monitoring_service" "api_service" {
  service_id   = "${var.project_name}-${var.environment}-api"
  display_name = "${var.project_name} API Service"
  
  basic_service {
    service_type = "CLOUD_RUN"
    service_labels = {
      service_name = var.backend_service_name
      location     = var.region
    }
  }
  
  project = var.project_id
}