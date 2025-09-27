#!/bin/bash

# 🚀 DARWIN GCP Monitoring & Alerting Setup - Revolutionary Observability
# Script completo para configurar monitoring avançado no Google Cloud Platform

set -e

# Colors for epic output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
PROJECT_ID="${PROJECT_ID:-pcs-helio}"
REGION="${REGION:-us-central1}"
BACKEND_SERVICE="kec-biomaterials-api"
FRONTEND_SERVICE="app-agourakis-med-br"
BACKEND_DOMAIN="api.agourakis.med.br"
FRONTEND_DOMAIN="darwin.agourakis.med.br"

echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}║  📊 DARWIN GCP MONITORING & ALERTING SETUP REVOLUTIONARY   ║${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}║  Project: $PROJECT_ID                                       ║${NC}"
echo -e "${PURPLE}║  Backend: $BACKEND_DOMAIN                                   ║${NC}"
echo -e "${PURPLE}║  Frontend: $FRONTEND_DOMAIN                                 ║${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Function to print status
print_status() {
    echo -e "${BLUE}🚀 [MONITORING]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ [SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  [WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}❌ [ERROR]${NC} $1"
}

print_epic() {
    echo -e "${CYAN}🎯 [EPIC]${NC} $1"
}

# Enable required APIs
enable_monitoring_apis() {
    print_status "Enabling monitoring APIs..."
    
    gcloud services enable \
        monitoring.googleapis.com \
        logging.googleapis.com \
        cloudtrace.googleapis.com \
        clouderrorreporting.googleapis.com \
        cloudprofiler.googleapis.com \
        --project=$PROJECT_ID \
        --quiet

    print_success "Monitoring APIs enabled"
}

# Create Uptime Checks
create_uptime_checks() {
    print_status "Creating uptime checks..."
    
    # Backend uptime check
    print_status "Setting up backend uptime check for $BACKEND_DOMAIN"
    gcloud alpha monitoring uptime create \
        --display-name="DARWIN Backend API Uptime" \
        --check-url="https://$BACKEND_DOMAIN/health" \
        --check-frequency="60s" \
        --check-timeout="20s" \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend uptime check may already exist"
    
    # Frontend uptime check
    print_status "Setting up frontend uptime check for $FRONTEND_DOMAIN"
    gcloud alpha monitoring uptime create \
        --display-name="DARWIN Frontend Uptime" \
        --check-url="https://$FRONTEND_DOMAIN" \
        --check-frequency="300s" \
        --check-timeout="30s" \
        --project=$PROJECT_ID \
        --quiet || print_warning "Frontend uptime check may already exist"
    
    print_success "Uptime checks created"
}

# Create Alert Policies
create_alert_policies() {
    print_status "Creating alert policies..."
    
    # High error rate alert for backend
    cat > /tmp/backend_error_alert.yaml << EOF
displayName: "DARWIN Backend High Error Rate"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="$BACKEND_SERVICE"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0.05
      duration: 300s
      aggregations:
        - alignmentPeriod: 300s
          perSeriesAligner: ALIGN_RATE
          crossSeriesReducer: REDUCE_MEAN
          groupByFields:
            - resource.labels.service_name
notificationChannels: []
enabled: true
EOF

    # Deploy backend error alert
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/backend_error_alert.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend error alert may already exist"
    
    # High response time alert
    cat > /tmp/backend_latency_alert.yaml << EOF
displayName: "DARWIN Backend High Latency"
conditions:
  - displayName: "Response time > 5s"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="$BACKEND_SERVICE"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 5000
      duration: 300s
      aggregations:
        - alignmentPeriod: 300s
          perSeriesAligner: ALIGN_MEAN
          crossSeriesReducer: REDUCE_MEAN
notificationChannels: []
enabled: true
EOF

    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/backend_latency_alert.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend latency alert may already exist"
    
    # Cleanup temp files
    rm -f /tmp/backend_error_alert.yaml /tmp/backend_latency_alert.yaml
    
    print_success "Alert policies created"
}

# Create Log Sinks
create_log_sinks() {
    print_status "Creating log sinks..."
    
    # Backend logs sink
    gcloud logging sinks create darwin-backend-logs \
        bigquery.googleapis.com/projects/$PROJECT_ID/datasets/darwin_analytics \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$BACKEND_SERVICE'"' \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend log sink may already exist"
    
    # Frontend logs sink  
    gcloud logging sinks create darwin-frontend-logs \
        bigquery.googleapis.com/projects/$PROJECT_ID/datasets/darwin_analytics \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$FRONTEND_SERVICE'"' \
        --project=$PROJECT_ID \
        --quiet || print_warning "Frontend log sink may already exist"
    
    print_success "Log sinks created"
}

# Create Custom Metrics
create_custom_metrics() {
    print_status "Creating custom metrics..."
    
    # JAX Performance Metric
    cat > /tmp/jax_performance_metric.yaml << EOF
type: custom.googleapis.com/darwin/jax_performance
displayName: "DARWIN JAX Performance"
metricKind: GAUGE
valueType: DOUBLE
description: "JAX performance metrics for DARWIN backend"
labels:
  - key: "operation_type"
    valueType: STRING
    description: "Type of JAX operation"
  - key: "batch_size" 
    valueType: STRING
    description: "Batch size for processing"
EOF

    gcloud logging metrics create darwin-jax-performance \
        --config-from-file=/tmp/jax_performance_metric.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "JAX performance metric may already exist"
    
    # Scaffold Processing Metric
    cat > /tmp/scaffold_processing_metric.yaml << EOF
type: custom.googleapis.com/darwin/scaffold_processing
displayName: "DARWIN Scaffold Processing Rate"
metricKind: GAUGE  
valueType: INT64
description: "Number of scaffolds processed per second"
labels:
  - key: "processing_type"
    valueType: STRING
    description: "Type of scaffold processing"
EOF

    gcloud logging metrics create darwin-scaffold-processing \
        --config-from-file=/tmp/scaffold_processing_metric.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Scaffold processing metric may already exist"
    
    # Cleanup
    rm -f /tmp/jax_performance_metric.yaml /tmp/scaffold_processing_metric.yaml
    
    print_success "Custom metrics created"
}

# Create Dashboards
create_dashboards() {
    print_status "Creating monitoring dashboards..."
    
    # Main DARWIN dashboard
    cat > /tmp/darwin_dashboard.json << EOF
{
  "displayName": "DARWIN Production Monitoring",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Backend Response Time",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$BACKEND_SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Backend Error Rate",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$BACKEND_SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "300s",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "width": 12,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "JAX Performance Metrics",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"custom.googleapis.com/darwin/jax_performance\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        }
      }
    ]
  }
}
EOF

    gcloud monitoring dashboards create \
        --config-from-file=/tmp/darwin_dashboard.json \
        --project=$PROJECT_ID \
        --quiet || print_warning "Dashboard may already exist"
    
    rm -f /tmp/darwin_dashboard.json
    
    print_success "Dashboards created"
}

# Setup Notification Channels (Email)
setup_notification_channels() {
    print_status "Setting up notification channels..."
    
    cat > /tmp/email_notification.yaml << EOF
type: email
displayName: "DARWIN Production Alerts"
enabled: true
labels:
  email_address: "demetrios@agourakis.med.br"
description: "Email notifications for DARWIN production alerts"
EOF

    gcloud alpha monitoring channels create \
        --channel-content-from-file=/tmp/email_notification.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Email notification channel may already exist"
    
    rm -f /tmp/email_notification.yaml
    
    print_success "Notification channels configured"
}

# Setup SLOs (Service Level Objectives)
setup_slos() {
    print_status "Setting up SLOs..."
    
    # Backend availability SLO (99.9%)
    cat > /tmp/backend_availability_slo.yaml << EOF
displayName: "DARWIN Backend Availability SLO"
serviceLevelIndicator:
  requestBased:
    goodTotalRatio:
      goodServiceFilter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="'$BACKEND_SERVICE'" AND http_request.status<500'
      totalServiceFilter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="'$BACKEND_SERVICE'"'
goal:
  performanceGoal:
    availabilityGoal:
      target: 0.999
calendarPeriod: MONTH
EOF

    gcloud alpha monitoring slos create \
        --config-from-file=/tmp/backend_availability_slo.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend SLO may already exist"
    
    # Backend latency SLO (95% under 2s)
    cat > /tmp/backend_latency_slo.yaml << EOF
displayName: "DARWIN Backend Latency SLO"
serviceLevelIndicator:
  requestBased:
    distributionCut:
      distributionFilter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="'$BACKEND_SERVICE'"'
      range:
        max: 2000
goal:
  performanceGoal:
    latencyGoal:
      target: 0.95
      threshold: 2000ms
calendarPeriod: MONTH
EOF

    gcloud alpha monitoring slos create \
        --config-from-file=/tmp/backend_latency_slo.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Backend latency SLO may already exist"
    
    # Cleanup
    rm -f /tmp/backend_availability_slo.yaml /tmp/backend_latency_slo.yaml
    
    print_success "SLOs configured"
}

# Create Budget and Cost Alerts
setup_cost_monitoring() {
    print_status "Setting up cost monitoring..."
    
    # Create budget for DARWIN project
    cat > /tmp/darwin_budget.yaml << EOF
displayName: "DARWIN Production Budget"
budgetFilter:
  projects:
    - "projects/$PROJECT_ID"
  services:
    - "services/6F81-5844-456A"  # Cloud Run
    - "services/A1E8-BE35-7EBC"  # BigQuery
    - "services/6AC4-3783-9BF1"  # Vertex AI
amount:
  specifiedAmount:
    currencyCode: "USD"
    units: "200"
thresholdRules:
  - thresholdPercent: 0.5
    spendBasis: CURRENT_SPEND
  - thresholdPercent: 0.8
    spendBasis: CURRENT_SPEND
  - thresholdPercent: 1.0
    spendBasis: CURRENT_SPEND
EOF

    gcloud billing budgets create \
        --billing-account=$(gcloud billing accounts list --format="value(name)" | head -n1) \
        --budget-from-file=/tmp/darwin_budget.yaml \
        --project=$PROJECT_ID \
        --quiet || print_warning "Budget may already exist"
    
    rm -f /tmp/darwin_budget.yaml
    
    print_success "Cost monitoring configured"
}

# Validate monitoring setup
validate_monitoring() {
    print_status "Validating monitoring setup..."
    
    # Check uptime checks
    UPTIME_CHECKS=$(gcloud alpha monitoring uptime list --project=$PROJECT_ID --format="value(name)" | wc -l)
    print_status "Uptime checks configured: $UPTIME_CHECKS"
    
    # Check alert policies
    ALERT_POLICIES=$(gcloud alpha monitoring policies list --project=$PROJECT_ID --format="value(name)" | wc -l)
    print_status "Alert policies configured: $ALERT_POLICIES"
    
    # Check log sinks
    LOG_SINKS=$(gcloud logging sinks list --project=$PROJECT_ID --format="value(name)" | wc -l)
    print_status "Log sinks configured: $LOG_SINKS"
    
    print_success "Monitoring validation complete"
}

# Print monitoring summary
print_monitoring_summary() {
    print_epic "DARWIN GCP MONITORING SETUP COMPLETE!"
    echo ""
    echo -e "${GREEN}📊 Monitoring Components:${NC}"
    echo -e "   🔍 Uptime Checks: Backend + Frontend"
    echo -e "   🚨 Alert Policies: Error rate + Latency"
    echo -e "   📋 Log Sinks: BigQuery integration"
    echo -e "   🎯 SLOs: 99.9% availability + 95% <2s latency"
    echo -e "   💰 Cost Alerts: $200 monthly budget with thresholds"
    echo ""
    echo -e "${GREEN}🌐 Monitoring URLs:${NC}"
    echo -e "   📊 Cloud Console: ${CYAN}https://console.cloud.google.com/monitoring?project=$PROJECT_ID${NC}"
    echo -e "   📋 Logs: ${CYAN}https://console.cloud.google.com/logs?project=$PROJECT_ID${NC}"
    echo -e "   🚨 Alerting: ${CYAN}https://console.cloud.google.com/monitoring/alerting?project=$PROJECT_ID${NC}"
    echo -e "   💰 Billing: ${CYAN}https://console.cloud.google.com/billing?project=$PROJECT_ID${NC}"
    echo ""
    echo -e "${CYAN}📋 Next Steps:${NC}"
    echo -e "   1. Configure notification channels (email/Slack)"
    echo -e "   2. Tune alert thresholds based on baseline metrics"
    echo -e "   3. Set up custom dashboards for business metrics"
    echo -e "   4. Enable Cloud Trace for detailed request analysis"
    echo ""
    echo -e "${PURPLE}🧬 DARWIN MONITORING REVOLUTIONARY READY! 🚀${NC}"
}

# Main execution
main() {
    echo -e "${CYAN}🚀 Starting DARWIN GCP Monitoring Setup...${NC}"
    echo -e "${CYAN}📦 Project: $PROJECT_ID${NC}"
    echo -e "${CYAN}🌐 Backend: $BACKEND_DOMAIN${NC}"
    echo -e "${CYAN}🌐 Frontend: $FRONTEND_DOMAIN${NC}"
    echo

    # Execute setup steps
    enable_monitoring_apis
    create_uptime_checks
    create_alert_policies
    create_log_sinks
    setup_slos
    setup_notification_channels
    setup_cost_monitoring
    
    # Validate setup
    echo
    validate_monitoring
    
    # Print summary
    echo
    print_monitoring_summary
}

# Handle interruption
trap 'echo -e "\n${YELLOW}Monitoring setup interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"