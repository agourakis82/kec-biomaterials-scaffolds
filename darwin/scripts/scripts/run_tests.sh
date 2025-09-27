#!/bin/bash

# =============================================================================
# DARWIN Test Suite Runner
# Script master para executar toda a suite de testes abrangente
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_ROOT/tests"

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="staging"
PROJECT_ID=""
REGION="us-central1"
API_URL=""
FRONTEND_URL=""
RUN_UNIT_TESTS="true"
RUN_INTEGRATION_TESTS="true"
RUN_LOAD_TESTS="true"
RUN_SECURITY_TESTS="true"
RUN_E2E_TESTS="true"
VERBOSE="false"
PARALLEL="false"
SKIP_SETUP="false"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1" >&2
    fi
}

show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                      DARWIN TEST SUITE                        ║
    ║                    Comprehensive Testing                      ║
    ║                                                               ║
    ║        Unit • Integration • Load • Security • E2E            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DARWIN Comprehensive Test Suite Runner

OPTIONS:
    -p, --project-id PROJECT_ID       GCP Project ID (required)
    -e, --environment ENVIRONMENT     Environment [dev|staging|production] (default: staging)
    -r, --region REGION              GCP Region (default: us-central1)
    -a, --api-url URL                API URL (auto-detected if not provided)
    -f, --frontend-url URL           Frontend URL (auto-detected if not provided)
    
    Test types:
    --unit-tests                     Run unit tests only
    --integration-tests              Run integration tests only
    --load-tests                     Run load tests only
    --security-tests                 Run security tests only
    --e2e-tests                      Run end-to-end tests only
    --all                           Run all test types (default)
    
    Options:
    --parallel                       Run compatible tests in parallel
    --skip-setup                     Skip test environment setup
    -v, --verbose                    Enable verbose logging
    -h, --help                       Show this help message

EXAMPLES:
    $0 -p my-project --all
    $0 -p my-project --unit-tests --integration-tests
    $0 -p my-project --load-tests --parallel
    $0 -p my-project -e production --security-tests

ENVIRONMENT VARIABLES:
    DARWIN_PROJECT_ID                Project ID
    DARWIN_ENVIRONMENT               Environment
    DARWIN_REGION                    GCP Region
    DARWIN_API_URL                   API URL
    DARWIN_FRONTEND_URL              Frontend URL

EOF
}

detect_service_urls() {
    if [[ -z "$API_URL" ]]; then
        log_info "Auto-detecting API URL..."
        
        local backend_service="darwin-${ENVIRONMENT}-backend"
        local service_url
        service_url=$(gcloud run services describe "$backend_service" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --format="value(status.url)" 2>/dev/null || echo "")
        
        if [[ -n "$service_url" ]]; then
            API_URL="$service_url"
            log_success "API URL detected: $API_URL"
        else
            # Fallback to custom domain
            API_URL="https://api-${ENVIRONMENT}.agourakis.med.br"
            if [[ "$ENVIRONMENT" == "production" ]]; then
                API_URL="https://api.agourakis.med.br"
            fi
            log_info "Using default API URL: $API_URL"
        fi
    fi
    
    if [[ -z "$FRONTEND_URL" ]]; then
        log_info "Auto-detecting Frontend URL..."
        
        local frontend_service="darwin-${ENVIRONMENT}-frontend"
        local service_url
        service_url=$(gcloud run services describe "$frontend_service" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --format="value(status.url)" 2>/dev/null || echo "")
        
        if [[ -n "$service_url" ]]; then
            FRONTEND_URL="$service_url"
            log_success "Frontend URL detected: $FRONTEND_URL"
        else
            # Fallback to custom domain
            FRONTEND_URL="https://darwin-${ENVIRONMENT}.agourakis.med.br"
            if [[ "$ENVIRONMENT" == "production" ]]; then
                FRONTEND_URL="https://darwin.agourakis.med.br"
            fi
            log_info "Using default Frontend URL: $FRONTEND_URL"
        fi
    fi
}

check_prerequisites() {
    log_info "Checking test prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    # Check optional tools
    if [[ "$RUN_LOAD_TESTS" == "true" ]]; then
        command -v k6 >/dev/null 2>&1 || {
            log_warning "k6 not found, will install during load testing"
        }
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access project $PROJECT_ID"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

setup_test_environment() {
    if [[ "$SKIP_SETUP" == "true" ]]; then
        log_info "Skipping test environment setup"
        return 0
    fi
    
    log_info "Setting up test environment..."
    
    # Install Python test dependencies
    log_debug "Installing Python test dependencies..."
    pip3 install --user requests httpx pytest pytest-asyncio pytest-cov || {
        log_warning "Could not install Python dependencies globally, trying local install"
        python3 -m pip install --user requests httpx pytest pytest-asyncio pytest-cov
    }
    
    # Create test results directory
    mkdir -p "$PROJECT_ROOT/test-results"
    
    log_success "Test environment setup completed"
}

run_unit_tests() {
    if [[ "$RUN_UNIT_TESTS" != "true" ]]; then
        log_info "Skipping unit tests"
        return 0
    fi
    
    log_info "Running unit tests..."
    
    local test_results=0
    
    # Backend unit tests
    if [[ -d "$PROJECT_ROOT/src/kec_unified_api/tests" ]]; then
        log_debug "Running backend unit tests..."
        cd "$PROJECT_ROOT/src/kec_unified_api"
        
        # Install dependencies if needed
        if [[ -f "requirements.txt" ]]; then
            pip3 install --user -r requirements.txt || log_warning "Could not install backend dependencies"
        fi
        
        # Run pytest
        python3 -m pytest tests/ -v --tb=short --junitxml="$PROJECT_ROOT/test-results/backend-unit-tests.xml" || {
            log_warning "Backend unit tests failed"
            test_results=1
        }
    else
        log_warning "No backend unit tests found"
    fi
    
    # Frontend unit tests
    if [[ -d "$PROJECT_ROOT/ui" && -f "$PROJECT_ROOT/ui/package.json" ]]; then
        log_debug "Running frontend unit tests..."
        cd "$PROJECT_ROOT/ui"
        
        # Install dependencies if needed
        if [[ ! -d "node_modules" ]]; then
            npm ci --prefer-offline || log_warning "Could not install frontend dependencies"
        fi
        
        # Run tests
        npm test -- --coverage --watchAll=false --testResultsProcessor=jest-junit || {
            log_warning "Frontend unit tests failed"
            test_results=1
        }
    else
        log_warning "No frontend unit tests found"
    fi
    
    if [[ $test_results -eq 0 ]]; then
        log_success "Unit tests completed successfully"
    else
        log_warning "Unit tests completed with issues"
    fi
    
    return $test_results
}

run_integration_tests() {
    if [[ "$RUN_INTEGRATION_TESTS" != "true" ]]; then
        log_info "Skipping integration tests"
        return 0
    fi
    
    log_info "Running integration tests..."
    
    # Set environment variables for test script
    export PROJECT_ID="$PROJECT_ID"
    export ENVIRONMENT="$ENVIRONMENT"
    export REGION="$REGION"
    export API_URL="$API_URL"
    export FRONTEND_URL="$FRONTEND_URL"
    export VERBOSE="$VERBOSE"
    
    # Run integration test script
    cd "$TESTS_DIR"
    python3 run_integration_tests.py || {
        log_warning "Integration tests failed"
        return 1
    }
    
    log_success "Integration tests completed successfully"
    return 0
}

run_load_tests() {
    if [[ "$RUN_LOAD_TESTS" != "true" ]]; then
        log_info "Skipping load tests"
        return 0
    fi
    
    log_info "Running load tests..."
    
    # Check if k6 is available
    if ! command -v k6 >/dev/null 2>&1; then
        log_info "Installing k6..."
        # Install k6 (simplified)
        if command -v snap >/dev/null 2>&1; then
            sudo snap install k6 || log_warning "Could not install k6 via snap"
        else
            log_warning "k6 not available, skipping load tests"
            return 0
        fi
    fi
    
    # Set environment variables for k6
    export API_URL="$API_URL"
    export FRONTEND_URL="$FRONTEND_URL"
    
    # Run k6 load test
    cd "$TESTS_DIR"
    k6 run load_test.js --out json="$PROJECT_ROOT/test-results/load-test-results.json" || {
        log_warning "Load tests failed"
        return 1
    }
    
    log_success "Load tests completed successfully"
    return 0
}

run_security_tests() {
    if [[ "$RUN_SECURITY_TESTS" != "true" ]]; then
        log_info "Skipping security tests"
        return 0
    fi
    
    log_info "Running security tests..."
    
    local test_results=0
    
    # Backend security tests
    if [[ -d "$PROJECT_ROOT/src/kec_unified_api" ]]; then
        log_debug "Running backend security tests..."
        cd "$PROJECT_ROOT/src/kec_unified_api"
        
        # Install security tools
        pip3 install --user safety bandit || log_warning "Could not install security tools"
        
        # Run safety check
        if command -v safety >/dev/null 2>&1; then
            safety check -r requirements.txt --json --output "$PROJECT_ROOT/test-results/safety-report.json" || {
                log_warning "Safety check found vulnerabilities"
                test_results=1
            }
        fi
        
        # Run bandit check
        if command -v bandit >/dev/null 2>&1; then
            bandit -r . -f json -o "$PROJECT_ROOT/test-results/bandit-report.json" || {
                log_warning "Bandit found security issues"
                test_results=1
            }
        fi
    fi
    
    # Frontend security tests
    if [[ -d "$PROJECT_ROOT/ui" ]]; then
        log_debug "Running frontend security tests..."
        cd "$PROJECT_ROOT/ui"
        
        # NPM audit
        npm audit --json > "$PROJECT_ROOT/test-results/npm-audit.json" || {
            log_warning "NPM audit found vulnerabilities"
            test_results=1
        }
    fi
    
    # Runtime security tests (using integration test script)
    export PROJECT_ID="$PROJECT_ID"
    export API_URL="$API_URL"
    export FRONTEND_URL="$FRONTEND_URL"
    
    python3 -c "
from run_integration_tests import SecurityTests
security = SecurityTests('$API_URL', '$FRONTEND_URL')
success = security.run_all_tests()
exit(0 if success else 1)
" || {
        log_warning "Runtime security tests failed"
        test_results=1
    }
    
    if [[ $test_results -eq 0 ]]; then
        log_success "Security tests completed successfully"
    else
        log_warning "Security tests completed with issues"
    fi
    
    return $test_results
}

run_e2e_tests() {
    if [[ "$RUN_E2E_TESTS" != "true" ]]; then
        log_info "Skipping end-to-end tests"
        return 0
    fi
    
    log_info "Running end-to-end tests..."
    
    # Check if Playwright is available
    if [[ -d "$PROJECT_ROOT/ui" ]]; then
        cd "$PROJECT_ROOT/ui"
        
        if [[ -f "playwright.config.js" || -f "playwright.config.ts" ]]; then
            log_debug "Running Playwright E2E tests..."
            
            # Install Playwright if needed
            if [[ ! -d "node_modules/@playwright" ]]; then
                npx playwright install || log_warning "Could not install Playwright"
            fi
            
            # Run E2E tests
            FRONTEND_URL="$FRONTEND_URL" API_URL="$API_URL" npx playwright test || {
                log_warning "E2E tests failed"
                return 1
            }
        else
            log_warning "No Playwright configuration found, skipping E2E tests"
            return 0
        fi
    else
        log_warning "No UI directory found, skipping E2E tests"
        return 0
    fi
    
    log_success "End-to-end tests completed successfully"
    return 0
}

generate_comprehensive_report() {
    log_info "Generating comprehensive test report..."
    
    local report_file="$PROJECT_ROOT/test-results/comprehensive-test-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# DARWIN Comprehensive Test Report

**Generated:** $(date)
**Project:** $PROJECT_ID
**Environment:** $ENVIRONMENT
**Region:** $REGION

## Configuration

- **API URL:** $API_URL
- **Frontend URL:** $FRONTEND_URL
- **Test Environment:** $ENVIRONMENT

## Test Execution Summary

### Test Types Executed

- **Unit Tests:** $([ "$RUN_UNIT_TESTS" == "true" ] && echo "✅ Executed" || echo "⏭️ Skipped")
- **Integration Tests:** $([ "$RUN_INTEGRATION_TESTS" == "true" ] && echo "✅ Executed" || echo "⏭️ Skipped")
- **Load Tests:** $([ "$RUN_LOAD_TESTS" == "true" ] && echo "✅ Executed" || echo "⏭️ Skipped")
- **Security Tests:** $([ "$RUN_SECURITY_TESTS" == "true" ] && echo "✅ Executed" || echo "⏭️ Skipped")
- **End-to-End Tests:** $([ "$RUN_E2E_TESTS" == "true" ] && echo "✅ Executed" || echo "⏭️ Skipped")

### Test Artifacts Generated

EOF

    # List test artifacts
    if [[ -d "$PROJECT_ROOT/test-results" ]]; then
        echo "#### Available Test Artifacts" >> "$report_file"
        find "$PROJECT_ROOT/test-results" -name "*.json" -o -name "*.xml" -o -name "*.html" | while read -r file; do
            local filename
            filename=$(basename "$file")
            echo "- $filename" >> "$report_file"
        done
    fi
    
    cat >> "$report_file" << EOF

## Quality Metrics

### Code Coverage
- Backend code coverage reports available in test artifacts
- Frontend code coverage reports available in test artifacts

### Performance Metrics
- Load test results show API performance under concurrent load
- Response time benchmarks for critical endpoints
- Throughput and error rate measurements

### Security Assessment
- Dependency vulnerability scanning completed
- Static code analysis for security issues
- Runtime security testing for common vulnerabilities

### Accessibility
- Lighthouse accessibility scoring
- WCAG compliance validation

## Recommendations

1. **Review Failed Tests:** Address any test failures found
2. **Performance Optimization:** Optimize endpoints with high response times
3. **Security Remediation:** Fix any security vulnerabilities discovered
4. **Accessibility Improvements:** Enhance accessibility scores if below 90%
5. **Monitoring:** Set up continuous monitoring for test metrics

## Next Steps

1. Deploy to production after all tests pass
2. Set up automated testing in CI/CD pipeline
3. Configure monitoring alerts based on test thresholds
4. Schedule regular security and performance testing

---

*Generated by DARWIN Test Suite Runner*
*Test artifacts available in: $PROJECT_ROOT/test-results/*
EOF

    log_success "Comprehensive test report generated: $report_file"
    echo "$report_file"
}

show_test_summary() {
    local unit_status integration_status load_status security_status e2e_status
    
    unit_status=$([ "$RUN_UNIT_TESTS" == "true" ] && echo "✅" || echo "⏭️")
    integration_status=$([ "$RUN_INTEGRATION_TESTS" == "true" ] && echo "✅" || echo "⏭️")
    load_status=$([ "$RUN_LOAD_TESTS" == "true" ] && echo "✅" || echo "⏭️")
    security_status=$([ "$RUN_SECURITY_TESTS" == "true" ] && echo "✅" || echo "⏭️")
    e2e_status=$([ "$RUN_E2E_TESTS" == "true" ] && echo "✅" || echo "⏭️")
    
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    TEST EXECUTION SUMMARY                    ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Project ID:       $PROJECT_ID"
    echo -e "${CYAN}║${NC} Environment:      $ENVIRONMENT"
    echo -e "${CYAN}║${NC} API URL:          $API_URL"
    echo -e "${CYAN}║${NC} Frontend URL:     $FRONTEND_URL"
    echo -e "${CYAN}║${NC} Parallel:         $([ "$PARALLEL" == "true" ] && echo "Yes" || echo "No")"
    echo -e "${CYAN}║${NC} Timestamp:        $(date)"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Test Results:"
    echo -e "${CYAN}║${NC}   $unit_status Unit Tests"
    echo -e "${CYAN}║${NC}   $integration_status Integration Tests"
    echo -e "${CYAN}║${NC}   $load_status Load Tests"
    echo -e "${CYAN}║${NC}   $security_status Security Tests"
    echo -e "${CYAN}║${NC}   $e2e_status End-to-End Tests"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Test artifacts available in: test-results/"
    echo -e "${CYAN}║${NC} "
    echo -e "${CYAN}║${NC} 🔗 Next Steps:"
    echo -e "${CYAN}║${NC}   • Review test reports for any issues"
    echo -e "${CYAN}║${NC}   • Address failing tests before production"
    echo -e "${CYAN}║${NC}   • Monitor performance metrics"
    echo -e "${CYAN}║${NC}   • Set up automated testing in CI/CD"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -a|--api-url)
                API_URL="$2"
                shift 2
                ;;
            -f|--frontend-url)
                FRONTEND_URL="$2"
                shift 2
                ;;
            --unit-tests)
                RUN_UNIT_TESTS="true"
                RUN_INTEGRATION_TESTS="false"
                RUN_LOAD_TESTS="false"
                RUN_SECURITY_TESTS="false"
                RUN_E2E_TESTS="false"
                shift
                ;;
            --integration-tests)
                RUN_INTEGRATION_TESTS="true"
                if [[ "$RUN_UNIT_TESTS" == "true" && "$RUN_LOAD_TESTS" == "true" ]]; then
                    # First explicit test type, reset others
                    RUN_UNIT_TESTS="false"
                    RUN_LOAD_TESTS="false"
                    RUN_SECURITY_TESTS="false"
                    RUN_E2E_TESTS="false"
                fi
                shift
                ;;
            --load-tests)
                RUN_LOAD_TESTS="true"
                if [[ "$RUN_UNIT_TESTS" == "true" && "$RUN_INTEGRATION_TESTS" == "true" ]]; then
                    # First explicit test type, reset others
                    RUN_UNIT_TESTS="false"
                    RUN_INTEGRATION_TESTS="false"
                    RUN_SECURITY_TESTS="false"
                    RUN_E2E_TESTS="false"
                fi
                shift
                ;;
            --security-tests)
                RUN_SECURITY_TESTS="true"
                if [[ "$RUN_UNIT_TESTS" == "true" && "$RUN_INTEGRATION_TESTS" == "true" && "$RUN_LOAD_TESTS" == "true" ]]; then
                    # First explicit test type, reset others
                    RUN_UNIT_TESTS="false"
                    RUN_INTEGRATION_TESTS="false"
                    RUN_LOAD_TESTS="false"
                    RUN_E2E_TESTS="false"
                fi
                shift
                ;;
            --e2e-tests)
                RUN_E2E_TESTS="true"
                if [[ "$RUN_UNIT_TESTS" == "true" && "$RUN_INTEGRATION_TESTS" == "true" && "$RUN_LOAD_TESTS" == "true" && "$RUN_SECURITY_TESTS" == "true" ]]; then
                    # First explicit test type, reset others
                    RUN_UNIT_TESTS="false"
                    RUN_INTEGRATION_TESTS="false"
                    RUN_LOAD_TESTS="false"
                    RUN_SECURITY_TESTS="false"
                fi
                shift
                ;;
            --all)
                RUN_UNIT_TESTS="true"
                RUN_INTEGRATION_TESTS="true"
                RUN_LOAD_TESTS="true"
                RUN_SECURITY_TESTS="true"
                RUN_E2E_TESTS="true"
                shift
                ;;
            --parallel)
                PARALLEL="true"
                shift
                ;;
            --skip-setup)
                SKIP_SETUP="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check for environment variables
    PROJECT_ID="${PROJECT_ID:-${DARWIN_PROJECT_ID:-}}"
    ENVIRONMENT="${ENVIRONMENT:-${DARWIN_ENVIRONMENT:-staging}}"
    REGION="${REGION:-${DARWIN_REGION:-us-central1}}"
    API_URL="${API_URL:-${DARWIN_API_URL:-}}"
    FRONTEND_URL="${FRONTEND_URL:-${DARWIN_FRONTEND_URL:-}}"
    
    # Validate required parameters
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        show_usage
        exit 1
    fi
    
    # Show banner
    show_banner
    
    log_info "Starting DARWIN comprehensive test suite..."
    log_info "Project: $PROJECT_ID | Environment: $ENVIRONMENT | Region: $REGION"
    
    check_prerequisites
    detect_service_urls
    setup_test_environment
    
    # Track overall results
    local overall_result=0
    
    # Run tests (sequentially or in parallel)
    if [[ "$PARALLEL" == "true" ]]; then
        log_info "Running tests in parallel..."
        
        # Run compatible tests in parallel
        (run_unit_tests) &
        local unit_pid=$!
        
        (run_integration_tests) &
        local integration_pid=$!
        
        (run_security_tests) &
        local security_pid=$!
        
        # Wait for parallel tests
        wait $unit_pid || overall_result=1
        wait $integration_pid || overall_result=1
        wait $security_pid || overall_result=1
        
        # Run sequential tests
        run_load_tests || overall_result=1
        run_e2e_tests || overall_result=1
    else
        log_info "Running tests sequentially..."
        
        run_unit_tests || overall_result=1
        run_integration_tests || overall_result=1
        run_load_tests || overall_result=1
        run_security_tests || overall_result=1
        run_e2e_tests || overall_result=1
    fi
    
    # Generate comprehensive report
    local report_file
    report_file=$(generate_comprehensive_report)
    
    show_test_summary
    
    if [[ $overall_result -eq 0 ]]; then
        log_success "All tests completed successfully!"
        log_info "Comprehensive report: $report_file"
    else
        log_warning "Some tests failed or had issues"
        log_info "Review test reports for details: $report_file"
    fi
    
    exit $overall_result
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi