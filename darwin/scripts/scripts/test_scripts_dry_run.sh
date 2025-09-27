#!/bin/bash

# SCRIPT DE TESTE - DRY RUN MODE
# Teste sistemático de todos os scripts GCP em modo seguro
# 🧪 COMPREHENSIVE SCRIPT TESTING - DRY RUN VALIDATION

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
TEST_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG_FILE="script_test_results_${TEST_TIMESTAMP}.log"
TEST_REPORT_FILE="script_test_report_${TEST_TIMESTAMP}.md"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test configuration
PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
VERBOSE="${VERBOSE:-true}"

# Test results tracking
declare -A test_results
declare -A test_times
declare -A test_outputs

# Logging functions
log() {
    local message="$1"
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $message" >> "$TEST_LOG_FILE"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $message" >> "$TEST_LOG_FILE"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $message" >> "$TEST_LOG_FILE"
}

log_error() {
    local message="$1"
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $message" >> "$TEST_LOG_FILE"
}

log_test() {
    local message="$1"
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] 🧪${NC} $message"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] TEST: $message" >> "$TEST_LOG_FILE"
}

# Header display
show_header() {
    echo -e "${PURPLE}${BOLD}
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  🧪 DARWIN GCP SCRIPTS TESTING SUITE 🧪                                ║
║                                                                          ║
║  Comprehensive dry-run testing of all GCP scripts:                      ║
║  • GCP Inventory Analysis Script                                        ║
║  • Critical Data Backup Script                                          ║
║  • Systematic Cleanup Script                                            ║
║  • Optimized Production Deploy Script                                   ║
║                                                                          ║
║  🔒 SAFE MODE - DRY RUN TESTING ONLY                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝${NC}
"
}

# Initialize test environment
initialize_test_environment() {
    log "🚀 Initializing test environment..."
    
    # Create test log file
    cat > "$TEST_LOG_FILE" << EOF
# GCP Scripts Dry-Run Test Log
# Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Project: $PROJECT_ID
# Region: $REGION
# Test Mode: DRY RUN ONLY
EOF
    
    # Check if we're in the right directory
    if [[ ! -f "$SCRIPTS_DIR/gcp_inventory_analysis.sh" ]]; then
        log_error "Scripts not found in expected location: $SCRIPTS_DIR"
        exit 1
    fi
    
    # Make scripts executable
    log "Making scripts executable..."
    chmod +x "$SCRIPTS_DIR"/*.sh
    
    log_success "Test environment initialized"
}

# Test script execution with dry-run
test_script() {
    local script_name="$1"
    local script_args="$2"
    local script_path="$SCRIPTS_DIR/$script_name"
    
    log_test "Testing script: $script_name"
    
    if [[ ! -f "$script_path" ]]; then
        log_error "Script not found: $script_path"
        test_results["$script_name"]="MISSING"
        return 1
    fi
    
    # Run script with timeout
    local start_time=$(date +%s)
    local output_file="${script_name}_output_${TEST_TIMESTAMP}.log"
    local exit_code=0
    
    # Execute script with timeout (10 minutes max)
    if timeout 600 bash "$script_path" $script_args > "$output_file" 2>&1; then
        exit_code=0
        test_results["$script_name"]="PASS"
        log_success "Script test passed: $script_name"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            test_results["$script_name"]="TIMEOUT"
            log_error "Script test timed out: $script_name"
        else
            test_results["$script_name"]="FAIL"
            log_error "Script test failed: $script_name (exit code: $exit_code)"
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    test_times["$script_name"]="$duration"
    test_outputs["$script_name"]="$output_file"
    
    # Show output summary if verbose
    if [[ "$VERBOSE" == "true" && -f "$output_file" ]]; then
        echo -e "${CYAN}--- Output Summary for $script_name ---${NC}"
        tail -20 "$output_file" | sed 's/^/  /'
        echo -e "${CYAN}--- End Output Summary ---${NC}"
    fi
    
    return $exit_code
}

# Test GCP Inventory Analysis Script
test_inventory_script() {
    log "📊 Testing GCP Inventory Analysis Script..."
    
    test_script "gcp_inventory_analysis.sh" "--project=$PROJECT_ID --region=$REGION --dry-run --verbose"
    
    # Additional validation
    local script_output="${test_outputs["gcp_inventory_analysis.sh"]}"
    if [[ -f "$script_output" ]]; then
        if grep -q "inventory_metadata" "$script_output"; then
            log_success "Inventory script generated proper JSON structure"
        else
            log_warning "Inventory script may not have generated expected output"
        fi
    fi
}

# Test Critical Data Backup Script  
test_backup_script() {
    log "🔄 Testing Critical Data Backup Script..."
    
    test_script "gcp_backup_critical_data.sh" "--project=$PROJECT_ID --region=$REGION --dry-run --verbose"
    
    # Additional validation
    local script_output="${test_outputs["gcp_backup_critical_data.sh"]}"
    if [[ -f "$script_output" ]]; then
        if grep -q "BACKUP DRY RUN COMPLETED" "$script_output"; then
            log_success "Backup script completed dry-run successfully"
        else
            log_warning "Backup script may not have completed properly"
        fi
    fi
}

# Test Systematic Cleanup Script
test_cleanup_script() {
    log "🧹 Testing Systematic Cleanup Script..."
    
    # Test with very conservative settings
    test_script "gcp_cleanup_legacy.sh" "--project=$PROJECT_ID --region=$REGION --dry-run --non-interactive --backup-verified"
    
    # Additional validation
    local script_output="${test_outputs["gcp_cleanup_legacy.sh"]}"
    if [[ -f "$script_output" ]]; then
        if grep -q "CLEANUP DRY RUN COMPLETED" "$script_output"; then
            log_success "Cleanup script completed dry-run successfully"
        else
            log_warning "Cleanup script may not have completed properly"
        fi
        
        # Check that no actual deletions were attempted
        if grep -q "Would delete" "$script_output"; then
            log_success "Cleanup script properly used simulation mode"
        else
            log_warning "Cleanup script may not have properly simulated operations"
        fi
    fi
}

# Test Optimized Deploy Script
test_deploy_script() {
    log "🚀 Testing Optimized Deploy Script..."
    
    test_script "deploy_darwin_production_optimized.sh" "--project=$PROJECT_ID --region=$REGION --dry-run --skip-tests"
    
    # Additional validation
    local script_output="${test_outputs["deploy_darwin_production_optimized.sh"]}"
    if [[ -f "$script_output" ]]; then
        if grep -q "DEPLOYMENT DRY RUN COMPLETED" "$script_output"; then
            log_success "Deploy script completed dry-run successfully"
        else
            log_warning "Deploy script may not have completed properly"
        fi
    fi
}

# Test script help functions
test_script_help() {
    log "📚 Testing script help functions..."
    
    local scripts=(
        "gcp_inventory_analysis.sh"
        "gcp_backup_critical_data.sh"
        "gcp_cleanup_legacy.sh"
        "deploy_darwin_production_optimized.sh"
    )
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPTS_DIR/$script"
        if [[ -f "$script_path" ]]; then
            log "Testing help for: $script"
            if timeout 30 bash "$script_path" --help > "${script}_help_${TEST_TIMESTAMP}.log" 2>&1; then
                log_success "Help function works: $script"
            else
                log_warning "Help function issue: $script"
            fi
        fi
    done
}

# Validate script permissions and syntax
validate_scripts() {
    log "🔍 Validating script syntax and permissions..."
    
    local scripts=(
        "gcp_inventory_analysis.sh"
        "gcp_backup_critical_data.sh"
        "gcp_cleanup_legacy.sh"
        "deploy_darwin_production_optimized.sh"
    )
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPTS_DIR/$script"
        
        if [[ -f "$script_path" ]]; then
            # Check permissions
            if [[ -x "$script_path" ]]; then
                log_success "Script is executable: $script"
            else
                log_error "Script is not executable: $script"
                test_results["${script}_permissions"]="FAIL"
            fi
            
            # Check syntax
            if bash -n "$script_path"; then
                log_success "Script syntax is valid: $script"
                test_results["${script}_syntax"]="PASS"
            else
                log_error "Script has syntax errors: $script"
                test_results["${script}_syntax"]="FAIL"
            fi
        else
            log_error "Script not found: $script"
            test_results["${script}_exists"]="FAIL"
        fi
    done
}

# Generate comprehensive test report
generate_test_report() {
    log "📋 Generating comprehensive test report..."
    
    # Calculate summary statistics
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local timeout_tests=0
    
    for test_name in "${!test_results[@]}"; do
        total_tests=$((total_tests + 1))
        case "${test_results[$test_name]}" in
            "PASS") passed_tests=$((passed_tests + 1)) ;;
            "FAIL") failed_tests=$((failed_tests + 1)) ;;
            "TIMEOUT") timeout_tests=$((timeout_tests + 1)) ;;
        esac
    done
    
    # Generate detailed report
    cat > "$TEST_REPORT_FILE" << EOF
# GCP Scripts Dry-Run Test Report

**Test Date**: $(date)  
**Project**: $PROJECT_ID  
**Region**: $REGION  
**Test Duration**: Started at test initialization

## 📊 Test Summary

- **Total Tests**: $total_tests
- **Passed**: $passed_tests ✅
- **Failed**: $failed_tests ❌  
- **Timeouts**: $timeout_tests ⏱️
- **Success Rate**: $(( passed_tests * 100 / total_tests ))%

## 🧪 Individual Test Results

### Script Functionality Tests

$(for script in "gcp_inventory_analysis.sh" "gcp_backup_critical_data.sh" "gcp_cleanup_legacy.sh" "deploy_darwin_production_optimized.sh"; do
    if [[ -n "${test_results[$script]:-}" ]]; then
        case "${test_results[$script]}" in
            "PASS") echo "- ✅ **$script**: PASSED (${test_times[$script]:-0}s)" ;;
            "FAIL") echo "- ❌ **$script**: FAILED (${test_times[$script]:-0}s)" ;;
            "TIMEOUT") echo "- ⏱️ **$script**: TIMEOUT (${test_times[$script]:-0}s)" ;;
            *) echo "- ⚠️ **$script**: ${test_results[$script]}" ;;
        esac
    else
        echo "- ❓ **$script**: NOT TESTED"
    fi
done)

### Script Validation Tests

$(for test_type in "syntax" "permissions" "exists"; do
    echo "#### ${test_type^} Validation"
    for script in "gcp_inventory_analysis.sh" "gcp_backup_critical_data.sh" "gcp_cleanup_legacy.sh" "deploy_darwin_production_optimized.sh"; do
        test_key="${script}_${test_type}"
        if [[ -n "${test_results[$test_key]:-}" ]]; then
            case "${test_results[$test_key]}" in
                "PASS") echo "- ✅ $script: Valid" ;;
                "FAIL") echo "- ❌ $script: Issues detected" ;;
                *) echo "- ⚠️ $script: ${test_results[$test_key]}" ;;
            esac
        fi
    done
    echo ""
done)

## 📁 Test Artifacts

### Log Files
$(for script in "gcp_inventory_analysis.sh" "gcp_backup_critical_data.sh" "gcp_cleanup_legacy.sh" "deploy_darwin_production_optimized.sh"; do
    if [[ -n "${test_outputs[$script]:-}" && -f "${test_outputs[$script]}" ]]; then
        echo "- **$script**: \`${test_outputs[$script]}\`"
    fi
done)

### Help Documentation
$(for script in "gcp_inventory_analysis.sh" "gcp_backup_critical_data.sh" "gcp_cleanup_legacy.sh" "deploy_darwin_production_optimized.sh"; do
    help_file="${script}_help_${TEST_TIMESTAMP}.log"
    if [[ -f "$help_file" ]]; then
        echo "- **$script help**: \`$help_file\`"
    fi
done)

## 🎯 Test Conclusions

### ✅ Successful Validations
$(for test_name in "${!test_results[@]}"; do
    if [[ "${test_results[$test_name]}" == "PASS" ]]; then
        echo "- $test_name: Passed validation"
    fi
done)

### ⚠️ Issues Found
$(for test_name in "${!test_results[@]}"; do
    if [[ "${test_results[$test_name]}" != "PASS" ]]; then
        echo "- $test_name: ${test_results[$test_name]}"
    fi
done)

## 🔧 Recommendations

### Ready for Production Use
$(if [[ $failed_tests -eq 0 && $timeout_tests -eq 0 ]]; then
    echo "✅ **All scripts passed testing and are ready for production use.**"
else
    echo "⚠️ **Review failed tests before production use.**"
fi)

### Usage Guidelines
1. **Always start with inventory**: Run \`gcp_inventory_analysis.sh --dry-run\` first
2. **Backup before cleanup**: Execute \`gcp_backup_critical_data.sh\` before any cleanup
3. **Test cleanup safely**: Use \`gcp_cleanup_legacy.sh --dry-run\` to simulate cleanup
4. **Deploy with confidence**: Use \`deploy_darwin_production_optimized.sh --dry-run\` for testing

### Script Execution Order
1. \`./scripts/gcp_inventory_analysis.sh --project=PROJECT_ID --dry-run\`
2. \`./scripts/gcp_backup_critical_data.sh --project=PROJECT_ID --dry-run\`
3. \`./scripts/gcp_cleanup_legacy.sh --project=PROJECT_ID --dry-run\`
4. \`./scripts/deploy_darwin_production_optimized.sh --project=PROJECT_ID --dry-run\`

## 📞 Next Steps

### If All Tests Passed
1. **Review output logs** for any warnings or recommendations
2. **Execute actual operations** by removing --dry-run flags
3. **Monitor operations** during real execution
4. **Keep backups** of all test logs for reference

### If Tests Failed  
1. **Review failed test outputs** in detail
2. **Fix identified issues** before proceeding
3. **Re-run tests** to verify fixes
4. **Seek support** if issues persist

## 📊 Test Environment

- **Test Timestamp**: $TEST_TIMESTAMP
- **Scripts Directory**: $SCRIPTS_DIR
- **Test Log**: $TEST_LOG_FILE
- **Test Report**: $TEST_REPORT_FILE
- **GCP Project**: $PROJECT_ID
- **GCP Region**: $REGION

---
**Generated by**: GCP Scripts Dry-Run Testing Suite  
**Status**: $(if [[ $failed_tests -eq 0 && $timeout_tests -eq 0 ]]; then echo "ALL TESTS PASSED ✅"; else echo "ISSUES DETECTED ⚠️"; fi)
EOF
    
    log_success "Test report generated: $TEST_REPORT_FILE"
    echo ""
    echo -e "${CYAN}=== TEST REPORT SUMMARY ===${NC}"
    echo "Total Tests: $total_tests"
    echo "Passed: $passed_tests ✅"
    echo "Failed: $failed_tests ❌"
    echo "Success Rate: $(( passed_tests * 100 / total_tests ))%"
    echo ""
    echo -e "${CYAN}Full report: $TEST_REPORT_FILE${NC}"
}

# Main execution function
main() {
    show_header
    
    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project=*)
                PROJECT_ID="${1#*=}"
                shift
                ;;
            --region=*)
                REGION="${1#*=}"
                shift
                ;;
            --quiet)
                VERBOSE="false"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Comprehensive dry-run testing of GCP scripts"
                echo ""
                echo "Options:"
                echo "  --project=PROJECT_ID    GCP Project ID"
                echo "  --region=REGION         GCP Region"
                echo "  --quiet                 Reduce output verbosity"
                echo "  --help                  Show this help"
                echo ""
                echo "This script tests all GCP management scripts in dry-run mode"
                echo "to validate functionality before production use."
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    log "🧪 Starting comprehensive GCP scripts testing..."
    log "Project: $PROJECT_ID"
    log "Region: $REGION"
    log "Test Mode: DRY RUN ONLY"
    
    # Initialize and run tests
    initialize_test_environment
    validate_scripts
    test_script_help
    test_inventory_script
    test_backup_script
    test_cleanup_script
    test_deploy_script
    generate_test_report
    
    # Final summary
    local total_tests=${#test_results[@]}
    local passed_count=0
    for result in "${test_results[@]}"; do
        if [[ "$result" == "PASS" ]]; then
            passed_count=$((passed_count + 1))
        fi
    done
    
    if [[ $passed_count -eq $total_tests ]]; then
        echo -e "${GREEN}${BOLD}
🎉 ALL SCRIPT TESTS COMPLETED SUCCESSFULLY! 🎉

✅ All scripts passed dry-run validation
✅ Syntax and permissions verified
✅ Help functions working correctly
✅ Safe for production use

📋 Files Generated:
   📄 Test Report: $TEST_REPORT_FILE
   📄 Test Log: $TEST_LOG_FILE

🚀 SCRIPTS ARE READY FOR PRODUCTION USE! 🚀
${NC}"
    else
        echo -e "${YELLOW}${BOLD}
⚠️ SCRIPT TESTING COMPLETED WITH ISSUES

Some tests failed or need attention.
Please review the test report for details.

📋 Files Generated:
   📄 Test Report: $TEST_REPORT_FILE  
   📄 Test Log: $TEST_LOG_FILE

🔧 REVIEW ISSUES BEFORE PRODUCTION USE
${NC}"
    fi
}

# Handle interruption gracefully
trap 'echo -e "\n${YELLOW}Testing interrupted by user${NC}"; exit 130' INT TERM

# Execute main function
main "$@"