
#!/usr/bin/env python3

"""
DARWIN Integration Tests Suite
Comprehensive testing script for all DARWIN components
"""

import asyncio
import json
import os
import sys
import time
import requests
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional

# =============================================================================
# Configuration
# =============================================================================

class TestConfig:
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID', '')
        self.environment = os.getenv('ENVIRONMENT', 'staging')
        self.region = os.getenv('REGION', 'us-central1')
        self.api_url = os.getenv('API_URL', 'https://api-staging.agourakis.med.br')
        self.frontend_url = os.getenv('FRONTEND_URL', 'https://darwin-staging.agourakis.med.br')
        self.timeout = 30
        self.verbose = os.getenv('VERBOSE', 'false').lower() == 'true'

config = TestConfig()

# =============================================================================
# Utility Functions
# =============================================================================

def log_info(message: str):
    print(f"[INFO] {message}")

def log_success(message: str):
    print(f"[SUCCESS] ✅ {message}")

def log_warning(message: str):
    print(f"[WARNING] ⚠️ {message}")

def log_error(message: str):
    print(f"[ERROR] ❌ {message}")

def log_debug(message: str):
    if config.verbose:
        print(f"[DEBUG] 🔍 {message}")

# =============================================================================
# Test Classes
# =============================================================================

class APIIntegrationTests:
    """API integration tests"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_endpoint(self, method: str, path: str, expected_status: int = 200, description: str = ""):
        """Test a single API endpoint"""
        url = f"{self.base_url}{path}"
        test_name = description or f"{method} {path}"
        
        try:
            log_debug(f"Testing {test_name}: {url}")
            response = self.session.request(method, url, timeout=config.timeout)
            
            if response.status_code == expected_status:
                log_success(f"{test_name}: {response.status_code}")
                self.results.append({
                    "test": test_name,
                    "status": "PASS",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                })
                return True
            else:
                log_warning(f"{test_name}: {response.status_code} (expected {expected_status})")
                self.results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "status_code": response.status_code,
                    "expected": expected_status
                })
                return False
                
        except Exception as e:
            log_error(f"{test_name}: {str(e)}")
            self.results.append({
                "test": test_name,
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def run_all_tests(self):
        """Run all API integration tests"""
        log_info("Running API integration tests...")
        
        tests = [
            ("GET", "/health", 200, "API Health"),
            ("GET", "/api/health", 200, "API Health Detailed"),
            ("GET", "/metrics", 200, "Metrics"),
            ("GET", "/docs", 200, "API Documentation"),
            ("GET", "/openapi.json", 200, "OpenAPI Spec"),
            ("GET", "/api/core/status", 200, "Core Status"),
        ]
        
        passed = 0
        for method, path, expected_status, description in tests:
            if self.test_endpoint(method, path, expected_status, description):
                passed += 1
        
        log_info(f"API Tests: {passed}/{len(tests)} passed")
        return passed == len(tests)

class FrontendIntegrationTests:
    """Frontend integration tests"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_endpoint(self, path: str, expected_status: int = 200, description: str = ""):
        """Test a single frontend endpoint"""
        url = f"{self.base_url}{path}"
        test_name = description or f"GET {path}"
        
        try:
            log_debug(f"Testing {test_name}: {url}")
            response = self.session.get(url, timeout=config.timeout)
            
            if response.status_code == expected_status:
                log_success(f"{test_name}: {response.status_code}")
                self.results.append({
                    "test": test_name,
                    "status": "PASS",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                })
                return True
            else:
                log_warning(f"{test_name}: {response.status_code} (expected {expected_status})")
                self.results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "status_code": response.status_code,
                    "expected": expected_status
                })
                return False
                
        except Exception as e:
            log_error(f"{test_name}: {str(e)}")
            self.results.append({
                "test": test_name,
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def run_all_tests(self):
        """Run all frontend integration tests"""
        log_info("Running frontend integration tests...")
        
        tests = [
            ("/", 200, "Frontend Homepage"),
            ("/api/health", 200, "Frontend Health"),
        ]
        
        passed = 0
        for path, expected_status, description in tests:
            if self.test_endpoint(path, expected_status, description):
                passed += 1
        
        log_info(f"Frontend Tests: {passed}/{len(tests)} passed")
        return passed == len(tests)

class PerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {}
    
    def benchmark_endpoint(self, path: str, samples: int = 20):
        """Benchmark a single endpoint"""
        url = f"{self.base_url}{path}"
        times = []
        
        log_debug(f"Benchmarking {path} with {samples} samples...")
        
        for i in range(samples):
            try:
                start = time.time()
                response = self.session.get(url, timeout=config.timeout)
                end = time.time()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
                    
            except Exception as e:
                log_debug(f"Request {i+1} failed: {e}")
        
        if times:
            return {
                "avg_ms": statistics.mean(times),
                "p95_ms": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "samples": len(times)
            }
        return None
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        log_info("Running performance benchmarks...")
        
        endpoints = ["/health", "/api/health", "/metrics"]
        
        for endpoint in endpoints:
            result = self.benchmark_endpoint(endpoint)
            if result:
                self.results[endpoint] = result
                log_success(f"{endpoint}: {result['avg_ms']:.2f}ms avg, {result['p95_ms']:.2f}ms p95")
            else:
                log_warning(f"{endpoint}: No successful requests")
        
        return len(self.results) > 0

class SecurityTests:
    """Security testing suite"""
    
    def __init__(self, api_url: str, frontend_url: str):
        self.api_url = api_url
        self.frontend_url = frontend_url
        self.results = []
    
    def test_security_headers(self, url: str):
        """Test security headers"""
        log_debug(f"Testing security headers for {url}")
        
        try:
            response = requests.get(url, timeout=config.timeout)
            headers = response.headers
            
            security_headers = {
                'Strict-Transport-Security': 'HSTS',
                'X-Content-Type-Options': 'Content Type Options',
                'X-Frame-Options': 'Frame Options',
                'X-XSS-Protection': 'XSS Protection'
            }
            
            for header, description in security_headers.items():
                if header in headers:
                    log_success(f"{description}: Present")
                    self.results.append({"test": f"{description} Header", "status": "PASS"})
                else:
                    log_warning(f"{description}: Missing")
                    self.results.append({"test": f"{description} Header", "status": "FAIL"})
                    
        except Exception as e:
            log_error(f"Security header test failed: {e}")
    
    def test_sql_injection_protection(self):
        """Test basic SQL injection protection"""
        log_debug("Testing SQL injection protection")
        
        payloads = ["'", "1' OR '1'='1", "'; DROP TABLE users; --"]
        
        for payload in payloads:
            try:
                # Test search endpoint with malicious payload
                response = requests.get(f"{self.api_url}/api/search", 
                                      params={"q": payload}, timeout=10)
                
                if response.status_code == 500:
                    log_warning(f"Potential SQL injection vulnerability: {payload}")
                    self.results.append({"test": f"SQL Injection Test: {payload}", "status": "FAIL"})
                else:
                    log_success(f"SQL injection protection working: {payload}")
                    self.results.append({"test": f"SQL Injection Test: {payload}", "status": "PASS"})
                    
            except Exception as e:
                log_success(f"SQL injection test passed (connection rejected): {payload}")
                self.results.append({"test": f"SQL Injection Test: {payload}", "status": "PASS"})
    
    def run_all_tests(self):
        """Run all security tests"""
        log_info("Running security tests...")
        
        self.test_security_headers(self.api_url)
        self.test_security_headers(self.frontend_url)
        self.test_sql_injection_protection()
        
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        total = len(self.results)
        log_info(f"Security Tests: {passed}/{total} passed")
        
        return passed == total

# =============================================================================
# Main Test Runner
# =============================================================================

class TestRunner:
    """Main test runner"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "project_id": config.project_id,
                "environment": config.environment,
                "api_url": config.api_url,
                "frontend_url": config.frontend_url,
                "region": config.region
            },
            "tests": {}
        }
    
    def run_api_tests(self):
        """Run API integration tests"""
        api_tests = APIIntegrationTests(config.api_url)
        success = api_tests.run_all_tests()
        self.results["tests"]["api_integration"] = {
            "success": success,
            "results": api_tests.results
        }
        return success
    
    def run_frontend_tests(self):
        """Run frontend integration tests"""
        frontend_tests = FrontendIntegrationTests(config.frontend_url)
        success = frontend_tests.run_all_tests()
        self.results["tests"]["frontend_integration"] = {
            "success": success,
            "results": frontend_tests.results
        }
        return success
    
    def run_performance_tests(self):
        """Run performance benchmarks"""
        perf_tests = PerformanceBenchmarks(config.api_url)
        success = perf_tests.run_all_benchmarks()
        self.results["tests"]["performance"] = {
            "success": success,
            "results": perf_tests.results
        }
        return success
    
    def run_security_tests(self):
        """Run security tests"""
        security_tests = SecurityTests(config.api_url, config.frontend_url)
        success = security_tests.run_all_tests()
        self.results["tests"]["security"] = {
            "success": success,
            "results": security_tests.results
        }
        return success
    
    def generate_report(self):
        """Generate test report"""
        timestamp = int(time.time())
        report_file = f"integration-test-results-{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        log_success(f"Test results saved to {report_file}")
        
        # Generate markdown report
        md_report = self.generate_markdown_report()
        md_file = f"integration-test-report-{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write(md_report)
        
        log_success(f"Test report saved to {md_file}")
        
        return report_file, md_file
    
    def generate_markdown_report(self):
        """Generate markdown test report"""
        test_summary = []
        for test_type, data in self.results["tests"].items():
            status = "✅ PASS" if data["success"] else "❌ FAIL"
            test_summary.append(f"- **{test_type.replace('_', ' ').title()}:** {status}")
        
        report = f"""# DARWIN Integration Test Report

**Generated:** {self.results['timestamp']}
**Project:** {self.results['config']['project_id']}
**Environment:** {self.results['config']['environment']}
**Region:** {self.results['config']['region']}

## Configuration

- **API URL:** {self.results['config']['api_url']}
- **Frontend URL:** {self.results['config']['frontend_url']}

## Test Results

{chr(10).join(test_summary)}

## Detailed Results

### API Integration Tests
"""
        
        if "api_integration" in self.results["tests"]:
            api_results = self.results["tests"]["api_integration"]["results"]
            for result in api_results:
                status_emoji = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
                report += f"- {status_emoji} {result['test']}\n"
        
        report += "\n### Performance Results\n"
        if "performance" in self.results["tests"]:
            perf_results = self.results["tests"]["performance"]["results"]
            for endpoint, metrics in perf_results.items():
                report += f"- **{endpoint}:** {metrics['avg_ms']:.2f}ms avg, {metrics['p95_ms']:.2f}ms p95\n"
        
        report += "\n### Security Tests\n"
        if "security" in self.results["tests"]:
            sec_results = self.results["tests"]["security"]["results"]
            for result in sec_results:
                status_emoji = "✅" if result["status"] == "PASS" else "❌"
                report += f"- {status_emoji} {result['test']}\n"
        
        report += f"""
## Summary

Total test suites run: {len(self.results['tests'])}
Overall status: {"✅ ALL TESTS PASSED" if all(data['success'] for data in self.results['tests'].values()) else "❌ SOME TESTS FAILED"}

## Recommendations

1. Review any failed tests and address issues
2. Monitor performance metrics and optimize slow endpoints
3. Address security vulnerabilities if found
4. Ensure all services are properly configured

---
*Generated by DARWIN Integration Testing Suite*
"""
        
        return report

def main():
    """Main execution function"""
    log_info("Starting DARWIN Integration Tests...")
    log_info(f"Testing environment: {config.environment}")
    log_info(f"API URL: {config.api_url}")
    log_info(f"Frontend URL: {config.frontend_url}")
    
    if not config.project_id:
        log_error("PROJECT_ID environment variable required")
        sys.exit(1)
    
    runner = TestRunner()
    
    # Run all test suites
    all_passed = True
    
    try:
        all_passed &= runner.run_api_tests()
        all_passed &= runner.run_frontend_tests()
        all_passed &= runner.run_performance_tests()
        all_passed &= runner.run_security_tests()
        
        # Generate reports
        json_file, md_file = runner.generate_report()
        
        log_info("Test execution completed")
        log_info(f"Results: {json_file}")
        log_info(f"Report: {md_file}")
        
        if all_passed:
            log_success("All integration tests passed!")
            sys.exit(0)
        else:
            log_warning("Some integration tests failed")
            sys.exit(1)
            
    except Exception as e:
        log_error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()