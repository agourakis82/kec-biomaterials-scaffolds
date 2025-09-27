#!/usr/bin/env python3
"""Test Vertex AI Setup - DARWIN Production Validation

🧪 VERTEX AI SETUP VALIDATION SCRIPT
Script épico para testar e validar setup completo do Vertex AI:
- Authentication verification
- Model access testing  
- Service account permissions
- Storage bucket access
- BigQuery connectivity
- Custom model endpoints

Usage: python scripts/test_vertex_ai.py
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from kec_unified_api.services.vertex_ai_client import VertexAIClient, VertexAIConfig, VertexAIModel
    from kec_unified_api.core.logging import setup_logging, get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Setup logging
setup_logging()
logger = get_logger("darwin.vertex_ai_test")

# Colors for console output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NC = '\033[0m'  # No Color

def print_header():
    """Print test header."""
    print(f"""
{Colors.PURPLE}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🧪 DARWIN VERTEX AI SETUP VALIDATION & TESTING 🧪         ║
║                                                              ║
║  Comprehensive testing of Vertex AI infrastructure:         ║
║  • Authentication & Service Accounts                        ║
║  • Model Access & Availability                              ║
║  • Storage & BigQuery Connectivity                          ║
║  • Custom Model Endpoints                                   ║
║  • Performance & Latency Testing                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.NC}
""")

def log_test(test_name: str, status: str = "running"):
    """Log test status with colors."""
    if status == "running":
        print(f"{Colors.BLUE}🧪 [TESTING]{Colors.NC} {test_name}...")
    elif status == "success":
        print(f"{Colors.GREEN}✅ [SUCCESS]{Colors.NC} {test_name}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️  [WARNING]{Colors.NC} {test_name}")
    elif status == "error":
        print(f"{Colors.RED}❌ [ERROR]{Colors.NC} {test_name}")
    elif status == "info":
        print(f"{Colors.CYAN}ℹ️  [INFO]{Colors.NC} {test_name}")

def print_section(section_name: str):
    """Print section header."""
    print(f"\n{Colors.WHITE}{Colors.BOLD}{'='*20} {section_name} {'='*20}{Colors.NC}")

class VertexAITester:
    """Comprehensive Vertex AI testing class."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.client: Optional[VertexAIClient] = None
        self.config: Optional[VertexAIConfig] = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Vertex AI tests."""
        print_header()
        
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_warned": 0,
            "test_details": {}
        }
        
        # Test categories
        test_categories = [
            ("Environment Setup", self.test_environment_setup),
            ("Authentication", self.test_authentication),
            ("Client Initialization", self.test_client_initialization),
            ("Model Access", self.test_model_access),
            ("Storage Access", self.test_storage_access),
            ("BigQuery Access", self.test_bigquery_access),
            ("Text Generation", self.test_text_generation),
            ("Custom Models", self.test_custom_models),
            ("Performance", self.test_performance)
        ]
        
        for category_name, test_function in test_categories:
            print_section(category_name)
            
            try:
                category_result = await test_function()
                test_results["test_details"][category_name] = category_result
                
                if category_result.get("status") == "success":
                    test_results["tests_passed"] += 1
                elif category_result.get("status") == "warning":
                    test_results["tests_warned"] += 1
                else:
                    test_results["tests_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Test category {category_name} failed: {e}")
                test_results["test_details"][category_name] = {
                    "status": "error",
                    "error": str(e)
                }
                test_results["tests_failed"] += 1
        
        # Determine overall status
        if test_results["tests_failed"] == 0:
            if test_results["tests_warned"] == 0:
                test_results["overall_status"] = "success"
            else:
                test_results["overall_status"] = "warning"
        else:
            test_results["overall_status"] = "failed"
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    async def test_environment_setup(self) -> Dict[str, Any]:
        """Test environment variables and configuration."""
        log_test("Environment Configuration", "running")
        
        result = {
            "status": "success",
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check required environment variables
        required_env_vars = [
            "GCP_PROJECT_ID",
            "GCP_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS"
        ]
        
        for env_var in required_env_vars:
            value = os.getenv(env_var)
            if value:
                result["checks"][env_var] = "✓ Set"
                log_test(f"Environment variable {env_var}", "success")
            else:
                result["checks"][env_var] = "✗ Missing"
                result["warnings"].append(f"Environment variable {env_var} not set")
                log_test(f"Environment variable {env_var}", "warning")
        
        # Check service account key files
        key_files = [
            "./secrets/vertex-ai-main-key.json",
            "./secrets/model-training-key.json",
            "./secrets/data-pipeline-key.json"
        ]
        
        for key_file in key_files:
            if os.path.exists(key_file):
                result["checks"][f"Key file {key_file}"] = "✓ Exists"
                log_test(f"Service account key {key_file}", "success")
            else:
                result["checks"][f"Key file {key_file}"] = "✗ Missing"
                result["warnings"].append(f"Key file {key_file} not found")
                log_test(f"Service account key {key_file}", "warning")
        
        # Check configuration file
        config_file = "./config/vertex_ai_config.yaml"
        if os.path.exists(config_file):
            result["checks"]["Configuration file"] = "✓ Exists"
            log_test("Configuration file", "success")
        else:
            result["checks"]["Configuration file"] = "✗ Missing"
            result["warnings"].append("Configuration file not found")
            log_test("Configuration file", "warning")
        
        if result["warnings"]:
            result["status"] = "warning"
        
        return result
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test GCP authentication."""
        log_test("GCP Authentication", "running")
        
        result = {
            "status": "success",
            "authentication_method": "unknown",
            "project_id": None,
            "errors": []
        }
        
        try:
            # Try to import GCP libraries
            try:
                from google.auth import default
                from google.auth.exceptions import DefaultCredentialsError
                
                credentials, project = default()
                result["authentication_method"] = "Application Default Credentials"
                result["project_id"] = project
                
                log_test("GCP Authentication", "success")
                log_test(f"Project ID: {project}", "info")
                
            except DefaultCredentialsError as e:
                result["status"] = "error"
                result["errors"].append(f"Authentication failed: {e}")
                log_test("GCP Authentication", "error")
                
        except ImportError:
            result["status"] = "warning"
            result["errors"].append("Google Cloud libraries not available")
            log_test("GCP Authentication", "warning")
            
        return result
    
    async def test_client_initialization(self) -> Dict[str, Any]:
        """Test Vertex AI client initialization."""
        log_test("Vertex AI Client Initialization", "running")
        
        result = {
            "status": "success",
            "client_initialized": False,
            "available_models": 0,
            "errors": []
        }
        
        try:
            # Create config
            self.config = VertexAIConfig(
                project_id=os.getenv("GCP_PROJECT_ID", "darwin-biomaterials-scaffolds"),
                location=os.getenv("GCP_LOCATION", "us-central1"),
                service_account_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
            
            # Initialize client
            self.client = VertexAIClient(self.config)
            await self.client.initialize()
            
            result["client_initialized"] = self.client.is_initialized
            
            if self.client.is_initialized:
                # Get available models
                models_info = await self.client.get_available_models()
                result["available_models"] = models_info.get("total_models", 0)
                
                log_test("Client initialization", "success")
                log_test(f"Available models: {result['available_models']}", "info")
            else:
                result["status"] = "warning"
                result["errors"].append("Client initialized but not fully ready")
                log_test("Client initialization", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Client initialization", "error")
            
        return result
    
    async def test_model_access(self) -> Dict[str, Any]:
        """Test access to different Vertex AI models."""
        log_test("Model Access Testing", "running")
        
        result = {
            "status": "success",
            "models_tested": 0,
            "models_available": 0,
            "model_results": {},
            "errors": []
        }
        
        if not self.client or not self.client.is_initialized:
            result["status"] = "error"
            result["errors"].append("Client not initialized")
            return result
        
        # Test different models
        models_to_test = [
            VertexAIModel.GEMINI_1_5_PRO,
            VertexAIModel.GEMINI_1_5_FLASH,
            VertexAIModel.TEXT_BISON,
            VertexAIModel.MED_GEMINI_1_5_PRO
        ]
        
        for model in models_to_test:
            result["models_tested"] += 1
            
            try:
                # Simple test query
                test_prompt = f"Hello, this is a test of {model.value}. Please respond briefly."
                
                response = await self.client.generate_text(
                    prompt=test_prompt,
                    model=model,
                    max_tokens=50
                )
                
                if response and response.content:
                    result["models_available"] += 1
                    result["model_results"][model.value] = {
                        "status": "available",
                        "response_length": len(response.content),
                        "response_time_ms": response.response_time_ms
                    }
                    log_test(f"Model {model.value}", "success")
                else:
                    result["model_results"][model.value] = {
                        "status": "no_response"
                    }
                    log_test(f"Model {model.value}", "warning")
                    
            except Exception as e:
                result["model_results"][model.value] = {
                    "status": "error",
                    "error": str(e)
                }
                log_test(f"Model {model.value}", "error")
        
        if result["models_available"] == 0:
            result["status"] = "error"
            result["errors"].append("No models available")
        elif result["models_available"] < result["models_tested"]:
            result["status"] = "warning"
            
        log_test(f"Models available: {result['models_available']}/{result['models_tested']}", "info")
        
        return result
    
    async def test_storage_access(self) -> Dict[str, Any]:
        """Test Google Cloud Storage access."""
        log_test("Storage Access Testing", "running")
        
        result = {
            "status": "success",
            "buckets_tested": 0,
            "buckets_accessible": 0,
            "bucket_results": {},
            "errors": []
        }
        
        try:
            from google.cloud import storage
            
            # Initialize storage client
            client = storage.Client()
            
            # Test buckets
            project_id = os.getenv("GCP_PROJECT_ID", "darwin-biomaterials-scaffolds")
            buckets_to_test = [
                f"darwin-training-data-{project_id}",
                f"darwin-model-artifacts-{project_id}",
                f"darwin-experiment-logs-{project_id}"
            ]
            
            for bucket_name in buckets_to_test:
                result["buckets_tested"] += 1
                
                try:
                    bucket = client.bucket(bucket_name)
                    # Try to get bucket metadata
                    bucket.reload()
                    
                    result["buckets_accessible"] += 1
                    result["bucket_results"][bucket_name] = {
                        "status": "accessible",
                        "location": bucket.location,
                        "storage_class": bucket.storage_class
                    }
                    log_test(f"Bucket gs://{bucket_name}", "success")
                    
                except Exception as e:
                    result["bucket_results"][bucket_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    log_test(f"Bucket gs://{bucket_name}", "warning")
            
            if result["buckets_accessible"] == 0:
                result["status"] = "warning"
                result["errors"].append("No storage buckets accessible")
            
        except ImportError:
            result["status"] = "warning"
            result["errors"].append("Google Cloud Storage library not available")
            log_test("Storage libraries", "warning")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Storage access", "error")
            
        return result
    
    async def test_bigquery_access(self) -> Dict[str, Any]:
        """Test BigQuery access."""
        log_test("BigQuery Access Testing", "running")
        
        result = {
            "status": "success",
            "datasets_tested": 0,
            "datasets_accessible": 0,
            "dataset_results": {},
            "errors": []
        }
        
        try:
            from google.cloud import bigquery
            
            # Initialize BigQuery client
            client = bigquery.Client()
            
            # Test datasets
            project_id = os.getenv("GCP_PROJECT_ID", "darwin-biomaterials-scaffolds")
            datasets_to_test = [
                "darwin_research_insights",
                "darwin_performance_metrics",
                "darwin_scaffold_results"
            ]
            
            for dataset_id in datasets_to_test:
                result["datasets_tested"] += 1
                
                try:
                    dataset_ref = client.dataset(dataset_id, project=project_id)
                    dataset = client.get_dataset(dataset_ref)
                    
                    result["datasets_accessible"] += 1
                    result["dataset_results"][dataset_id] = {
                        "status": "accessible",
                        "location": dataset.location,
                        "created": dataset.created.isoformat() if dataset.created else None
                    }
                    log_test(f"Dataset {dataset_id}", "success")
                    
                except Exception as e:
                    result["dataset_results"][dataset_id] = {
                        "status": "error",
                        "error": str(e)
                    }
                    log_test(f"Dataset {dataset_id}", "warning")
            
            if result["datasets_accessible"] == 0:
                result["status"] = "warning"
                result["errors"].append("No BigQuery datasets accessible")
                
        except ImportError:
            result["status"] = "warning"
            result["errors"].append("Google Cloud BigQuery library not available")
            log_test("BigQuery libraries", "warning")
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("BigQuery access", "error")
            
        return result
    
    async def test_text_generation(self) -> Dict[str, Any]:
        """Test text generation capabilities."""
        log_test("Text Generation Testing", "running")
        
        result = {
            "status": "success",
            "prompts_tested": 0,
            "successful_generations": 0,
            "generation_results": {},
            "average_response_time": 0.0,
            "errors": []
        }
        
        if not self.client or not self.client.is_initialized:
            result["status"] = "error"
            result["errors"].append("Client not initialized")
            return result
        
        # Test prompts
        test_prompts = [
            {
                "name": "simple_greeting",
                "prompt": "Hello! How are you today?",
                "model": VertexAIModel.GEMINI_1_5_PRO
            },
            {
                "name": "biomaterials_question",
                "prompt": "What are the key factors to consider when designing scaffolds for tissue engineering?",
                "model": VertexAIModel.GEMINI_1_5_PRO
            },
            {
                "name": "technical_analysis",
                "prompt": "Explain the concept of spectral entropy in network analysis.",
                "model": VertexAIModel.GEMINI_1_5_PRO
            }
        ]
        
        total_response_time = 0.0
        
        for test_case in test_prompts:
            result["prompts_tested"] += 1
            
            try:
                response = await self.client.generate_text(
                    prompt=test_case["prompt"],
                    model=test_case["model"],
                    max_tokens=200
                )
                
                if response and response.content:
                    result["successful_generations"] += 1
                    total_response_time += response.response_time_ms
                    
                    result["generation_results"][test_case["name"]] = {
                        "status": "success",
                        "response_length": len(response.content),
                        "response_time_ms": response.response_time_ms,
                        "model_used": response.model_used
                    }
                    log_test(f"Generation '{test_case['name']}'", "success")
                else:
                    result["generation_results"][test_case["name"]] = {
                        "status": "no_response"
                    }
                    log_test(f"Generation '{test_case['name']}'", "warning")
                    
            except Exception as e:
                result["generation_results"][test_case["name"]] = {
                    "status": "error",
                    "error": str(e)
                }
                log_test(f"Generation '{test_case['name']}'", "error")
        
        if result["successful_generations"] > 0:
            result["average_response_time"] = total_response_time / result["successful_generations"]
        
        if result["successful_generations"] == 0:
            result["status"] = "error"
            result["errors"].append("No successful text generations")
        elif result["successful_generations"] < result["prompts_tested"]:
            result["status"] = "warning"
            
        log_test(f"Successful generations: {result['successful_generations']}/{result['prompts_tested']}", "info")
        log_test(f"Average response time: {result['average_response_time']:.1f}ms", "info")
        
        return result
    
    async def test_custom_models(self) -> Dict[str, Any]:
        """Test custom DARWIN model endpoints."""
        log_test("Custom Models Testing", "running")
        
        result = {
            "status": "success",
            "custom_models_tested": 0,
            "custom_models_available": 0,
            "model_results": {},
            "errors": []
        }
        
        if not self.client or not self.client.is_initialized:
            result["status"] = "error"
            result["errors"].append("Client not initialized")
            return result
        
        # Test custom DARWIN models
        custom_models = [
            VertexAIModel.DARWIN_BIOMATERIALS,
            VertexAIModel.DARWIN_MEDICAL,
            VertexAIModel.DARWIN_PHARMACOLOGY,
            VertexAIModel.DARWIN_QUANTUM
        ]
        
        for model in custom_models:
            result["custom_models_tested"] += 1
            
            try:
                # Test with domain-specific prompt
                test_prompt = f"Analyze this topic from your specialized {model.value.split('-')[1]} perspective: scaffold optimization."
                
                response = await self.client.generate_text(
                    prompt=test_prompt,
                    model=model,
                    max_tokens=100
                )
                
                if response and response.content:
                    result["custom_models_available"] += 1
                    result["model_results"][model.value] = {
                        "status": "available",
                        "specialization": model.value.split("-")[1],
                        "response_quality": "specialized" if model.value.split("-")[1] in response.content.lower() else "generic"
                    }
                    log_test(f"Custom model {model.value}", "success")
                else:
                    result["model_results"][model.value] = {
                        "status": "no_response"
                    }
                    log_test(f"Custom model {model.value}", "warning")
                    
            except Exception as e:
                result["model_results"][model.value] = {
                    "status": "not_deployed",
                    "error": str(e)
                }
                log_test(f"Custom model {model.value}", "warning")
        
        # Custom models might not be deployed yet - this is expected
        if result["custom_models_available"] == 0:
            result["status"] = "warning"
            result["errors"].append("Custom models not yet deployed (expected during initial setup)")
        
        log_test(f"Custom models available: {result['custom_models_available']}/{result['custom_models_tested']}", "info")
        
        return result
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        log_test("Performance Testing", "running")
        
        result = {
            "status": "success",
            "performance_metrics": {},
            "latency_test": {},
            "throughput_test": {},
            "errors": []
        }
        
        if not self.client:
            result["status"] = "error"
            result["errors"].append("Client not initialized")
            return result
        
        try:
            # Get client status
            status = await self.client.get_client_status()
            result["performance_metrics"] = status.get("performance_metrics", {})
            
            # Simple latency test
            start_time = asyncio.get_event_loop().time()
            
            response = await self.client.generate_text(
                prompt="What is 2+2?",
                model=VertexAIModel.GEMINI_1_5_FLASH,  # Use fastest model
                max_tokens=10
            )
            
            end_time = asyncio.get_event_loop().time()
            total_latency = (end_time - start_time) * 1000  # Convert to ms
            
            result["latency_test"] = {
                "total_latency_ms": total_latency,
                "model_response_time_ms": response.response_time_ms if response else 0,
                "network_overhead_ms": total_latency - (response.response_time_ms if response else 0)
            }
            
            log_test(f"Latency test: {total_latency:.1f}ms total", "info")
            log_test("Performance testing", "success")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Performance testing", "error")
        
        return result
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print_section("TEST SUMMARY")
        
        overall_status = results["overall_status"]
        passed = results["tests_passed"]
        warned = results["tests_warned"] 
        failed = results["tests_failed"]
        total = passed + warned + failed
        
        # Overall status
        if overall_status == "success":
            print(f"{Colors.GREEN}🎉 OVERALL STATUS: ALL TESTS PASSED! 🎉{Colors.NC}")
        elif overall_status == "warning":
            print(f"{Colors.YELLOW}⚠️  OVERALL STATUS: TESTS PASSED WITH WARNINGS{Colors.NC}")
        else:
            print(f"{Colors.RED}❌ OVERALL STATUS: SOME TESTS FAILED{Colors.NC}")
        
        # Test counts
        print(f"\n📊 Test Results:")
        print(f"   {Colors.GREEN}✅ Passed: {passed}{Colors.NC}")
        print(f"   {Colors.YELLOW}⚠️  Warnings: {warned}{Colors.NC}")
        print(f"   {Colors.RED}❌ Failed: {failed}{Colors.NC}")
        print(f"   📝 Total: {total}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if failed > 0:
            print(f"   {Colors.RED}• Fix failed tests before proceeding to production{Colors.NC}")
            print(f"   {Colors.RED}• Check service account permissions and API enablement{Colors.NC}")
        
        if warned > 0:
            print(f"   {Colors.YELLOW}• Review warnings - some features may be limited{Colors.NC}")
            print(f"   {Colors.YELLOW}• Custom models require deployment before full functionality{Colors.NC}")
        
        if overall_status == "success":
            print(f"   {Colors.GREEN}• Vertex AI setup is ready for production deployment! 🚀{Colors.NC}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"vertex_ai_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")

async def main():
    """Main test execution."""
    try:
        tester = VertexAITester()
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        if results["overall_status"] == "success":
            sys.exit(0)
        elif results["overall_status"] == "warning":
            sys.exit(1)  # Warnings
        else:
            sys.exit(2)  # Errors
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Test execution failed: {e}{Colors.NC}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())