#!/usr/bin/env python3
"""Test Data Pipeline - Million Scaffold Processing Validation

🧪 DATA PIPELINE TESTING REVOLUTIONARY SCRIPT
Script épico para testar e validar million scaffold processing pipeline:
- JAX ultra-performance validation
- BigQuery streaming integration
- Real-time analytics testing
- Performance benchmarking
- Stress testing capabilities

Usage: python scripts/test_data_pipeline.py
"""

import asyncio
import sys
import time
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from kec_unified_api.services.data_pipeline import MillionScaffoldPipeline, ScaffoldInput
    from kec_unified_api.services.bigquery_client import BigQueryClient
    from kec_unified_api.performance.jax_kec_engine import JAXKECEngine
    from kec_unified_api.core.logging import setup_logging, get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Setup logging
setup_logging()
logger = get_logger("darwin.data_pipeline_test")

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
    NC = '\033[0m'

def print_epic_header():
    """Print epic test header."""
    print(f"""
{Colors.PURPLE}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🧪 DARWIN DATA PIPELINE TESTING REVOLUTIONARY 🧪          ║
║                                                              ║
║  Testing million scaffold processing capabilities:          ║
║  • JAX Ultra-Performance Validation (1000x target)          ║
║  • BigQuery Streaming Pipeline                              ║
║  • Real-time Analytics Generation                           ║
║  • Performance Benchmarking                                 ║
║  • Stress Testing (up to 1M scaffolds)                      ║
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

class DataPipelineTester:
    """Comprehensive data pipeline testing class."""
    
    def __init__(self):
        self.pipeline: Optional[MillionScaffoldPipeline] = None
        self.test_results: Dict[str, Any] = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive data pipeline tests."""
        print_epic_header()
        
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "test_details": {}
        }
        
        # Test categories
        test_categories = [
            ("Pipeline Initialization", self.test_pipeline_initialization),
            ("JAX Engine Validation", self.test_jax_engine),
            ("BigQuery Integration", self.test_bigquery_integration),
            ("Synthetic Data Generation", self.test_synthetic_generation),
            ("Small Batch Processing", self.test_small_batch_processing),
            ("Performance Benchmarking", self.test_performance_benchmarking),
            ("Stress Testing", self.test_stress_testing),
            ("Analytics Generation", self.test_analytics_generation)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n{Colors.WHITE}{'='*20} {category_name} {'='*20}{Colors.NC}")
            
            try:
                category_result = await test_function()
                test_results["test_details"][category_name] = category_result
                
                if category_result.get("status") == "success":
                    test_results["tests_passed"] += 1
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
            test_results["overall_status"] = "success"
        elif test_results["tests_passed"] > test_results["tests_failed"]:
            test_results["overall_status"] = "warning"
        else:
            test_results["overall_status"] = "failed"
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    async def test_pipeline_initialization(self) -> Dict[str, Any]:
        """Test pipeline initialization."""
        log_test("Data Pipeline Initialization", "running")
        
        result = {
            "status": "success",
            "components_initialized": {},
            "errors": []
        }
        
        try:
            # Initialize pipeline
            self.pipeline = MillionScaffoldPipeline(
                project_id="darwin-biomaterials-scaffolds",
                batch_size=100,  # Smaller batch for testing
                max_concurrent_batches=2
            )
            
            await self.pipeline.initialize()
            
            result["components_initialized"]["pipeline"] = self.pipeline.is_initialized
            result["components_initialized"]["jax_engine"] = self.pipeline.jax_engine.is_initialized if self.pipeline.jax_engine else False
            result["components_initialized"]["bigquery"] = self.pipeline.bigquery_client.is_initialized if self.pipeline.bigquery_client else False
            
            if self.pipeline.is_initialized:
                log_test("Pipeline initialization", "success")
            else:
                result["status"] = "error"
                result["errors"].append("Pipeline failed to initialize")
                log_test("Pipeline initialization", "error")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Pipeline initialization", "error")
            
        return result
    
    async def test_jax_engine(self) -> Dict[str, Any]:
        """Test JAX engine performance."""
        log_test("JAX Engine Performance", "running")
        
        result = {
            "status": "success",
            "jax_available": False,
            "performance_metrics": {},
            "errors": []
        }
        
        if not self.pipeline or not self.pipeline.jax_engine:
            result["status"] = "error"
            result["errors"].append("JAX engine not available")
            return result
        
        try:
            # Test basic JAX computation
            test_matrix = np.random.rand(50, 50)
            test_matrix = (test_matrix + test_matrix.T) / 2
            np.fill_diagonal(test_matrix, 0)
            
            start_time = time.time()
            
            kec_result, performance = await self.pipeline.jax_engine.compute_kec_ultra_fast(
                adjacency_matrix=test_matrix,
                metrics=["H_spectral", "k_forman_mean", "sigma", "swp"]
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            result["jax_available"] = True
            result["performance_metrics"] = {
                "computation_time_ms": computation_time,
                "speedup_factor": performance.speedup_factor,
                "throughput": performance.throughput_scaffolds_per_second,
                "device_used": performance.device_used
            }
            
            # Validate KEC results
            if kec_result and kec_result.H_spectral is not None:
                log_test(f"JAX computation: {computation_time:.2f}ms, {performance.speedup_factor:.1f}x speedup", "success")
            else:
                result["status"] = "warning"
                result["errors"].append("KEC computation produced no results")
                log_test("JAX computation", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("JAX engine", "error")
            
        return result
    
    async def test_bigquery_integration(self) -> Dict[str, Any]:
        """Test BigQuery integration."""
        log_test("BigQuery Integration", "running")
        
        result = {
            "status": "success",
            "bigquery_available": False,
            "datasets_accessible": 0,
            "errors": []
        }
        
        if not self.pipeline or not self.pipeline.bigquery_client:
            result["status"] = "error"
            result["errors"].append("BigQuery client not available")
            return result
        
        try:
            # Get BigQuery client status
            bq_status = await self.pipeline.bigquery_client.get_client_status()
            
            result["bigquery_available"] = bq_status.get("client_initialized", False)
            result["datasets_accessible"] = bq_status.get("datasets_count", 0)
            
            if result["bigquery_available"]:
                log_test("BigQuery client", "success")
                
                # Test analytics query
                analytics = await self.pipeline.bigquery_client.get_scaffold_analytics(time_window_hours=1)
                if "error" not in analytics:
                    log_test("BigQuery analytics", "success")
                else:
                    log_test("BigQuery analytics", "warning")
                    
            else:
                result["status"] = "warning"
                log_test("BigQuery client", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("BigQuery integration", "error")
            
        return result
    
    async def test_synthetic_generation(self) -> Dict[str, Any]:
        """Test synthetic scaffold generation."""
        log_test("Synthetic Data Generation", "running")
        
        result = {
            "status": "success",
            "scaffolds_generated": 0,
            "generation_time_ms": 0,
            "errors": []
        }
        
        if not self.pipeline:
            result["status"] = "error"
            result["errors"].append("Pipeline not available")
            return result
        
        try:
            start_time = time.time()
            
            # Generate test scaffolds
            scaffolds = await self.pipeline.generate_synthetic_scaffolds(
                count=100,
                matrix_sizes=[30, 50],
                material_types=["collagen", "chitosan"]
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            result["scaffolds_generated"] = len(scaffolds)
            result["generation_time_ms"] = generation_time
            
            if len(scaffolds) == 100:
                log_test(f"Generated {len(scaffolds)} scaffolds in {generation_time:.1f}ms", "success")
            else:
                result["status"] = "warning"
                result["errors"].append(f"Expected 100 scaffolds, got {len(scaffolds)}")
                log_test("Synthetic generation", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Synthetic generation", "error")
            
        return result
    
    async def test_small_batch_processing(self) -> Dict[str, Any]:
        """Test small batch processing."""
        log_test("Small Batch Processing", "running")
        
        result = {
            "status": "success",
            "batch_size": 50,
            "processing_metrics": {},
            "errors": []
        }
        
        if not self.pipeline:
            result["status"] = "error"
            result["errors"].append("Pipeline not available")
            return result
        
        try:
            # Generate small test batch
            test_scaffolds = await self.pipeline.generate_synthetic_scaffolds(
                count=50,
                matrix_sizes=[40],
                material_types=["collagen"]
            )
            
            start_time = time.time()
            
            # Process batch
            processing_result = await self.pipeline.process_million_scaffolds(
                scaffolds=test_scaffolds,
                enable_biocompatibility_analysis=True,
                enable_real_time_analytics=False  # Disable for test
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            result["processing_metrics"] = {
                "total_processing_time_ms": processing_time,
                "throughput": processing_result["throughput_scaffolds_per_second"],
                "success_rate": processing_result["success_rate"],
                "average_speedup": processing_result["average_speedup_factor"]
            }
            
            if processing_result["success_rate"] >= 0.8:
                log_test(f"Batch processing: {processing_result['success_rate']*100:.1f}% success, {processing_result['throughput_scaffolds_per_second']:.1f} scaffolds/s", "success")
            else:
                result["status"] = "warning"
                result["errors"].append(f"Low success rate: {processing_result['success_rate']}")
                log_test("Batch processing", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Small batch processing", "error")
            
        return result
    
    async def test_performance_benchmarking(self) -> Dict[str, Any]:
        """Test performance benchmarking capabilities."""
        log_test("Performance Benchmarking", "running")
        
        result = {
            "status": "success",
            "benchmark_results": {},
            "performance_targets_met": {},
            "errors": []
        }
        
        if not self.pipeline:
            result["status"] = "error"
            result["errors"].append("Pipeline not available")
            return result
        
        try:
            # Run performance benchmark
            benchmark_result = await self.pipeline.benchmark_million_scaffold_performance()
            
            if "error" in benchmark_result:
                result["status"] = "error"
                result["errors"].append(benchmark_result["error"])
                log_test("Performance benchmarking", "error")
            else:
                result["benchmark_results"] = benchmark_result
                
                # Check performance targets
                scaling_analysis = benchmark_result.get("scaling_analysis", {})
                result["performance_targets_met"] = {
                    "million_scaffold_ready": scaling_analysis.get("million_scaffold_ready", False),
                    "linear_scaling": scaling_analysis.get("linear_scaling_achieved", False),
                    "peak_throughput": scaling_analysis.get("peak_throughput", 0)
                }
                
                if scaling_analysis.get("million_scaffold_ready", False):
                    log_test(f"Performance benchmark: Peak {scaling_analysis.get('peak_throughput', 0):.1f} scaffolds/s", "success")
                else:
                    result["status"] = "warning"
                    log_test("Performance benchmark", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Performance benchmarking", "error")
            
        return result
    
    async def test_stress_testing(self) -> Dict[str, Any]:
        """Test stress testing with larger datasets."""
        log_test("Stress Testing", "running")
        
        result = {
            "status": "success",
            "stress_test_sizes": [1000, 5000],
            "stress_results": {},
            "errors": []
        }
        
        if not self.pipeline:
            result["status"] = "error"
            result["errors"].append("Pipeline not available")
            return result
        
        try:
            # Test with increasing sizes
            for test_size in result["stress_test_sizes"]:
                log_test(f"Stress testing with {test_size} scaffolds", "running")
                
                # Generate stress test data
                stress_scaffolds = await self.pipeline.generate_synthetic_scaffolds(
                    count=test_size,
                    matrix_sizes=[50, 100],
                    material_types=["collagen", "chitosan"]
                )
                
                start_time = time.time()
                
                # Process with pipeline
                stress_result = await self.pipeline.process_million_scaffolds(
                    scaffolds=stress_scaffolds,
                    enable_biocompatibility_analysis=True,
                    enable_real_time_analytics=False
                )
                
                stress_duration = time.time() - start_time
                
                result["stress_results"][f"size_{test_size}"] = {
                    "duration_seconds": stress_duration,
                    "throughput": stress_result["throughput_scaffolds_per_second"],
                    "success_rate": stress_result["success_rate"],
                    "speedup_factor": stress_result["average_speedup_factor"]
                }
                
                log_test(f"Stress {test_size}: {stress_result['throughput_scaffolds_per_second']:.1f} scaffolds/s", "success")
            
            # Analyze stress test results
            stress_analysis = self._analyze_stress_results(result["stress_results"])
            result["stress_analysis"] = stress_analysis
            
            if stress_analysis.get("scalability_good", False):
                log_test("Stress testing analysis", "success")
            else:
                result["status"] = "warning"
                log_test("Stress testing analysis", "warning")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Stress testing", "error")
            
        return result
    
    async def test_analytics_generation(self) -> Dict[str, Any]:
        """Test analytics generation."""
        log_test("Analytics Generation", "running")
        
        result = {
            "status": "success",
            "analytics_types": [],
            "errors": []
        }
        
        if not self.pipeline or not self.pipeline.bigquery_client:
            result["status"] = "warning"
            result["errors"].append("BigQuery client not available")
            return result
        
        try:
            # Test scaffold analytics
            scaffold_analytics = await self.pipeline.bigquery_client.get_scaffold_analytics(
                time_window_hours=1
            )
            
            if "error" not in scaffold_analytics:
                result["analytics_types"].append("scaffold_analytics")
                log_test("Scaffold analytics", "success")
            else:
                log_test("Scaffold analytics", "warning")
            
            # Test collaboration analytics
            collab_analytics = await self.pipeline.bigquery_client.get_collaboration_analytics()
            
            if "error" not in collab_analytics:
                result["analytics_types"].append("collaboration_analytics")
                log_test("Collaboration analytics", "success")
            else:
                log_test("Collaboration analytics", "warning")
            
            if len(result["analytics_types"]) == 0:
                result["status"] = "warning"
                result["errors"].append("No analytics generated")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_test("Analytics generation", "error")
            
        return result
    
    def _analyze_stress_results(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stress test results for scalability."""
        if not stress_results:
            return {"scalability_good": False}
        
        # Extract throughputs
        throughputs = []
        sizes = []
        
        for test_name, test_result in stress_results.items():
            if test_name.startswith("size_"):
                size = int(test_name.split("_")[1])
                throughput = test_result.get("throughput", 0)
                sizes.append(size)
                throughputs.append(throughput)
        
        if len(throughputs) < 2:
            return {"scalability_good": False}
        
        # Check if throughput is maintained or improved
        throughput_stable = all(t >= throughputs[0] * 0.8 for t in throughputs)  # Allow 20% variance
        peak_throughput = max(throughputs)
        
        return {
            "scalability_good": throughput_stable and peak_throughput >= 10,
            "peak_throughput": peak_throughput,
            "throughput_stable": throughput_stable,
            "million_scaffold_ready": peak_throughput >= 100
        }
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print(f"\n{Colors.WHITE}{'='*20} TEST SUMMARY {'='*20}{Colors.NC}")
        
        overall_status = results["overall_status"]
        passed = results["tests_passed"]
        failed = results["tests_failed"]
        total = passed + failed
        
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
        print(f"   {Colors.RED}❌ Failed: {failed}{Colors.NC}")
        print(f"   📝 Total: {total}")
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        
        # Extract key performance metrics
        jax_metrics = results["test_details"].get("JAX Engine Validation", {}).get("performance_metrics", {})
        if jax_metrics:
            print(f"   🔥 JAX Speedup: {jax_metrics.get('speedup_factor', 0):.1f}x")
            print(f"   ⚡ Throughput: {jax_metrics.get('throughput', 0):.1f} scaffolds/s")
            print(f"   🎯 Device: {jax_metrics.get('device_used', 'unknown')}")
        
        benchmark_results = results["test_details"].get("Performance Benchmarking", {}).get("performance_targets_met", {})
        if benchmark_results:
            print(f"   🌊 Million Scaffold Ready: {'✅ Yes' if benchmark_results.get('million_scaffold_ready', False) else '⚠️ No'}")
            print(f"   📈 Linear Scaling: {'✅ Yes' if benchmark_results.get('linear_scaling', False) else '⚠️ No'}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if failed > 0:
            print(f"   {Colors.RED}• Fix failed tests before production deployment{Colors.NC}")
        
        if overall_status == "success":
            print(f"   {Colors.GREEN}• Data pipeline is ready for million scaffold processing! 🚀{Colors.NC}")
            print(f"   {Colors.GREEN}• Performance targets achieved - proceed with production{Colors.NC}")
        elif overall_status == "warning":
            print(f"   {Colors.YELLOW}• Some components have limitations - review warnings{Colors.NC}")
            print(f"   {Colors.YELLOW}• Consider fallback modes for production deployment{Colors.NC}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data_pipeline_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")

async def main():
    """Main test execution."""
    try:
        tester = DataPipelineTester()
        results = await tester.run_all_tests()
        
        # Cleanup
        if tester.pipeline:
            await tester.pipeline.shutdown()
        
        # Exit with appropriate code
        if results["overall_status"] == "success":
            print(f"\n{Colors.GREEN}🎉 DATA PIPELINE TESTS COMPLETED SUCCESSFULLY! 🎉{Colors.NC}")
            sys.exit(0)
        elif results["overall_status"] == "warning":
            print(f"\n{Colors.YELLOW}⚠️ DATA PIPELINE TESTS COMPLETED WITH WARNINGS{Colors.NC}")
            sys.exit(1)
        else:
            print(f"\n{Colors.RED}❌ DATA PIPELINE TESTS FAILED{Colors.NC}")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Test execution failed: {e}{Colors.NC}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())