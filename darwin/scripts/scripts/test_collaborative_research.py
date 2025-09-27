#!/usr/bin/env python3
"""Test Collaborative Research - AutoGen GroupChat Validation

🤝 COLLABORATIVE RESEARCH TESTING REVOLUTIONARY SCRIPT
Script épico para testar e validar AutoGen GroupChat e collaborative research:
- Multi-agent collaboration scenarios
- GroupChat manager orchestration
- Cross-domain research analysis
- Interdisciplinary insight generation
- Revolutionary research discoveries

Usage: python scripts/test_collaborative_research.py
"""

import asyncio
import sys
import time
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from kec_unified_api.ai_agents.research_team import ResearchTeamCoordinator
    from kec_unified_api.ai_agents.agent_models import (
        CollaborativeResearchRequest,
        CrossDomainRequest,
        AgentSpecialization,
        ResearchPriority
    )
    from kec_unified_api.core.logging import setup_logging, get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and dependencies are installed")
    sys.exit(1)

# Setup logging
setup_logging()
logger = get_logger("darwin.collaborative_research_test")

# Colors for epic console output
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

def print_revolutionary_header():
    """Print revolutionary collaboration test header."""
    print(f"""
{Colors.PURPLE}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🤝 DARWIN COLLABORATIVE RESEARCH TESTING REVOLUTIONARY 🤝 ║
║                                                              ║
║  Testing AutoGen GroupChat Multi-Agent Collaboration:       ║
║  • Multi-Agent Research Scenarios                           ║
║  • GroupChat Manager Orchestration                          ║
║  • Cross-Domain Analysis Validation                         ║
║  • Interdisciplinary Insight Generation                     ║
║  • Revolutionary Research Discovery Testing                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.NC}
""")

def log_collaboration_test(test_name: str, status: str = "running"):
    """Log collaboration test status with colors."""
    if status == "running":
        print(f"{Colors.BLUE}🤝 [COLLABORATION]{Colors.NC} {test_name}...")
    elif status == "success":
        print(f"{Colors.GREEN}✅ [SUCCESS]{Colors.NC} {test_name}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️  [WARNING]{Colors.NC} {test_name}")
    elif status == "error":
        print(f"{Colors.RED}❌ [ERROR]{Colors.NC} {test_name}")
    elif status == "insight":
        print(f"{Colors.CYAN}💡 [INSIGHT]{Colors.NC} {test_name}")

class CollaborativeResearchTester:
    """Epic collaborative research testing class."""
    
    def __init__(self):
        self.research_team: Optional[ResearchTeamCoordinator] = None
        self.collaboration_scenarios = self._create_collaboration_scenarios()
        
    def _create_collaboration_scenarios(self) -> List[Dict[str, Any]]:
        """Create epic collaboration test scenarios."""
        return [
            {
                "name": "Quantum-Enhanced Biomaterials Research",
                "description": "Cross-domain analysis of quantum effects in biomaterial scaffolds",
                "request": CollaborativeResearchRequest(
                    research_question="How can quantum coherence effects in biomaterial scaffolds enhance tissue engineering outcomes?",
                    context="Investigating quantum biology applications in scaffold design for neural tissue engineering",
                    target_specializations=[
                        AgentSpecialization.BIOMATERIALS,
                        AgentSpecialization.QUANTUM_MECHANICS,
                        AgentSpecialization.CLINICAL_PSYCHIATRY
                    ],
                    max_agents=3,
                    max_rounds=8,
                    priority=ResearchPriority.HIGH,
                    include_synthesis=True
                ),
                "expected_insights": 3,
                "expected_domains": ["biomaterials", "quantum_mechanics", "clinical"]
            },
            {
                "name": "Precision Medicine Scaffold Optimization",
                "description": "Medical and pharmacological perspectives on personalized scaffold design",
                "request": CollaborativeResearchRequest(
                    research_question="Design personalized scaffold optimization strategy considering patient pharmacogenomics and clinical factors.",
                    context="Developing precision medicine approach for scaffold-based drug delivery systems",
                    target_specializations=[
                        AgentSpecialization.BIOMATERIALS,
                        AgentSpecialization.PHARMACOLOGY,
                        AgentSpecialization.CLINICAL_PSYCHIATRY,
                        AgentSpecialization.MATHEMATICS
                    ],
                    max_agents=4,
                    max_rounds=10,
                    priority=ResearchPriority.HIGH,
                    include_synthesis=True
                ),
                "expected_insights": 4,
                "expected_domains": ["biomaterials", "pharmacology", "clinical", "mathematics"]
            },
            {
                "name": "Philosophical Foundations of AI-Assisted Research",
                "description": "Philosophical analysis of consciousness in AI research collaboration",
                "request": CollaborativeResearchRequest(
                    research_question="What are the philosophical implications of AI consciousness in collaborative scientific research?",
                    context="Exploring epistemological foundations of AI-assisted interdisciplinary research",
                    target_specializations=[
                        AgentSpecialization.PHILOSOPHY,
                        AgentSpecialization.LITERATURE,
                        AgentSpecialization.SYNTHESIS
                    ],
                    max_agents=3,
                    max_rounds=6,
                    priority=ResearchPriority.MEDIUM,
                    include_synthesis=True
                ),
                "expected_insights": 3,
                "expected_domains": ["philosophy", "literature", "synthesis"]
            },
            {
                "name": "Mathematical Optimization of KEC Metrics",
                "description": "Mathematical and computational approach to KEC optimization",
                "request": CollaborativeResearchRequest(
                    research_question="Develop mathematical framework for optimizing KEC metrics using spectral graph theory and quantum computing.",
                    context="Creating comprehensive mathematical foundation for scaffold optimization",
                    target_specializations=[
                        AgentSpecialization.MATHEMATICS,
                        AgentSpecialization.QUANTUM_MECHANICS,
                        AgentSpecialization.BIOMATERIALS
                    ],
                    max_agents=3,
                    max_rounds=7,
                    priority=ResearchPriority.HIGH,
                    include_synthesis=True
                ),
                "expected_insights": 3,
                "expected_domains": ["mathematics", "quantum_mechanics", "biomaterials"]
            },
            {
                "name": "Comprehensive Interdisciplinary Analysis",
                "description": "Full team collaboration on complex research question",
                "request": CollaborativeResearchRequest(
                    research_question="Revolutionize tissue engineering through quantum-enhanced, AI-optimized, precision medicine scaffolds.",
                    context="Ultimate interdisciplinary challenge requiring all specialist expertise",
                    target_specializations=None,  # All agents
                    max_agents=8,  # Full team
                    max_rounds=15,
                    priority=ResearchPriority.CRITICAL,
                    include_synthesis=True
                ),
                "expected_insights": 8,
                "expected_domains": ["all_domains"]
            }
        ]
    
    async def run_all_collaboration_tests(self) -> Dict[str, Any]:
        """Run comprehensive collaborative research tests."""
        print_revolutionary_header()
        
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "scenarios_tested": 0,
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "collaboration_metrics": {},
            "scenario_results": {}
        }
        
        try:
            # Initialize research team
            await self._initialize_research_team()
            
            if not self.research_team or not self.research_team.is_initialized:
                test_results["overall_status"] = "failed"
                test_results["error"] = "Research team initialization failed"
                return test_results
            
            # Test each collaboration scenario
            for i, scenario in enumerate(self.collaboration_scenarios):
                scenario_name = scenario["name"]
                
                print(f"\n{Colors.WHITE}{'='*15} Scenario {i+1}: {scenario_name} {'='*15}{Colors.NC}")
                print(f"{Colors.CYAN}📋 Description: {scenario['description']}{Colors.NC}")
                
                test_results["scenarios_tested"] += 1
                
                try:
                    scenario_result = await self._test_collaboration_scenario(scenario)
                    test_results["scenario_results"][scenario_name] = scenario_result
                    
                    if scenario_result.get("status") == "success":
                        test_results["scenarios_passed"] += 1
                        log_collaboration_test(f"Scenario: {scenario_name}", "success")
                    else:
                        test_results["scenarios_failed"] += 1
                        log_collaboration_test(f"Scenario: {scenario_name}", "warning")
                        
                except Exception as e:
                    logger.error(f"Scenario {scenario_name} failed: {e}")
                    test_results["scenario_results"][scenario_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    test_results["scenarios_failed"] += 1
                    log_collaboration_test(f"Scenario: {scenario_name}", "error")
            
            # Test cross-domain analysis
            cross_domain_result = await self._test_cross_domain_analysis()
            test_results["cross_domain_analysis"] = cross_domain_result
            
            # Calculate collaboration metrics
            collaboration_metrics = await self._calculate_collaboration_metrics(test_results)
            test_results["collaboration_metrics"] = collaboration_metrics
            
            # Determine overall status
            if test_results["scenarios_failed"] == 0:
                test_results["overall_status"] = "success"
            elif test_results["scenarios_passed"] > test_results["scenarios_failed"]:
                test_results["overall_status"] = "warning"
            else:
                test_results["overall_status"] = "failed"
            
            # Print comprehensive summary
            self._print_collaboration_summary(test_results)
            
        except Exception as e:
            logger.error(f"Collaborative research testing failed: {e}")
            test_results["overall_status"] = "failed"
            test_results["error"] = str(e)
        
        finally:
            # Cleanup
            if self.research_team:
                await self.research_team.shutdown()
        
        return test_results
    
    async def _initialize_research_team(self):
        """Initialize AutoGen research team."""
        try:
            log_collaboration_test("Research Team Initialization", "running")
            
            self.research_team = ResearchTeamCoordinator()
            await self.research_team.initialize()
            
            if self.research_team.is_initialized:
                log_collaboration_test("Research Team Initialization", "success")
                
                # Get team status
                team_status = await self.research_team.get_team_status()
                print(f"{Colors.CYAN}🤖 Team Status: {team_status.total_agents} agents, {team_status.active_agents} active{Colors.NC}")
                
            else:
                log_collaboration_test("Research Team Initialization", "error")
                raise RuntimeError("Research team failed to initialize")
                
        except Exception as e:
            logger.error(f"Research team initialization failed: {e}")
            log_collaboration_test("Research Team Initialization", "error")
            raise
    
    async def _test_collaboration_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual collaboration scenario."""
        scenario_name = scenario["name"]
        request = scenario["request"]
        
        result = {
            "status": "success",
            "scenario_name": scenario_name,
            "collaboration_metrics": {},
            "quality_assessment": {},
            "insights_analysis": {},
            "errors": []
        }
        
        try:
            log_collaboration_test(f"Executing: {scenario_name}", "running")
            
            # Measure collaboration time
            start_time = time.time()
            
            # Execute collaborative research
            collaboration_response = await self.research_team.collaborative_research(request)
            
            collaboration_time = (time.time() - start_time) * 1000  # ms
            
            # Analyze collaboration response
            result["collaboration_metrics"] = {
                "response_time_ms": collaboration_time,
                "participating_agents": len(collaboration_response.participating_agents),
                "insights_generated": len(collaboration_response.insights),
                "confidence_score": collaboration_response.confidence_score,
                "synthesis_available": bool(collaboration_response.synthesis),
                "execution_time_seconds": collaboration_response.execution_time_seconds
            }
            
            # Quality assessment
            quality_score = self._assess_collaboration_quality(collaboration_response, scenario)
            result["quality_assessment"] = {
                "overall_quality": quality_score,
                "domain_coverage": self._analyze_domain_coverage(collaboration_response, scenario),
                "insight_depth": self._analyze_insight_depth(collaboration_response),
                "interdisciplinary_connections": len(collaboration_response.insights) >= scenario["expected_insights"]
            }
            
            # Insights analysis
            result["insights_analysis"] = {
                "total_insights": len(collaboration_response.insights),
                "average_confidence": sum(i.confidence for i in collaboration_response.insights) / len(collaboration_response.insights) if collaboration_response.insights else 0,
                "insight_types": [i.type.value for i in collaboration_response.insights],
                "agent_contributions": {agent: len([i for i in collaboration_response.insights if agent.lower() in i.agent_specialization.value.lower()]) for agent in collaboration_response.participating_agents}
            }
            
            # Determine success criteria
            success_criteria = [
                collaboration_response.status == "completed",
                len(collaboration_response.insights) >= scenario["expected_insights"] * 0.7,  # 70% threshold
                collaboration_response.confidence_score >= 0.6,
                quality_score >= 0.7
            ]
            
            if all(success_criteria):
                result["status"] = "success"
                log_collaboration_test(f"Quality Score: {quality_score:.2f}", "success")
            elif sum(success_criteria) >= len(success_criteria) * 0.7:  # 70% criteria met
                result["status"] = "warning"
                log_collaboration_test(f"Quality Score: {quality_score:.2f}", "warning")
            else:
                result["status"] = "error"
                result["errors"].append(f"Quality criteria not met: {sum(success_criteria)}/{len(success_criteria)}")
                log_collaboration_test(f"Quality Score: {quality_score:.2f}", "error")
            
            # Log insights preview
            if collaboration_response.insights:
                print(f"{Colors.CYAN}💡 Sample Insight: {collaboration_response.insights[0].content[:100]}...{Colors.NC}")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_collaboration_test(f"Scenario: {scenario_name}", "error")
        
        return result
    
    async def _test_cross_domain_analysis(self) -> Dict[str, Any]:
        """Test cross-domain analysis functionality."""
        log_collaboration_test("Cross-Domain Analysis", "running")
        
        result = {
            "status": "success",
            "cross_domain_tests": {},
            "errors": []
        }
        
        try:
            # Test cross-domain scenarios
            cross_domain_scenarios = [
                {
                    "name": "Biomaterials-Quantum Cross-Analysis",
                    "request": CrossDomainRequest(
                        research_topic="Quantum effects in biomaterial scaffold optimization",
                        primary_domain=AgentSpecialization.BIOMATERIALS,
                        secondary_domains=[AgentSpecialization.QUANTUM_MECHANICS],
                        specific_question="How can quantum tunneling improve drug delivery in scaffolds?"
                    )
                },
                {
                    "name": "Medical-Pharmacology Cross-Analysis", 
                    "request": CrossDomainRequest(
                        research_topic="Precision medicine approaches to scaffold-based drug delivery",
                        primary_domain=AgentSpecialization.CLINICAL_PSYCHIATRY,
                        secondary_domains=[AgentSpecialization.PHARMACOLOGY, AgentSpecialization.BIOMATERIALS]
                    )
                }
            ]
            
            for scenario in cross_domain_scenarios:
                scenario_name = scenario["name"]
                log_collaboration_test(f"Cross-Domain: {scenario_name}", "running")
                
                try:
                    start_time = time.time()
                    
                    cross_domain_response = await self.research_team.cross_domain_analysis(scenario["request"])
                    
                    analysis_time = (time.time() - start_time) * 1000
                    
                    # Analyze cross-domain response
                    cross_domain_metrics = {
                        "analysis_time_ms": analysis_time,
                        "cross_domain_insights": len(cross_domain_response.cross_domain_insights),
                        "domain_connections": len(cross_domain_response.domain_connections.get("novel_connections", [])),
                        "novel_perspectives": len(cross_domain_response.novel_perspectives),
                        "interdisciplinary_opportunities": len(cross_domain_response.interdisciplinary_opportunities),
                        "confidence_by_domain": cross_domain_response.confidence_by_domain
                    }
                    
                    result["cross_domain_tests"][scenario_name] = {
                        "status": "success",
                        "metrics": cross_domain_metrics
                    }
                    
                    log_collaboration_test(f"Cross-Domain: {scenario_name}", "success")
                    
                except Exception as e:
                    result["cross_domain_tests"][scenario_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    log_collaboration_test(f"Cross-Domain: {scenario_name}", "error")
            
            # Overall cross-domain status
            successful_tests = sum(1 for test in result["cross_domain_tests"].values() if test["status"] == "success")
            total_tests = len(result["cross_domain_tests"])
            
            if successful_tests == total_tests:
                log_collaboration_test("Cross-Domain Analysis Overall", "success")
            elif successful_tests > 0:
                result["status"] = "warning"
                log_collaboration_test("Cross-Domain Analysis Overall", "warning")
            else:
                result["status"] = "error"
                log_collaboration_test("Cross-Domain Analysis Overall", "error")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_collaboration_test("Cross-Domain Analysis", "error")
        
        return result
    
    def _assess_collaboration_quality(self, response: Any, scenario: Dict[str, Any]) -> float:
        """Assess quality of collaborative research response."""
        try:
            quality_factors = []
            
            # Factor 1: Number of insights (30%)
            expected_insights = scenario["expected_insights"]
            actual_insights = len(response.insights) if response.insights else 0
            insights_score = min(1.0, actual_insights / expected_insights) if expected_insights > 0 else 0
            quality_factors.append(insights_score * 0.3)
            
            # Factor 2: Agent participation (25%)
            expected_agents = len(scenario["request"].target_specializations) if scenario["request"].target_specializations else scenario["request"].max_agents
            actual_agents = len(response.participating_agents) if response.participating_agents else 0
            participation_score = min(1.0, actual_agents / expected_agents) if expected_agents > 0 else 0
            quality_factors.append(participation_score * 0.25)
            
            # Factor 3: Confidence score (20%)
            confidence_score = response.confidence_score if hasattr(response, 'confidence_score') else 0.5
            quality_factors.append(confidence_score * 0.2)
            
            # Factor 4: Synthesis quality (15%)
            synthesis_score = 1.0 if response.synthesis and len(response.synthesis) > 100 else 0.5
            quality_factors.append(synthesis_score * 0.15)
            
            # Factor 5: Response completeness (10%)
            completeness_score = 1.0 if response.status == "completed" else 0.3
            quality_factors.append(completeness_score * 0.1)
            
            return sum(quality_factors)
            
        except Exception as e:
            logger.warning(f"Quality assessment error: {e}")
            return 0.5
    
    def _analyze_domain_coverage(self, response: Any, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain coverage in collaboration response."""
        try:
            expected_domains = scenario["expected_domains"]
            
            if "all_domains" in expected_domains:
                # For full team scenarios
                covered_domains = set()
                for insight in response.insights:
                    covered_domains.add(insight.agent_specialization.value)
                
                return {
                    "expected_domains": "all_available",
                    "covered_domains": list(covered_domains),
                    "coverage_count": len(covered_domains),
                    "coverage_complete": len(covered_domains) >= 6  # At least 6 different domains
                }
            else:
                # For specific domain scenarios
                covered_domains = set()
                for insight in response.insights:
                    domain = insight.agent_specialization.value
                    if any(exp_domain in domain for exp_domain in expected_domains):
                        covered_domains.add(domain)
                
                return {
                    "expected_domains": expected_domains,
                    "covered_domains": list(covered_domains),
                    "coverage_percentage": len(covered_domains) / len(expected_domains) if expected_domains else 0,
                    "coverage_complete": len(covered_domains) >= len(expected_domains) * 0.8  # 80% coverage
                }
                
        except Exception as e:
            logger.warning(f"Domain coverage analysis error: {e}")
            return {"coverage_complete": False}
    
    def _analyze_insight_depth(self, response: Any) -> Dict[str, Any]:
        """Analyze depth and quality of insights."""
        try:
            if not response.insights:
                return {"average_depth": 0, "depth_quality": "poor"}
            
            # Analyze insight lengths and complexity
            insight_lengths = [len(insight.content) for insight in response.insights]
            avg_length = sum(insight_lengths) / len(insight_lengths)
            
            # Analyze confidence levels
            confidence_levels = [insight.confidence for insight in response.insights]
            avg_confidence = sum(confidence_levels) / len(confidence_levels)
            
            # Determine depth quality
            if avg_length >= 200 and avg_confidence >= 0.8:
                depth_quality = "excellent"
            elif avg_length >= 150 and avg_confidence >= 0.7:
                depth_quality = "good"
            elif avg_length >= 100 and avg_confidence >= 0.6:
                depth_quality = "moderate"
            else:
                depth_quality = "poor"
            
            return {
                "average_depth": avg_length,
                "average_confidence": avg_confidence,
                "depth_quality": depth_quality,
                "total_insights": len(response.insights)
            }
            
        except Exception as e:
            logger.warning(f"Insight depth analysis error: {e}")
            return {"depth_quality": "unknown"}
    
    async def _calculate_collaboration_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall collaboration metrics."""
        try:
            all_metrics = []
            total_insights = 0
            total_response_time = 0
            successful_collaborations = 0
            
            for scenario_result in test_results["scenario_results"].values():
                if scenario_result.get("status") in ["success", "warning"]:
                    successful_collaborations += 1
                    
                    metrics = scenario_result.get("collaboration_metrics", {})
                    if metrics:
                        all_metrics.append(metrics)
                        total_insights += metrics.get("insights_generated", 0)
                        total_response_time += metrics.get("response_time_ms", 0)
            
            if all_metrics:
                avg_response_time = total_response_time / len(all_metrics)
                avg_insights_per_collaboration = total_insights / len(all_metrics)
                
                return {
                    "total_collaborations_tested": test_results["scenarios_tested"],
                    "successful_collaborations": successful_collaborations,
                    "success_rate": successful_collaborations / test_results["scenarios_tested"] if test_results["scenarios_tested"] > 0 else 0,
                    "average_response_time_ms": avg_response_time,
                    "average_insights_per_collaboration": avg_insights_per_collaboration,
                    "total_insights_generated": total_insights,
                    "collaboration_efficiency": avg_insights_per_collaboration / (avg_response_time / 1000) if avg_response_time > 0 else 0  # insights per second
                }
            else:
                return {"error": "No successful collaborations to analyze"}
                
        except Exception as e:
            logger.error(f"Collaboration metrics calculation error: {e}")
            return {"error": str(e)}
    
    def _print_collaboration_summary(self, results: Dict[str, Any]):
        """Print comprehensive collaboration test summary."""
        print(f"\n{Colors.WHITE}{'='*20} COLLABORATION TEST SUMMARY {'='*20}{Colors.NC}")
        
        overall_status = results["overall_status"]
        scenarios_tested = results["scenarios_tested"]
        scenarios_passed = results["scenarios_passed"]
        scenarios_failed = results["scenarios_failed"]
        
        # Overall status
        if overall_status == "success":
            print(f"{Colors.GREEN}🎉 OVERALL STATUS: ALL COLLABORATIONS SUCCESSFUL! 🎉{Colors.NC}")
        elif overall_status == "warning":
            print(f"{Colors.YELLOW}⚠️  OVERALL STATUS: COLLABORATIONS WORKING WITH LIMITATIONS{Colors.NC}")
        else:
            print(f"{Colors.RED}❌ OVERALL STATUS: COLLABORATION FAILURES DETECTED{Colors.NC}")
        
        # Scenario summary
        print(f"\n🤝 Collaboration Scenarios:")
        print(f"   {Colors.GREEN}✅ Successful: {scenarios_passed}{Colors.NC}")
        print(f"   {Colors.RED}❌ Failed: {scenarios_failed}{Colors.NC}")
        print(f"   📝 Total: {scenarios_tested}")
        
        # Collaboration metrics
        metrics = results.get("collaboration_metrics", {})
        if metrics and "error" not in metrics:
            print(f"\n📊 Collaboration Performance:")
            print(f"   🎯 Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")
            print(f"   ⚡ Avg Response Time: {metrics.get('average_response_time_ms', 0):.1f}ms")
            print(f"   💡 Avg Insights per Collaboration: {metrics.get('average_insights_per_collaboration', 0):.1f}")
            print(f"   🚀 Collaboration Efficiency: {metrics.get('collaboration_efficiency', 0):.2f} insights/second")
        
        # Individual scenario results
        print(f"\n🔬 Scenario Results:")
        for scenario_name, scenario_result in results["scenario_results"].items():
            status = scenario_result.get("status", "unknown")
            quality = scenario_result.get("quality_assessment", {}).get("overall_quality", 0)
            
            if status == "success":
                emoji = "✅"
                color = Colors.GREEN
            elif status == "warning":
                emoji = "⚠️"
                color = Colors.YELLOW
            else:
                emoji = "❌"
                color = Colors.RED
            
            print(f"   {color}{emoji} {scenario_name}: Quality {quality:.2f}{Colors.NC}")
        
        # Cross-domain analysis
        cross_domain = results.get("cross_domain_analysis", {})
        if cross_domain.get("status") == "success":
            print(f"\n🌐 Cross-Domain Analysis: {Colors.GREEN}✅ Operational{Colors.NC}")
        else:
            print(f"\n🌐 Cross-Domain Analysis: {Colors.YELLOW}⚠️ Limited{Colors.NC}")
        
        # Production readiness
        print(f"\n🚀 Production Readiness Assessment:")
        
        if overall_status == "success":
            print(f"   {Colors.GREEN}✅ AutoGen GroupChat fully operational{Colors.NC}")
            print(f"   {Colors.GREEN}✅ Multi-agent collaboration working{Colors.NC}")
            print(f"   {Colors.GREEN}✅ Cross-domain analysis functional{Colors.NC}")
            print(f"   {Colors.GREEN}✅ Ready for revolutionary research!{Colors.NC}")
        elif overall_status == "warning":
            print(f"   {Colors.YELLOW}⚠️ Collaborative research working with limitations{Colors.NC}")
            print(f"   {Colors.YELLOW}⚠️ Some scenarios may have reduced quality{Colors.NC}")
            print(f"   {Colors.YELLOW}⚠️ Monitor collaboration quality in production{Colors.NC}")
        else:
            print(f"   {Colors.RED}❌ Collaboration system needs fixes{Colors.NC}")
            print(f"   {Colors.RED}❌ AutoGen GroupChat not fully functional{Colors.NC}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"collaborative_research_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")

async def main():
    """Main test execution."""
    try:
        tester = CollaborativeResearchTester()
        results = await tester.run_all_collaboration_tests()
        
        # Exit with appropriate code
        if results["overall_status"] == "success":
            print(f"\n{Colors.GREEN}🎉 COLLABORATIVE RESEARCH TESTS COMPLETED SUCCESSFULLY! 🎉{Colors.NC}")
            print(f"{Colors.GREEN}🤝 AutoGen GroupChat and multi-agent collaboration are REVOLUTIONARY! 🤝{Colors.NC}")
            sys.exit(0)
        elif results["overall_status"] == "warning":
            print(f"\n{Colors.YELLOW}⚠️ COLLABORATIVE RESEARCH TESTS COMPLETED WITH WARNINGS{Colors.NC}")
            sys.exit(1)
        else:
            print(f"\n{Colors.RED}❌ COLLABORATIVE RESEARCH TESTS FAILED{Colors.NC}")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Collaboration tests interrupted by user{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Collaboration test execution failed: {e}{Colors.NC}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())