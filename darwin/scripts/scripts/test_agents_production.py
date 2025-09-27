#!/usr/bin/env python3
"""Test DARWIN Agents Production - AutoGen Research Team Validation

🤖 AGENTS PRODUCTION TESTING REVOLUTIONARY SCRIPT
Script épico para testar e validar cada agent individual do AutoGen research team:
- Dr_Biomaterials: Scaffold analysis + KEC metrics expert
- Dr_Quantum: Quantum mechanics + quantum biology expert  
- Dr_Medical: Clinical diagnosis + precision medicine expert
- Dr_Pharmacology: Precision pharmacology + quantum pharmacology expert
- Dr_Mathematics: Spectral analysis + graph theory expert
- Dr_Philosophy: Consciousness studies + epistemology expert
- Dr_Literature: Scientific literature + research synthesis expert
- Dr_Synthesis: Interdisciplinary integration expert

Usage: python scripts/test_agents_production.py
"""

import asyncio
import sys
import time
import json
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from kec_unified_api.ai_agents.research_team import ResearchTeamCoordinator
    from kec_unified_api.ai_agents.agent_models import (
        CollaborativeResearchRequest,
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
logger = get_logger("darwin.agents_production_test")

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
    """Print epic agents test header."""
    print(f"""
{Colors.PURPLE}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🤖 DARWIN AGENTS PRODUCTION TESTING REVOLUTIONARY 🤖      ║
║                                                              ║
║  Testing AutoGen Multi-Agent Research Team:                 ║
║  • Individual Agent Specialization Validation              ║
║  • Domain Expertise Assessment                              ║
║  • Response Quality Analysis                                ║
║  • Performance & Latency Testing                            ║
║  • Production Readiness Validation                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.NC}
""")

def log_agent_test(agent_name: str, test_type: str, status: str = "running"):
    """Log agent test status with colors."""
    emoji_map = {
        "Dr_Biomaterials": "🧬",
        "Dr_Quantum": "🌌", 
        "Dr_Medical": "🏥",
        "Dr_Pharmacology": "💊",
        "Dr_Mathematics": "📊",
        "Dr_Philosophy": "🧠",
        "Dr_Literature": "📚",
        "Dr_Synthesis": "🔬"
    }
    
    emoji = emoji_map.get(agent_name, "🤖")
    
    if status == "running":
        print(f"{Colors.BLUE}🧪 [TESTING]{Colors.NC} {emoji} {agent_name} - {test_type}...")
    elif status == "success":
        print(f"{Colors.GREEN}✅ [SUCCESS]{Colors.NC} {emoji} {agent_name} - {test_type}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️  [WARNING]{Colors.NC} {emoji} {agent_name} - {test_type}")
    elif status == "error":
        print(f"{Colors.RED}❌ [ERROR]{Colors.NC} {emoji} {agent_name} - {test_type}")

class AgentsProductionTester:
    """Comprehensive agents production testing class."""
    
    def __init__(self):
        self.research_team: Optional[ResearchTeamCoordinator] = None
        self.test_results: Dict[str, Any] = {}
        
        # Agent test scenarios
        self.agent_test_scenarios = {
            AgentSpecialization.BIOMATERIALS: {
                "agent_name": "Dr_Biomaterials",
                "test_questions": [
                    "What are the optimal KEC metrics for bone tissue engineering scaffolds with high porosity?",
                    "How does scaffold pore interconnectivity affect biocompatibility and cell proliferation?",
                    "Compare collagen vs chitosan scaffolds for neural tissue engineering applications."
                ],
                "expected_keywords": ["scaffold", "biocompatibility", "KEC", "porosity", "tissue engineering"],
                "expertise_domains": ["biomaterials", "scaffolds", "tissue engineering", "biocompatibility"]
            },
            AgentSpecialization.QUANTUM_MECHANICS: {
                "agent_name": "Dr_Quantum",
                "test_questions": [
                    "Analyze quantum coherence effects in biomaterial scaffolds and potential applications.",
                    "Explain quantum tunneling effects in drug-biomaterial interactions.",
                    "How can quantum computing optimize scaffold architecture and KEC metrics prediction?"
                ],
                "expected_keywords": ["quantum", "coherence", "tunneling", "quantum computing", "quantum biology"],
                "expertise_domains": ["quantum mechanics", "quantum biology", "quantum computing"]
            },
            AgentSpecialization.CLINICAL_PSYCHIATRY: {
                "agent_name": "Dr_Medical",
                "test_questions": [
                    "Patient presents with treatment-resistant depression. Develop comprehensive treatment plan.",
                    "Evaluate biomaterial applications for spinal cord injury treatment and clinical considerations.",
                    "Design precision medicine approach for psychiatric patient with complex genetic profile."
                ],
                "expected_keywords": ["clinical", "diagnosis", "treatment", "patient", "medical"],
                "expertise_domains": ["clinical medicine", "psychiatry", "precision medicine"]
            },
            AgentSpecialization.PHARMACOLOGY: {
                "agent_name": "Dr_Pharmacology",
                "test_questions": [
                    "Design precision dosing strategy for CYP2D6 poor metabolizer requiring antipsychotic therapy.",
                    "Analyze quantum pharmacology effects in cytochrome P450 metabolism.",
                    "Develop personalized pharmacotherapy protocol considering genetic and quantum factors."
                ],
                "expected_keywords": ["pharmacology", "dosing", "metabolism", "pharmacogenomics", "precision"],
                "expertise_domains": ["pharmacology", "pharmacogenomics", "precision medicine"]
            },
            AgentSpecialization.MATHEMATICS: {
                "agent_name": "Dr_Mathematics",
                "test_questions": [
                    "Explain the mathematical foundations of spectral entropy in network analysis.",
                    "Derive the relationship between Forman curvature and scaffold mechanical properties.",
                    "Analyze small-world properties using graph-theoretic approaches for scaffold optimization."
                ],
                "expected_keywords": ["mathematical", "spectral", "analysis", "graph theory", "eigenvalues"],
                "expertise_domains": ["mathematics", "spectral analysis", "graph theory", "topology"]
            },
            AgentSpecialization.PHILOSOPHY: {
                "agent_name": "Dr_Philosophy",
                "test_questions": [
                    "Explore philosophical implications of consciousness in AI-assisted medical diagnosis.",
                    "Analyze epistemological foundations of interdisciplinary research in biomaterials.",
                    "Examine ethical considerations in quantum-enhanced medical interventions."
                ],
                "expected_keywords": ["philosophy", "consciousness", "epistemology", "ethics", "conceptual"],
                "expertise_domains": ["philosophy", "consciousness studies", "epistemology", "ethics"]
            },
            AgentSpecialization.LITERATURE: {
                "agent_name": "Dr_Literature",
                "test_questions": [
                    "Provide comprehensive literature review on quantum effects in biological systems.",
                    "Synthesize current research on biomaterial scaffold optimization strategies.",
                    "Identify research gaps in interdisciplinary quantum-biomaterial studies."
                ],
                "expected_keywords": ["literature", "research", "studies", "bibliography", "review"],
                "expertise_domains": ["literature review", "research synthesis", "bibliographic analysis"]
            },
            AgentSpecialization.SYNTHESIS: {
                "agent_name": "Dr_Synthesis",
                "test_questions": [
                    "Synthesize insights from biomaterials, quantum mechanics, and medical perspectives.",
                    "Integrate interdisciplinary findings into coherent research narrative.",
                    "Generate novel hypotheses from cross-domain analysis and synthesis."
                ],
                "expected_keywords": ["synthesis", "integration", "interdisciplinary", "connections", "insights"],
                "expertise_domains": ["synthesis", "integration", "interdisciplinary research"]
            }
        }
    
    async def run_all_agent_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests for all agents."""
        print_epic_header()
        
        test_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "agents_tested": 0,
            "agents_passed": 0,
            "agents_failed": 0,
            "agent_results": {},
            "team_performance": {}
        }
        
        try:
            # Initialize research team
            await self._initialize_research_team()
            
            if not self.research_team or not self.research_team.is_initialized:
                test_results["overall_status"] = "failed"
                test_results["error"] = "Research team initialization failed"
                return test_results
            
            # Test each agent individually
            for specialization, test_config in self.agent_test_scenarios.items():
                agent_name = test_config["agent_name"]
                
                print(f"\n{Colors.WHITE}{'='*15} Testing {agent_name} {'='*15}{Colors.NC}")
                
                test_results["agents_tested"] += 1
                
                try:
                    agent_result = await self._test_individual_agent(specialization, test_config)
                    test_results["agent_results"][agent_name] = agent_result
                    
                    if agent_result.get("status") == "success":
                        test_results["agents_passed"] += 1
                    else:
                        test_results["agents_failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Agent {agent_name} test failed: {e}")
                    test_results["agent_results"][agent_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    test_results["agents_failed"] += 1
            
            # Test team performance
            team_performance = await self._test_team_performance()
            test_results["team_performance"] = team_performance
            
            # Determine overall status
            if test_results["agents_failed"] == 0:
                test_results["overall_status"] = "success"
            elif test_results["agents_passed"] > test_results["agents_failed"]:
                test_results["overall_status"] = "warning"
            else:
                test_results["overall_status"] = "failed"
            
            # Print summary
            self._print_test_summary(test_results)
            
        except Exception as e:
            logger.error(f"Agent testing failed: {e}")
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
            log_agent_test("Research Team", "Initialization", "running")
            
            self.research_team = ResearchTeamCoordinator()
            await self.research_team.initialize()
            
            if self.research_team.is_initialized:
                log_agent_test("Research Team", "Initialization", "success")
            else:
                log_agent_test("Research Team", "Initialization", "error")
                
        except Exception as e:
            logger.error(f"Research team initialization failed: {e}")
            log_agent_test("Research Team", "Initialization", "error")
            raise
    
    async def _test_individual_agent(
        self,
        specialization: AgentSpecialization,
        test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test individual agent performance."""
        agent_name = test_config["agent_name"]
        
        result = {
            "status": "success",
            "agent_name": agent_name,
            "specialization": specialization.value,
            "tests_completed": 0,
            "tests_passed": 0,
            "response_quality": {},
            "performance_metrics": {},
            "errors": []
        }
        
        try:
            # Test each question for this agent
            for i, question in enumerate(test_config["test_questions"]):
                log_agent_test(agent_name, f"Question {i+1}", "running")
                
                result["tests_completed"] += 1
                
                # Create targeted research request
                research_request = CollaborativeResearchRequest(
                    research_question=question,
                    target_specializations=[specialization],
                    max_agents=1,  # Test individual agent
                    max_rounds=3,
                    priority=ResearchPriority.HIGH,
                    include_synthesis=False  # Focus on individual response
                )
                
                # Measure response time
                start_time = time.time()
                
                try:
                    # Get agent response
                    response = await self.research_team.collaborative_research(research_request)
                    
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    # Analyze response quality
                    quality_score = self._analyze_response_quality(
                        response,
                        test_config["expected_keywords"],
                        test_config["expertise_domains"]
                    )
                    
                    if quality_score >= 0.7:  # 70% quality threshold
                        result["tests_passed"] += 1
                        log_agent_test(agent_name, f"Question {i+1} (Quality: {quality_score:.2f})", "success")
                    else:
                        log_agent_test(agent_name, f"Question {i+1} (Quality: {quality_score:.2f})", "warning")
                    
                    # Store metrics
                    result["response_quality"][f"question_{i+1}"] = {
                        "quality_score": quality_score,
                        "response_time_ms": response_time,
                        "insights_count": len(response.insights),
                        "confidence_score": response.confidence_score
                    }
                    
                except Exception as e:
                    result["errors"].append(f"Question {i+1} failed: {str(e)}")
                    log_agent_test(agent_name, f"Question {i+1}", "error")
            
            # Calculate performance metrics
            if result["tests_completed"] > 0:
                result["performance_metrics"] = {
                    "success_rate": result["tests_passed"] / result["tests_completed"],
                    "average_quality": sum(q["quality_score"] for q in result["response_quality"].values()) / len(result["response_quality"]) if result["response_quality"] else 0,
                    "average_response_time": sum(q["response_time_ms"] for q in result["response_quality"].values()) / len(result["response_quality"]) if result["response_quality"] else 0
                }
                
                # Determine agent status
                if result["performance_metrics"]["success_rate"] >= 0.8 and result["performance_metrics"]["average_quality"] >= 0.7:
                    result["status"] = "success"
                    log_agent_test(agent_name, "Overall Performance", "success")
                elif result["performance_metrics"]["success_rate"] >= 0.6:
                    result["status"] = "warning"
                    log_agent_test(agent_name, "Overall Performance", "warning")
                else:
                    result["status"] = "error"
                    log_agent_test(agent_name, "Overall Performance", "error")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_agent_test(agent_name, "Testing", "error")
        
        return result
    
    def _analyze_response_quality(
        self,
        response: Any,
        expected_keywords: List[str],
        expertise_domains: List[str]
    ) -> float:
        """Analyze response quality for domain expertise."""
        try:
            if not response or not response.insights:
                return 0.0
            
            # Combine all insight content
            full_response = " ".join(insight.content for insight in response.insights)
            full_response_lower = full_response.lower()
            
            # Check for expected keywords
            keyword_score = 0.0
            for keyword in expected_keywords:
                if keyword.lower() in full_response_lower:
                    keyword_score += 1
            keyword_score = keyword_score / len(expected_keywords) if expected_keywords else 0
            
            # Check for domain expertise
            domain_score = 0.0
            for domain in expertise_domains:
                if domain.lower() in full_response_lower:
                    domain_score += 1
            domain_score = domain_score / len(expertise_domains) if expertise_domains else 0
            
            # Response length quality (adequate depth)
            length_score = min(1.0, len(full_response) / 200)  # Normalize to 200 chars minimum
            
            # Confidence score from response
            confidence_score = response.confidence_score if hasattr(response, 'confidence_score') else 0.5
            
            # Combined quality score
            quality_score = (
                keyword_score * 0.3 +      # 30% keywords
                domain_score * 0.3 +       # 30% domain expertise  
                length_score * 0.2 +       # 20% response depth
                confidence_score * 0.2     # 20% confidence
            )
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.warning(f"Response quality analysis error: {e}")
            return 0.5  # Default moderate score
    
    async def _test_team_performance(self) -> Dict[str, Any]:
        """Test overall research team performance."""
        log_agent_test("Research Team", "Team Performance", "running")
        
        result = {
            "status": "success",
            "team_metrics": {},
            "errors": []
        }
        
        try:
            # Get team status
            team_status = await self.research_team.get_team_status()
            
            result["team_metrics"] = {
                "total_agents": team_status.total_agents,
                "active_agents": team_status.active_agents,
                "completed_researches": team_status.completed_researches,
                "team_performance": team_status.team_performance
            }
            
            # Test multi-agent collaboration
            multi_agent_request = CollaborativeResearchRequest(
                research_question="How can quantum-enhanced biomaterials revolutionize neural tissue engineering?",
                target_specializations=[
                    AgentSpecialization.BIOMATERIALS,
                    AgentSpecialization.QUANTUM_MECHANICS,
                    AgentSpecialization.CLINICAL_PSYCHIATRY
                ],
                max_agents=3,
                max_rounds=5,
                include_synthesis=True
            )
            
            start_time = time.time()
            
            collaboration_response = await self.research_team.collaborative_research(multi_agent_request)
            
            collaboration_time = (time.time() - start_time) * 1000
            
            # Analyze collaboration quality
            if collaboration_response and collaboration_response.status == "completed":
                result["team_metrics"]["collaboration_test"] = {
                    "status": "success",
                    "response_time_ms": collaboration_time,
                    "participating_agents": len(collaboration_response.participating_agents),
                    "insights_generated": len(collaboration_response.insights),
                    "confidence_score": collaboration_response.confidence_score,
                    "synthesis_available": bool(collaboration_response.synthesis)
                }
                
                log_agent_test("Research Team", "Multi-Agent Collaboration", "success")
            else:
                result["status"] = "warning"
                result["errors"].append("Multi-agent collaboration failed")
                log_agent_test("Research Team", "Multi-Agent Collaboration", "warning")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            log_agent_test("Research Team", "Team Performance", "error")
        
        return result
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print(f"\n{Colors.WHITE}{'='*20} AGENTS TEST SUMMARY {'='*20}{Colors.NC}")
        
        overall_status = results["overall_status"]
        agents_tested = results["agents_tested"]
        agents_passed = results["agents_passed"]
        agents_failed = results["agents_failed"]
        
        # Overall status
        if overall_status == "success":
            print(f"{Colors.GREEN}🎉 OVERALL STATUS: ALL AGENTS OPERATIONAL! 🎉{Colors.NC}")
        elif overall_status == "warning":
            print(f"{Colors.YELLOW}⚠️  OVERALL STATUS: AGENTS OPERATIONAL WITH LIMITATIONS{Colors.NC}")
        else:
            print(f"{Colors.RED}❌ OVERALL STATUS: SOME AGENTS FAILED{Colors.NC}")
        
        # Agent summary
        print(f"\n🤖 Agent Test Results:")
        print(f"   {Colors.GREEN}✅ Operational: {agents_passed}{Colors.NC}")
        print(f"   {Colors.RED}❌ Failed: {agents_failed}{Colors.NC}")
        print(f"   📝 Total: {agents_tested}")
        
        # Individual agent performance
        print(f"\n👨‍🔬 Individual Agent Performance:")
        
        for agent_name, agent_result in results["agent_results"].items():
            if agent_result.get("status") == "success":
                emoji = "✅"
                color = Colors.GREEN
            elif agent_result.get("status") == "warning":
                emoji = "⚠️"
                color = Colors.YELLOW
            else:
                emoji = "❌"
                color = Colors.RED
            
            performance = agent_result.get("performance_metrics", {})
            success_rate = performance.get("success_rate", 0)
            avg_quality = performance.get("average_quality", 0)
            
            print(f"   {color}{emoji} {agent_name}: {success_rate*100:.1f}% success, {avg_quality:.2f} quality{Colors.NC}")
        
        # Team performance
        team_perf = results.get("team_performance", {})
        if team_perf.get("status") == "success":
            print(f"\n🎯 Team Collaboration: {Colors.GREEN}✅ Operational{Colors.NC}")
            collab_metrics = team_perf.get("team_metrics", {}).get("collaboration_test", {})
            if collab_metrics:
                print(f"   🤝 Multi-Agent Response: {collab_metrics.get('response_time_ms', 0):.1f}ms")
                print(f"   🧠 Agents Participated: {collab_metrics.get('participating_agents', 0)}")
                print(f"   💡 Insights Generated: {collab_metrics.get('insights_generated', 0)}")
        else:
            print(f"\n🎯 Team Collaboration: {Colors.YELLOW}⚠️ Limited{Colors.NC}")
        
        # Production readiness assessment
        print(f"\n🚀 Production Readiness Assessment:")
        
        if overall_status == "success":
            print(f"   {Colors.GREEN}✅ Ready for production deployment{Colors.NC}")
            print(f"   {Colors.GREEN}✅ All specialist agents operational{Colors.NC}")
            print(f"   {Colors.GREEN}✅ Multi-agent collaboration working{Colors.NC}")
        elif overall_status == "warning":
            print(f"   {Colors.YELLOW}⚠️ Ready with limitations{Colors.NC}")
            print(f"   {Colors.YELLOW}⚠️ Some agents may have reduced capability{Colors.NC}")
            print(f"   {Colors.YELLOW}⚠️ Monitor performance in production{Colors.NC}")
        else:
            print(f"   {Colors.RED}❌ Not ready for production{Colors.NC}")
            print(f"   {Colors.RED}❌ Critical agent failures detected{Colors.NC}")
            print(f"   {Colors.RED}❌ Fix issues before deployment{Colors.NC}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"agents_production_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")

async def main():
    """Main test execution."""
    try:
        tester = AgentsProductionTester()
        results = await tester.run_all_agent_tests()
        
        # Exit with appropriate code
        if results["overall_status"] == "success":
            print(f"\n{Colors.GREEN}🎉 AGENTS PRODUCTION TESTS COMPLETED SUCCESSFULLY! 🎉{Colors.NC}")
            print(f"{Colors.GREEN}🤖 All DARWIN research agents are ready for revolutionary research! 🤖{Colors.NC}")
            sys.exit(0)
        elif results["overall_status"] == "warning":
            print(f"\n{Colors.YELLOW}⚠️ AGENTS PRODUCTION TESTS COMPLETED WITH WARNINGS{Colors.NC}")
            sys.exit(1)
        else:
            print(f"\n{Colors.RED}❌ AGENTS PRODUCTION TESTS FAILED{Colors.NC}")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Agent tests interrupted by user{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Agent test execution failed: {e}{Colors.NC}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())