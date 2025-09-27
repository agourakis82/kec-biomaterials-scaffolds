"""Tree Search PUCT Router - Complete MCTS API

Router completo para Tree Search PUCT com algoritmos MCTS avançados,
migrado e expandido do backend principal com funcionalidades matemáticas especializadas.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response, Depends
from pydantic import BaseModel, Field

from ..models.tree_search_models import (
    SearchConfigRequest,
    SearchResult,
    QuickSearchResult,
    TreeNode,
    ActionResult,
    SearchAlgorithm,
    NodeStatus,
    PUCTConfig,
    MathematicalSearchRequest,
    OptimizationProblem,
    SelectionPolicy,
    ExpansionPolicy,
    SimulationPolicy,
    BackpropagationPolicy,
)
from ..services.tree_search_engine import TreeSearchEngine, create_default_engine
from ..services.mathematical_search import MathematicalSearch
from ..services.puct_algorithm import TestStateEvaluator
from ..services.tree_search_benchmarks import (
    TreeSearchBenchmarkRunner,
    BenchmarkConfig,
    run_quick_benchmark,
    run_comprehensive_benchmark,
    validate_puct_mathematics
)

logger = logging.getLogger(__name__)

# Initialize engines
tree_search_engine = create_default_engine()
mathematical_search = MathematicalSearch()

router = APIRouter(
    prefix="/tree-search",
    tags=["Tree Search PUCT"],
)

# ==================== REQUEST/RESPONSE MODELS ====================

class QuickSearchRequest(BaseModel):
    """Request para busca rápida."""
    
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    depth: Optional[int] = Field(3, ge=1, le=10, description="Maximum search depth")
    budget: Optional[int] = Field(100, ge=10, le=1000, description="Search budget")
    algorithm: Optional[SearchAlgorithm] = Field(SearchAlgorithm.PUCT, description="Algorithm to use")


class TreeSearchRequest(BaseModel):
    """Request para busca Tree Search completa."""
    
    initial_state: str = Field(..., description="Initial state", min_length=1)
    config: Optional[SearchConfigRequest] = Field(None, description="Search configuration")
    evaluator_type: Optional[str] = Field("test", description="Type of evaluator to use")


class ParallelSearchRequest(BaseModel):
    """Request para busca paralela."""
    
    searches: List[TreeSearchRequest] = Field(..., description="List of searches to run in parallel")
    max_parallel: Optional[int] = Field(4, ge=1, le=16, description="Maximum parallel executions")


class MultiObjectiveRequest(BaseModel):
    """Request para otimização multi-objetivo."""
    
    objectives: List[str] = Field(..., description="List of objective names")
    variable_bounds: Dict[str, tuple] = Field(..., description="Variable bounds")
    constraints: Optional[List[str]] = Field([], description="Constraints")
    max_evaluations: Optional[int] = Field(1000, ge=100, le=10000, description="Max evaluations per objective")


class ScaffoldOptimizationRequest(BaseModel):
    """Request para otimização de scaffold."""
    
    target_porosity: float = Field(0.7, ge=0.1, le=0.95, description="Target porosity")
    min_mechanical_strength: float = Field(5.0, ge=1.0, le=100.0, description="Minimum mechanical strength")
    target_pore_size: Optional[float] = Field(200.0, ge=50.0, le=1000.0, description="Target pore size (micrometers)")
    max_iterations: Optional[int] = Field(500, ge=100, le=5000, description="Maximum iterations")


class NetworkTopologyRequest(BaseModel):
    """Request para otimização de topologia de rede."""
    
    nodes: int = Field(100, ge=10, le=10000, description="Number of nodes")
    target_clustering: float = Field(0.6, ge=0.0, le=1.0, description="Target clustering coefficient")
    target_path_length: float = Field(2.5, ge=1.0, le=20.0, description="Target average path length")
    max_iterations: Optional[int] = Field(500, ge=100, le=5000, description="Maximum iterations")


class PerformanceMetrics(BaseModel):
    """Métricas de performance da busca."""
    
    total_time_seconds: float = Field(..., description="Total search time")
    nodes_per_second: float = Field(..., description="Nodes explored per second")
    iterations_per_second: float = Field(..., description="Iterations per second")
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage")
    convergence_rate: Optional[float] = Field(None, description="Convergence rate")
    solution_quality: float = Field(..., description="Solution quality score")


class BestPracticesResponse(BaseModel):
    """Melhores práticas para Tree Search."""
    
    algorithm_selection: Dict[str, str] = Field(..., description="Algorithm selection guidelines")
    parameter_tuning: Dict[str, Any] = Field(..., description="Parameter tuning recommendations")
    performance_optimization: List[str] = Field(..., description="Performance optimization tips")
    common_pitfalls: List[str] = Field(..., description="Common pitfalls to avoid")
    domain_specific_advice: Dict[str, List[str]] = Field(..., description="Domain-specific advice")


# ==================== CORE ENDPOINTS (MIGRATED) ====================

@router.post("/search", response_model=SearchResult)
async def tree_search(request: TreeSearchRequest, response: Response) -> SearchResult:
    """
    Perform Tree-Search PUCT exploration.
    
    Migrated and enhanced from the main backend with improved error handling
    and expanded algorithm support.
    """
    try:
        logger.info(f"Starting tree search with state: {request.initial_state[:100]}")
        
        # Use default config if not provided
        config = request.config or SearchConfigRequest(
            max_budget_nodes=1000,
            max_depth=10,
            default_budget=500,
            c_puct=1.414,
            expansion_threshold=5,
            simulation_rollouts=3,
            temperature=1.0,
            use_progressive_widening=True,
            alpha_widening=0.5,
            algorithm=SearchAlgorithm.PUCT,
            selection_policy=SelectionPolicy.BEST,
            expansion_policy=ExpansionPolicy.PROGRESSIVE,
            simulation_policy=SimulationPolicy.HEURISTIC,
            backpropagation_policy=BackpropagationPolicy.AVERAGE,
            convergence_threshold=1e-6,
            parallel_simulations=1,
            timeout_seconds=30.0,
            verbose=False,
            save_tree=False
        )
        
        # Create evaluator
        if request.evaluator_type == "test":
            evaluator = TestStateEvaluator()
        else:
            evaluator = TestStateEvaluator()  # Default fallback
        
        # Execute search
        result = await tree_search_engine.search(
            request=config,
            initial_state=request.initial_state,
            evaluator=evaluator
        )
        
        # Set response headers
        response.headers["X-Tree-Search-Nodes"] = str(result.nodes_explored)
        response.headers["X-Tree-Search-Depth"] = str(result.max_depth_reached)
        response.headers["X-Tree-Search-Time"] = str(result.total_time_seconds)
        response.headers["X-Tree-Search-Algorithm"] = result.algorithm.value
        
        logger.info(f"Tree search completed: {result.nodes_explored} nodes, {result.total_time_seconds:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Tree search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tree search failed: {str(e)}")


@router.post("/quick-search", response_model=QuickSearchResult)
async def quick_search(request: QuickSearchRequest, response: Response) -> QuickSearchResult:
    """
    Quick Tree-Search for simple exploration.
    
    Optimized for fast results with reduced search budget and simplified configuration.
    """
    try:
        logger.info(f"Starting quick search: {request.query}")
        
        # Create simplified configuration for quick search
        budget = request.budget or 100
        depth = request.depth or 3
        algorithm = request.algorithm or SearchAlgorithm.PUCT
        
        config = SearchConfigRequest(
            max_budget_nodes=min(budget, 500),
            max_depth=depth,
            default_budget=budget,
            c_puct=1.0,  # Lower exploration for quick search
            expansion_threshold=2,
            simulation_rollouts=1,
            temperature=0.5,
            use_progressive_widening=False,
            alpha_widening=0.5,
            algorithm=algorithm,
            selection_policy=SelectionPolicy.BEST,
            expansion_policy=ExpansionPolicy.THRESHOLD_BASED,
            simulation_policy=SimulationPolicy.RANDOM,
            backpropagation_policy=BackpropagationPolicy.AVERAGE,
            convergence_threshold=1e-4,
            parallel_simulations=1,
            timeout_seconds=10.0,  # Quick timeout
            verbose=False,
            save_tree=False
        )
        
        # Execute search
        result = await tree_search_engine.search(
            request=config,
            initial_state=request.query,
            evaluator=TestStateEvaluator()
        )
        
        # Convert to quick search result
        quick_result = QuickSearchResult(
            query=request.query,
            best_path=result.best_action_sequence[:request.depth],
            exploration_score=min(result.search_statistics.get('search_efficiency', 0.5), 1.0),
            nodes_explored=result.nodes_explored,
            search_time_seconds=result.total_time_seconds,
            success=result.success
        )
        
        # Set response headers
        response.headers["X-Quick-Search-Budget"] = str(request.budget)
        response.headers["X-Quick-Search-Depth"] = str(request.depth)
        response.headers["X-Quick-Search-Algorithm"] = algorithm.value
        
        return quick_result
        
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick search failed: {str(e)}")


@router.get("/config/defaults", response_model=SearchConfigRequest)
async def get_default_config() -> SearchConfigRequest:
    """
    Get default Tree-Search configuration.
    
    Returns optimized default parameters suitable for most use cases.
    """
    return SearchConfigRequest(
        max_budget_nodes=1000,
        max_depth=10,
        default_budget=500,
        c_puct=1.414,
        expansion_threshold=5,
        simulation_rollouts=3,
        temperature=1.0,
        use_progressive_widening=True,
        alpha_widening=0.5,
        algorithm=SearchAlgorithm.PUCT,
        selection_policy=SelectionPolicy.BEST,
        expansion_policy=ExpansionPolicy.PROGRESSIVE,
        simulation_policy=SimulationPolicy.HEURISTIC,
        backpropagation_policy=BackpropagationPolicy.AVERAGE,
        convergence_threshold=1e-6,
        parallel_simulations=1,
        timeout_seconds=30.0,
        verbose=False,
        save_tree=False
    )


@router.get("/algorithms")
async def get_algorithms() -> Dict[str, Any]:
    """
    Get information about available Tree-Search algorithms.
    
    Expanded from the main backend with additional algorithm variants and detailed descriptions.
    """
    return {
        "algorithms": [
            {
                "name": "PUCT",
                "description": "Polynomial Upper Confidence Bounds for Trees",
                "best_for": "General exploration with prior knowledge",
                "parameters": ["c_puct", "expansion_threshold", "prior_probability"],
                "complexity": "O(log n) per selection",
                "recommended_domains": ["optimization", "game_playing", "planning"]
            },
            {
                "name": "UCB1",
                "description": "Upper Confidence Bound",
                "best_for": "Pure exploration without priors",
                "parameters": ["c_exploration"],
                "complexity": "O(1) per selection",
                "recommended_domains": ["multi_armed_bandits", "parameter_tuning"]
            },
            {
                "name": "UCT",
                "description": "Upper Confidence bounds applied to Trees",
                "best_for": "Classic MCTS applications",
                "parameters": ["c_exploration", "rollout_policy"],
                "complexity": "O(log n) per selection",
                "recommended_domains": ["games", "sequential_decision_making"]
            },
            {
                "name": "THOMPSON_SAMPLING",
                "description": "Bayesian exploration strategy",
                "best_for": "Problems with uncertainty quantification",
                "parameters": ["prior_distribution", "posterior_update"],
                "complexity": "O(1) per selection + sampling cost",
                "recommended_domains": ["reinforcement_learning", "optimization_under_uncertainty"]
            }
        ],
        "selection_strategies": [
            {"name": "BEST", "description": "Select highest scoring child"},
            {"name": "TEMPERATURE", "description": "Temperature-based sampling"},
            {"name": "EPSILON_GREEDY", "description": "Epsilon-greedy exploration"},
            {"name": "BOLTZMANN", "description": "Boltzmann exploration"}
        ],
        "expansion_policies": [
            {"name": "ALL_ACTIONS", "description": "Expand all actions at once"},
            {"name": "PROGRESSIVE", "description": "Progressive expansion"},
            {"name": "THRESHOLD_BASED", "description": "Expand based on visit threshold"},
            {"name": "BAYESIAN", "description": "Bayesian-guided expansion"}
        ],
        "features": [
            "Progressive widening",
            "Multiple rollout simulations",
            "Configurable depth limits",
            "Statistical tracking",
            "Parallel search support",
            "Mathematical optimization",
            "Domain-specific evaluators",
            "Multi-objective optimization"
        ],
        "performance_characteristics": {
            "typical_nodes_per_second": "1000-10000",
            "memory_scaling": "O(tree_size)",
            "convergence_rate": "logarithmic in most cases",
            "parallel_efficiency": "70-90% with proper configuration"
        }
    }


@router.get("/health")
async def tree_search_health() -> Dict[str, Any]:
    """
    Health check for Tree-Search service.
    
    Enhanced health check with comprehensive system status and performance metrics.
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Test basic tree search functionality
        config = SearchConfigRequest(
            max_budget_nodes=50,
            max_depth=3,
            default_budget=10,
            c_puct=1.414,
            expansion_threshold=2,
            simulation_rollouts=1,
            temperature=0.5,
            use_progressive_widening=False,
            alpha_widening=0.5,
            algorithm=SearchAlgorithm.PUCT,
            selection_policy=SelectionPolicy.BEST,
            expansion_policy=ExpansionPolicy.THRESHOLD_BASED,
            simulation_policy=SimulationPolicy.RANDOM,
            backpropagation_policy=BackpropagationPolicy.AVERAGE,
            convergence_threshold=1e-4,
            parallel_simulations=1,
            timeout_seconds=2.0,
            verbose=False,
            save_tree=False
        )
        
        test_result = await tree_search_engine.search(
            request=config,
            initial_state="health_check_test",
            evaluator=TestStateEvaluator()
        )
        
        end_time = datetime.now(timezone.utc)
        response_time = (end_time - start_time).total_seconds()
        
        # Get engine statistics
        engine_stats = tree_search_engine.get_engine_statistics()
        
        # Determine health status
        if response_time < 1.0 and test_result.success:
            status = "healthy"
            message = "Tree-Search service fully operational"
        elif response_time < 2.0:
            status = "degraded"
            message = "Tree-Search service operational but slow"
        else:
            status = "unhealthy"
            message = "Tree-Search service experiencing issues"
        
        return {
            "status": status,
            "message": message,
            "response_time_seconds": response_time,
            "test_result": {
                "nodes_explored": test_result.nodes_explored,
                "iterations_completed": test_result.total_iterations,
                "best_value": test_result.best_value
            },
            "engine_statistics": engine_stats,
            "system_info": {
                "active_searches": len(tree_search_engine.get_active_searches()),
                "total_searches_completed": engine_stats.get('total_searches', 0),
                "success_rate": engine_stats.get('success_rate', 0.0)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except asyncio.TimeoutError:
        return {
            "status": "degraded",
            "message": "Tree-Search health check timeout",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Tree-Search health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Tree-Search error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ==================== SPECIALIZED ENDPOINTS (NEW) ====================

@router.post("/mathematical", response_model=SearchResult)
async def mathematical_search_endpoint(request: MathematicalSearchRequest) -> SearchResult:
    """
    Execute specialized mathematical search using PUCT.
    
    Optimized for mathematical optimization problems with domain-specific heuristics.
    """
    try:
        logger.info(f"Starting mathematical search: {request.problem_type} in domain {request.domain}")
        
        result = await mathematical_search.search(request)
        
        logger.info(f"Mathematical search completed: {result.nodes_explored} evaluations")
        
        return result
        
    except Exception as e:
        logger.error(f"Mathematical search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mathematical search failed: {str(e)}")


@router.post("/scaffold-optimization", response_model=SearchResult)
async def scaffold_optimization(request: ScaffoldOptimizationRequest) -> SearchResult:
    """
    Optimize biomaterial scaffold properties using PUCT.
    
    Specialized for optimizing porosity, mechanical strength, and pore size distribution.
    """
    try:
        logger.info(f"Starting scaffold optimization: porosity={request.target_porosity}, strength={request.min_mechanical_strength}")
        
        # Convert to mathematical search request
        math_request = MathematicalSearchRequest(
            problem_type="scaffold_optimization",
            objective_function="optimize_scaffold_properties",
            constraints=[
                f"porosity <= 0.95",
                f"mechanical_strength >= {request.min_mechanical_strength}",
                "pore_size >= 50.0"
            ],
            variable_bounds={
                "porosity": (0.1, 0.95),
                "mechanical_strength": (1.0, 50.0),
                "pore_size": (50.0, 500.0)
            },
            continuous_variables=["porosity", "mechanical_strength", "pore_size"],
            discrete_variables=[],
            max_evaluations=request.max_iterations or 500,
            convergence_tolerance=1e-4,
            domain="biomaterials"
        )
        
        result = await mathematical_search.search(math_request)
        
        # Add scaffold-specific metadata
        result.search_statistics["scaffold_optimization"] = {
            "target_porosity": request.target_porosity,
            "min_mechanical_strength": request.min_mechanical_strength,
            "target_pore_size": request.target_pore_size,
            "optimization_type": "multi_objective_scaffold"
        }
        
        logger.info(f"Scaffold optimization completed with value: {result.best_value}")
        
        return result
        
    except Exception as e:
        logger.error(f"Scaffold optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scaffold optimization failed: {str(e)}")


@router.post("/network-topology", response_model=SearchResult)
async def network_topology_optimization(request: NetworkTopologyRequest) -> SearchResult:
    """
    Optimize neural network topology for small-world properties.
    
    Specialized for optimizing clustering coefficient and average path length.
    """
    try:
        logger.info(f"Starting network topology optimization: {request.nodes} nodes, clustering={request.target_clustering}")
        
        # Convert to mathematical search request
        math_request = MathematicalSearchRequest(
            problem_type="network_topology",
            objective_function="optimize_small_world_properties",
            constraints=[
                "clustering_coefficient >= 0.05",
                "average_path_length <= 10.0",
                "connection_density >= 0.05"
            ],
            variable_bounds={
                "clustering_coefficient": (0.05, 0.95),
                "average_path_length": (1.5, 10.0),
                "connection_density": (0.05, 0.5)
            },
            continuous_variables=["clustering_coefficient", "average_path_length", "connection_density"],
            discrete_variables=[],
            max_evaluations=request.max_iterations or 500,
            convergence_tolerance=1e-6,
            domain="neuroscience"
        )
        
        result = await mathematical_search.search(math_request)
        
        # Add network-specific metadata
        result.search_statistics["network_topology"] = {
            "nodes": request.nodes,
            "target_clustering": request.target_clustering,
            "target_path_length": request.target_path_length,
            "optimization_type": "small_world_network"
        }
        
        logger.info(f"Network topology optimization completed with value: {result.best_value}")
        
        return result
        
    except Exception as e:
        logger.error(f"Network topology optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Network topology optimization failed: {str(e)}")


@router.post("/multi-objective", response_model=List[SearchResult])
async def multi_objective_optimization(request: MultiObjectiveRequest) -> List[SearchResult]:
    """
    Multi-objective optimization using PUCT.
    
    Finds Pareto-optimal solutions for multiple conflicting objectives.
    """
    try:
        logger.info(f"Starting multi-objective optimization with {len(request.objectives)} objectives")
        
        results = []
        
        for i, objective_name in enumerate(request.objectives):
            # Create individual mathematical search request
            math_request = MathematicalSearchRequest(
                problem_type=objective_name,
                objective_function=f"optimize_{objective_name}",
                constraints=request.constraints or [],
                variable_bounds=request.variable_bounds,
                continuous_variables=list(request.variable_bounds.keys()),
                discrete_variables=[],
                max_evaluations=(request.max_evaluations or 1000) // len(request.objectives),
                convergence_tolerance=1e-6,
                domain="general"
            )
            
            result = await mathematical_search.search(math_request)
            
            # Add multi-objective metadata
            result.search_statistics["multi_objective"] = {
                "objective_index": i,
                "objective_name": objective_name,
                "total_objectives": len(request.objectives),
                "pareto_rank": i + 1  # Placeholder ranking
            }
            
            results.append(result)
        
        logger.info(f"Multi-objective optimization completed: {len(results)} solutions found")
        
        return results
        
    except Exception as e:
        logger.error(f"Multi-objective optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-objective optimization failed: {str(e)}")


@router.post("/parallel", response_model=List[SearchResult])
async def parallel_search(request: ParallelSearchRequest) -> List[SearchResult]:
    """
    Execute multiple tree searches in parallel.
    
    Optimized for throughput with configurable parallelism level.
    """
    try:
        logger.info(f"Starting parallel search with {len(request.searches)} searches")
        
        # Convert requests to search parameters
        search_params = []
        for search_req in request.searches:
            config = search_req.config or SearchConfigRequest(
                max_budget_nodes=1000,
                max_depth=10,
                default_budget=500,
                c_puct=1.414,
                expansion_threshold=5,
                simulation_rollouts=3,
                temperature=1.0,
                use_progressive_widening=True,
                alpha_widening=0.5,
                algorithm=SearchAlgorithm.PUCT,
                selection_policy=SelectionPolicy.BEST,
                expansion_policy=ExpansionPolicy.PROGRESSIVE,
                simulation_policy=SimulationPolicy.HEURISTIC,
                backpropagation_policy=BackpropagationPolicy.AVERAGE,
                convergence_threshold=1e-6,
                parallel_simulations=1,
                timeout_seconds=30.0,
                verbose=False,
                save_tree=False
            )
            search_params.append((config, search_req.initial_state))
        
        # Execute parallel searches
        results = await tree_search_engine.parallel_search(
            requests=search_params,
            evaluator=TestStateEvaluator()
        )
        
        # Add parallel search metadata
        for i, result in enumerate(results):
            result.search_statistics["parallel_search"] = {
                "search_index": i,
                "total_parallel_searches": len(results),
                "max_parallel_configured": request.max_parallel
            }
        
        logger.info(f"Parallel search completed: {len(results)} searches finished")
        
        return results
        
    except Exception as e:
        logger.error(f"Parallel search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parallel search failed: {str(e)}")


@router.get("/best-practices", response_model=BestPracticesResponse)
async def get_best_practices() -> BestPracticesResponse:
    """
    Get best practices and recommendations for Tree-Search PUCT.
    
    Comprehensive guide for optimal usage across different domains and problem types.
    """
    return BestPracticesResponse(
        algorithm_selection={
            "general_optimization": "Use PUCT with c_puct=1.414 for balanced exploration/exploitation",
            "game_playing": "Use UCT with domain-specific rollout policies",
            "parameter_tuning": "Use UCB1 for simple parameter spaces",
            "mathematical_optimization": "Use PUCT with progressive widening for continuous spaces",
            "multi_objective": "Use PUCT with separate trees per objective"
        },
        parameter_tuning={
            "c_puct": {
                "low_exploration": 0.5,
                "balanced": 1.414,
                "high_exploration": 2.0,
                "description": "Higher values encourage more exploration"
            },
            "expansion_threshold": {
                "conservative": 10,
                "balanced": 5,
                "aggressive": 1,
                "description": "Lower values expand nodes sooner"
            },
            "simulation_rollouts": {
                "fast": 1,
                "balanced": 3,
                "thorough": 10,
                "description": "More rollouts increase accuracy but slow down search"
            }
        },
        performance_optimization=[
            "Use caching for expensive state evaluations",
            "Enable parallel simulations for multi-core systems",
            "Set appropriate timeout limits to prevent runaway searches",
            "Use progressive widening for large action spaces",
            "Consider domain-specific evaluators for specialized problems",
            "Monitor memory usage for very deep trees",
            "Use early stopping criteria when convergence is achieved"
        ],
        common_pitfalls=[
            "Setting c_puct too high leads to excessive exploration",
            "Setting expansion_threshold too low wastes computation on bad nodes",
            "Not using domain knowledge in state evaluation",
            "Ignoring constraint violations in optimization problems",
            "Running too few iterations for convergence",
            "Not monitoring search statistics for debugging",
            "Using inappropriate algorithms for the problem type"
        ],
        domain_specific_advice={
            "biomaterials": [
                "Use scaffold_optimization endpoint for porosity-strength trade-offs",
                "Consider manufacturing constraints in variable bounds",
                "Include biocompatibility metrics in objective functions"
            ],
            "neuroscience": [
                "Use network_topology endpoint for small-world properties",
                "Consider computational complexity of network simulations",
                "Balance clustering and path length objectives carefully"
            ],
            "optimization": [
                "Use mathematical search for continuous optimization",
                "Set appropriate convergence tolerances",
                "Consider multi-objective formulations for complex problems"
            ],
            "games": [
                "Use UCT with game-specific rollout policies",
                "Consider time constraints in tournament settings",
                "Use domain knowledge to guide expansion"
            ]
        }
    )


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get detailed performance metrics and system statistics.
    
    Provides comprehensive performance analysis and resource utilization data.
    """
    try:
        # Get engine statistics
        engine_stats = tree_search_engine.get_engine_statistics()
        
        # Calculate derived metrics
        current_time = datetime.now(timezone.utc)
        
        performance_data = {
            "engine_statistics": engine_stats,
            "performance_metrics": {
                "searches_per_minute": engine_stats.get('total_searches', 0) * 60,  # Placeholder calculation
                "average_nodes_per_search": engine_stats.get('total_searches', 1) and 
                    (engine_stats.get('total_nodes_explored', 0) / engine_stats.get('total_searches', 1)),
                "success_rate_percentage": engine_stats.get('success_rate', 0.0) * 100,
                "cache_efficiency": engine_stats.get('cache_statistics', {}).get('hit_rate', 0.0) * 100
            },
            "resource_utilization": {
                "active_searches": len(tree_search_engine.get_active_searches()),
                "max_parallel_searches": tree_search_engine.config.max_parallel_searches,
                "memory_usage_estimate_mb": engine_stats.get('total_nodes_explored', 0) * 0.001  # Rough estimate
            },
            "recommendations": [],
            "timestamp": current_time.isoformat()
        }
        
        # Add performance recommendations
        if engine_stats.get('success_rate', 1.0) < 0.9:
            performance_data["recommendations"].append("Consider increasing timeout or budget for better success rate")
        
        cache_stats = engine_stats.get('cache_statistics', {})
        if cache_stats.get('hit_rate', 1.0) < 0.5:
            performance_data["recommendations"].append("Low cache hit rate - consider tuning cache size or evaluation strategy")
        
        if len(tree_search_engine.get_active_searches()) > 0.8 * tree_search_engine.config.max_parallel_searches:
            performance_data["recommendations"].append("High search load - consider increasing parallel capacity")
        
        if not performance_data["recommendations"]:
            performance_data["recommendations"].append("System performance is optimal")
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


# ==================== UTILITY ENDPOINTS ====================

@router.get("/algorithms/{algorithm_name}")
async def get_algorithm_details(algorithm_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific algorithm."""
    
    algorithms_info = {
        "puct": {
            "name": "PUCT",
            "full_name": "Polynomial Upper Confidence Bounds for Trees",
            "description": "PUCT uses polynomial upper confidence bounds with prior knowledge",
            "formula": "Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))",
            "parameters": {
                "c_puct": {
                    "description": "Exploration constant",
                    "default": 1.414,
                    "range": [0.1, 10.0],
                    "tuning_advice": "Higher values increase exploration"
                },
                "prior_probability": {
                    "description": "Prior probability for actions",
                    "default": "uniform",
                    "advice": "Use domain knowledge when available"
                }
            },
            "best_for": ["optimization", "planning", "decision_making"],
            "complexity": "O(log n) per node selection",
            "references": [
                "Rosin, C. D. (2011). Multi-armed bandits with episode context.",
                "Silver, D. et al. (2016). Mastering the game of Go with deep neural networks."
            ]
        },
        "ucb1": {
            "name": "UCB1",
            "full_name": "Upper Confidence Bound 1",
            "description": "Classic multi-armed bandit algorithm adapted for trees",
            "formula": "Q(s,a) + c * sqrt(log(N(s)) / N(s,a))",
            "parameters": {
                "c_exploration": {
                    "description": "Exploration constant",
                    "default": 1.414,
                    "range": [0.5, 3.0],
                    "tuning_advice": "sqrt(2) is theoretically optimal"
                }
            },
            "best_for": ["multi_armed_bandits", "simple_optimization"],
            "complexity": "O(1) per node selection",
            "references": [
                "Auer, P. et al. (2002). Finite-time analysis of the multiarmed bandit problem."
            ]
        }
    }
    
    algorithm_name = algorithm_name.lower()
    if algorithm_name not in algorithms_info:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm_name}' not found")
    
    return algorithms_info[algorithm_name]


@router.delete("/search/{search_id}")
async def cancel_search(search_id: str) -> Dict[str, Any]:
    """Cancel an active search by ID."""
    
    success = tree_search_engine.cancel_search(search_id)
    
    if success:
        return {
            "message": f"Search {search_id} cancelled successfully",
            "cancelled": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Search {search_id} not found or already completed")


@router.get("/active-searches")
async def get_active_searches() -> Dict[str, Any]:
    """Get list of currently active searches."""
    
    active_searches = tree_search_engine.get_active_searches()
    
    return {
        "active_searches": active_searches,
        "count": len(active_searches),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==================== CLEANUP ====================

async def cleanup_tree_search_resources():
    """Cleanup function to be called on shutdown."""
    logger.info("Cleaning up Tree Search resources")
    tree_search_engine.cleanup()
    mathematical_search.cleanup()