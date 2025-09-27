"""Tree Search Engine - Complete MCTS Engine

Engine completo para Tree Search PUCT com suporte a algoritmos MCTS avançados,
busca paralela, otimização multi-objetivo e domínios especializados.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from .puct_algorithm import (
    PUCTAlgorithm,
    PUCTNode,
    StateEvaluator,
    TestStateEvaluator,
)
from ..models.tree_search_models import (
    SearchConfigRequest,
    SearchResult,
    SearchState,
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

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION MANAGEMENT ====================

@dataclass
class TreeSearchEngineConfig:
    """Configuração do Tree Search Engine."""
    
    # Core Algorithm Settings
    default_algorithm: SearchAlgorithm = SearchAlgorithm.PUCT
    default_iterations: int = 1000
    default_timeout: float = 30.0
    
    # Parallel Processing
    max_parallel_searches: int = 4
    enable_parallel_expansion: bool = True
    
    # Memory Management
    max_tree_nodes: int = 100000
    node_pruning_threshold: int = 50000
    enable_garbage_collection: bool = True
    
    # Performance Optimization
    cache_evaluations: bool = True
    cache_size: int = 10000
    use_jit_compilation: bool = False
    
    # Logging and Debugging
    enable_detailed_logging: bool = False
    save_search_trees: bool = False
    log_interval: int = 100


# ==================== EVALUATION CACHE ====================

class EvaluationCache:
    """Cache para avaliações de estado."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, datetime]] = {}
        self.access_count: Dict[str, int] = {}
    
    def get_key(self, state: Any) -> str:
        """Gera chave única para o estado."""
        return str(hash(str(state)))
    
    def get(self, state: Any) -> Optional[float]:
        """Recupera avaliação do cache."""
        key = self.get_key(state)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key][0]
        return None
    
    def put(self, state: Any, value: float):
        """Armazena avaliação no cache."""
        key = self.get_key(state)
        
        # Remove items antigos se necessário
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = (value, datetime.now(timezone.utc))
        self.access_count[key] = 1
    
    def _evict_lru(self):
        """Remove item menos recentemente usado."""
        if not self.cache:
            return
        
        # Find least recently used item
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def clear(self):
        """Limpa cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_count.values()) / max(1, len(self.access_count)),
        }


# ==================== CACHED EVALUATOR WRAPPER ====================

class CachedStateEvaluator(StateEvaluator):
    """Wrapper que adiciona cache a um StateEvaluator."""
    
    def __init__(self, evaluator: StateEvaluator, cache: Optional[EvaluationCache] = None):
        self.evaluator = evaluator
        self.cache = cache or EvaluationCache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def get_available_actions(self, state: Any) -> List[Tuple[str, float]]:
        """Delega para o evaluator base."""
        return await self.evaluator.get_available_actions(state)
    
    async def apply_action(self, state: Any, action: str) -> Tuple[Any, float, bool]:
        """Delega para o evaluator base."""
        return await self.evaluator.apply_action(state, action)
    
    async def evaluate_state(self, state: Any) -> float:
        """Avalia estado com cache."""
        # Try cache first
        cached_value = self.cache.get(state)
        if cached_value is not None:
            self.cache_hits += 1
            return cached_value
        
        # Evaluate and cache
        value = await self.evaluator.evaluate_state(state)
        self.cache.put(state, value)
        self.cache_misses += 1
        return value
    
    async def simulate_rollout(self, state: Any, max_depth: int = 10) -> float:
        """Delega para o evaluator base (rollouts não são cached)."""
        return await self.evaluator.simulate_rollout(state, max_depth)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            **self.cache.get_stats()
        }


# ==================== TREE SEARCH ENGINE ====================

class TreeSearchEngine:
    """Engine principal para Tree Search PUCT."""
    
    def __init__(self, config: Optional[TreeSearchEngineConfig] = None):
        self.config = config or TreeSearchEngineConfig()
        
        # Active searches
        self.active_searches: Dict[str, PUCTAlgorithm] = {}
        
        # Performance tracking
        self.total_searches = 0
        self.successful_searches = 0
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_searches)
        
        # Caches
        self.evaluation_cache = EvaluationCache(self.config.cache_size)
        
        logger.info(f"Tree Search Engine initialized with config: {self.config}")
    
    async def search(
        self,
        request: SearchConfigRequest,
        initial_state: Union[str, Dict[str, Any]],
        evaluator: Optional[StateEvaluator] = None
    ) -> SearchResult:
        """Executa busca Tree Search completa."""
        
        search_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting search {search_id} with algorithm {request.algorithm}")
        
        try:
            # Create evaluator
            if evaluator is None:
                evaluator = TestStateEvaluator()
            
            # Add cache wrapper if enabled
            if self.config.cache_evaluations:
                evaluator = CachedStateEvaluator(evaluator, self.evaluation_cache)
            
            # Convert request to PUCT config
            puct_config = self._create_puct_config(request)
            
            # Create algorithm
            algorithm = PUCTAlgorithm(
                evaluator=evaluator,
                config=puct_config,
                algorithm=request.algorithm or SearchAlgorithm.PUCT
            )
            
            # Store active search
            self.active_searches[search_id] = algorithm
            
            # Execute search - convert initial_state to string if needed for compatibility
            state_for_search = str(initial_state) if not isinstance(initial_state, str) else initial_state
            root_node = await algorithm.search(
                initial_state=state_for_search,
                max_iterations=request.default_budget or 1000,
                timeout_seconds=request.timeout_seconds or 30.0
            )
            
            # Create result
            result = await self._create_search_result(
                search_id, algorithm, root_node, request, start_time
            )
            
            self.successful_searches += 1
            logger.info(f"Search {search_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Search {search_id} failed: {e}")
            end_time = datetime.now(timezone.utc)
            
            # Return error result
            return SearchResult(
                search_id=search_id,
                algorithm=request.algorithm or SearchAlgorithm.PUCT,
                config=request,
                root_node=TreeNode(
                    node_id="error_root",
                    state=initial_state,
                    action_taken=None,
                    parent_id=None,
                    depth=0,
                    visits=0,
                    total_value=0.0,
                    mean_value=0.0,
                    variance=0.0,
                    puct_score=None,
                    prior_probability=1.0,
                    status=NodeStatus.UNVISITED,
                    is_terminal=True,
                    children_ids=[],
                    created_at=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc)
                ),
                best_action_sequence=[],
                best_value=0.0,
                search_tree=None,
                nodes_explored=0,
                max_depth_reached=0,
                total_iterations=0,
                start_time=start_time,
                end_time=end_time,
                total_time_seconds=(end_time - start_time).total_seconds(),
                converged=False,
                convergence_iteration=None,
                search_statistics={},
                success=False,
                error_message=str(e)
            )
            
        finally:
            # Cleanup
            if search_id in self.active_searches:
                algorithm = self.active_searches.pop(search_id)
                algorithm.cleanup()
            self.total_searches += 1
    
    async def parallel_search(
        self,
        requests: List[Tuple[SearchConfigRequest, Union[str, Dict[str, Any]]]],
        evaluator: Optional[StateEvaluator] = None
    ) -> List[SearchResult]:
        """Executa múltiplas buscas em paralelo."""
        
        logger.info(f"Starting parallel search with {len(requests)} requests")
        
        # Create tasks
        tasks = [
            self.search(request, initial_state, evaluator)
            for request, initial_state in requests
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel search {i} failed: {result}")
                # Create error result
                request, initial_state = requests[i]
                error_result = SearchResult(
                    search_id=f"parallel_error_{i}",
                    algorithm=request.algorithm or SearchAlgorithm.PUCT,
                    config=request,
                    root_node=TreeNode(
                        node_id=f"error_root_{i}",
                        state=initial_state,
                        action_taken=None,
                        parent_id=None,
                        depth=0,
                        visits=0,
                        total_value=0.0,
                        mean_value=0.0,
                        variance=0.0,
                        puct_score=None,
                        prior_probability=1.0,
                        status=NodeStatus.UNVISITED,
                        is_terminal=True,
                        children_ids=[],
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc)
                    ),
                    best_action_sequence=[],
                    best_value=0.0,
                    search_tree=None,
                    nodes_explored=0,
                    max_depth_reached=0,
                    total_iterations=0,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    total_time_seconds=0.0,
                    converged=False,
                    convergence_iteration=None,
                    search_statistics={},
                    success=False,
                    error_message=str(result)
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def mathematical_search(
        self,
        request: MathematicalSearchRequest,
        evaluator: Optional[StateEvaluator] = None
    ) -> SearchResult:
        """Executa busca matemática especializada."""
        
        # Convert mathematical request to standard search request
        search_request = SearchConfigRequest(
            max_budget_nodes=request.max_evaluations,
            max_depth=10,
            default_budget=min(request.max_evaluations, 1000),
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
            convergence_threshold=request.convergence_tolerance,
            parallel_simulations=1,
            timeout_seconds=30.0,
            verbose=True,
            save_tree=False
        )
        
        # Use mathematical state representation
        initial_state = {
            'problem_type': request.problem_type,
            'objective_function': request.objective_function,
            'variable_bounds': request.variable_bounds,
            'constraints': request.constraints,
            'domain': request.domain
        }
        
        # Create specialized evaluator if not provided
        if evaluator is None:
            evaluator = self._create_mathematical_evaluator(request)
        
        return await self.search(search_request, initial_state, evaluator)
    
    def _create_puct_config(self, request: SearchConfigRequest) -> PUCTConfig:
        """Cria configuração PUCT a partir do request."""
        return PUCTConfig(
            c_puct=request.c_puct or 1.414,
            use_virtual_loss=(request.parallel_simulations or 1) > 1,
            virtual_loss_value=1.0,
            expansion_threshold=request.expansion_threshold or 5,
            max_children=None,
            use_progressive_widening=request.use_progressive_widening or True,
            alpha_widening=request.alpha_widening or 0.5,
            use_prior=True,
            prior_weight=1.0,
            value_function="mean",
            backup_operator="average"
        )
    
    async def _create_search_result(
        self,
        search_id: str,
        algorithm: PUCTAlgorithm,
        root_node: PUCTNode,
        request: SearchConfigRequest,
        start_time: datetime
    ) -> SearchResult:
        """Cria resultado da busca."""
        
        end_time = datetime.now(timezone.utc)
        
        # Convert PUCTNode to TreeNode
        tree_root = TreeNode(
            node_id=root_node.node_id,
            state=root_node.state,
            action_taken=root_node.action_taken,
            parent_id=root_node.parent.node_id if root_node.parent else None,
            depth=root_node.depth,
            visits=root_node.visits,
            total_value=root_node.total_value,
            mean_value=root_node.mean_value,
            variance=root_node.value_variance,
            puct_score=None,
            prior_probability=root_node.prior_probability,
            status=root_node.status,
            is_terminal=root_node.is_terminal,
            children_ids=list(root_node.children.keys()),
            created_at=root_node.created_at,
            last_updated=root_node.last_updated
        )
        
        # Get best action sequence
        best_sequence = algorithm.get_best_action_sequence()
        best_value = root_node.mean_value
        
        # Get statistics
        stats = algorithm.get_search_statistics()
        
        # Add cache statistics if available
        if isinstance(algorithm.evaluator, CachedStateEvaluator):
            stats['cache_statistics'] = algorithm.evaluator.get_cache_stats()
        
        return SearchResult(
            search_id=search_id,
            algorithm=request.algorithm or SearchAlgorithm.PUCT,
            config=request,
            root_node=tree_root,
            best_action_sequence=best_sequence,
            best_value=best_value,
            search_tree=None,  # Optionally save full tree
            nodes_explored=algorithm.nodes_explored,
            max_depth_reached=stats.get('max_tree_depth', 0),
            total_iterations=algorithm.iterations_completed,
            start_time=start_time,
            end_time=end_time,
            total_time_seconds=(end_time - start_time).total_seconds(),
            converged=stats.get('converged', False),
            convergence_iteration=stats.get('convergence_iteration'),
            search_statistics=stats,
            success=True,
            error_message=None
        )
    
    def _create_mathematical_evaluator(self, request: MathematicalSearchRequest) -> StateEvaluator:
        """Cria evaluator matemático especializado."""
        
        class MathematicalEvaluator(StateEvaluator):
            """Evaluator para problemas matemáticos."""
            
            def __init__(self, math_request: MathematicalSearchRequest):
                self.request = math_request
                self.evaluation_count = 0
            
            async def get_available_actions(self, state: Dict[str, Any]) -> List[Tuple[str, float]]:
                """Retorna ações matemáticas disponíveis."""
                actions = []
                
                # Variable optimization actions
                for var_name in self.request.variable_bounds.keys():
                    actions.extend([
                        (f"increase_{var_name}", 0.5),
                        (f"decrease_{var_name}", 0.5),
                        (f"optimize_{var_name}", 0.8),
                    ])
                
                # Global optimization actions
                actions.extend([
                    ("refine_solution", 0.9),
                    ("explore_neighborhood", 0.7),
                    ("gradient_step", 0.6),
                    ("random_perturbation", 0.3),
                ])
                
                return actions
            
            async def apply_action(self, state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], float, bool]:
                """Aplica ação matemática."""
                new_state = state.copy()
                
                # Simple action simulation
                if "increase_" in action:
                    var_name = action.replace("increase_", "")
                    new_state[f"{var_name}_trend"] = "increasing"
                elif "decrease_" in action:
                    var_name = action.replace("decrease_", "")
                    new_state[f"{var_name}_trend"] = "decreasing"
                elif "optimize_" in action:
                    var_name = action.replace("optimize_", "")
                    new_state[f"{var_name}_optimized"] = True
                
                new_state["action_history"] = state.get("action_history", []) + [action]
                
                # Simulate reward (in real implementation, evaluate objective function)
                reward = np.random.random() * 0.1  # Small random reward
                
                # Terminal condition (max evaluations)
                is_terminal = len(new_state.get("action_history", [])) > 10
                
                return new_state, reward, is_terminal
            
            async def evaluate_state(self, state: Dict[str, Any]) -> float:
                """Avalia estado matemático."""
                self.evaluation_count += 1
                
                # Simple evaluation (in real implementation, evaluate objective function)
                score = 0.5 + np.random.random() * 0.5
                
                # Bonus for optimization actions
                if any("optimized" in key for key in state.keys()):
                    score += 0.2
                
                return score
            
            async def simulate_rollout(self, state: Dict[str, Any], max_depth: int = 5) -> float:
                """Simula rollout matemático."""
                total_value = 0.0
                current_state = state
                
                for _ in range(max_depth):
                    actions = await self.get_available_actions(current_state)
                    if not actions:
                        break
                    
                    # Select random action
                    action, _ = np.random.choice(actions)
                    new_state, reward, is_terminal = await self.apply_action(current_state, action)
                    
                    total_value += reward
                    current_state = new_state
                    
                    if is_terminal:
                        break
                
                return total_value / max_depth
        
        return MathematicalEvaluator(request)
    
    def get_active_searches(self) -> List[str]:
        """Retorna IDs das buscas ativas."""
        return list(self.active_searches.keys())
    
    def cancel_search(self, search_id: str) -> bool:
        """Cancela busca ativa."""
        if search_id in self.active_searches:
            algorithm = self.active_searches.pop(search_id)
            algorithm.cleanup()
            logger.info(f"Search {search_id} cancelled")
            return True
        return False
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine."""
        return {
            'total_searches': self.total_searches,
            'successful_searches': self.successful_searches,
            'success_rate': self.successful_searches / max(1, self.total_searches),
            'active_searches': len(self.active_searches),
            'cache_statistics': self.evaluation_cache.get_stats(),
            'config': {
                'max_parallel_searches': self.config.max_parallel_searches,
                'cache_enabled': self.config.cache_evaluations,
                'max_tree_nodes': self.config.max_tree_nodes,
            }
        }
    
    def cleanup(self):
        """Limpa recursos do engine."""
        # Cancel all active searches
        for search_id in list(self.active_searches.keys()):
            self.cancel_search(search_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear caches
        self.evaluation_cache.clear()
        
        logger.info("Tree Search Engine cleaned up")


# ==================== UTILITY FUNCTIONS ====================

def create_default_engine() -> TreeSearchEngine:
    """Cria engine com configuração padrão."""
    return TreeSearchEngine()


def create_high_performance_engine() -> TreeSearchEngine:
    """Cria engine otimizado para performance."""
    config = TreeSearchEngineConfig(
        max_parallel_searches=8,
        enable_parallel_expansion=True,
        cache_evaluations=True,
        cache_size=50000,
        max_tree_nodes=500000,
        use_jit_compilation=True,
        enable_detailed_logging=False
    )
    return TreeSearchEngine(config)


def create_debug_engine() -> TreeSearchEngine:
    """Cria engine para debugging."""
    config = TreeSearchEngineConfig(
        max_parallel_searches=1,
        enable_detailed_logging=True,
        save_search_trees=True,
        log_interval=10,
        cache_evaluations=False  # Disable cache for debugging
    )
    return TreeSearchEngine(config)