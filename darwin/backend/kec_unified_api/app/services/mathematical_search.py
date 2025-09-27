"""Mathematical Search - Specialized PUCT for Mathematical Optimization

Implementação especializada de busca matemática usando PUCT para domínios científicos:
- Biomaterials: Otimização de scaffolds, porosidade, resistência mecânica
- Neuroscience: Redes neurais, conectividade, small-world networks
- Philosophy: Sistemas lógicos, inferência, espaços de prova
- Quantum: Estados quânticos, circuitos, espaço de Hilbert
- Psychiatry: Modelos comportamentais, terapias, parâmetros
"""

from __future__ import annotations

import math
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

from .puct_algorithm import StateEvaluator, PUCTAlgorithm
from .tree_search_engine import TreeSearchEngine, TreeSearchEngineConfig
from ..models.tree_search_models import (
    MathematicalSearchRequest,
    OptimizationProblem,
    ConstraintSystem,
    ObjectiveFunction,
    SearchConfigRequest,
    SearchResult,
    PUCTConfig,
    SearchAlgorithm,
    SelectionPolicy,
    ExpansionPolicy,
    SimulationPolicy,
    BackpropagationPolicy,
)

logger = logging.getLogger(__name__)

# ==================== MATHEMATICAL DOMAINS ====================

class MathematicalDomain(str, Enum):
    """Domínios matemáticos especializados."""
    BIOMATERIALS = "biomaterials"
    NEUROSCIENCE = "neuroscience"
    PHILOSOPHY = "philosophy"
    QUANTUM_MECHANICS = "quantum_mechanics"
    PSYCHIATRY = "psychiatry"
    GENERAL = "general"


class OptimizationType(str, Enum):
    """Tipos de otimização."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"


# ==================== MATHEMATICAL STATE REPRESENTATION ====================

@dataclass
class MathematicalState:
    """Estado matemático para busca PUCT."""
    
    # Core state
    variables: Dict[str, float] = field(default_factory=dict)
    objective_value: Optional[float] = None
    constraint_violations: List[str] = field(default_factory=list)
    
    # Optimization history
    iteration: int = 0
    improvement_history: List[float] = field(default_factory=list)
    
    # Domain-specific data
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    domain_data: Dict[str, Any] = field(default_factory=dict)
    
    # Search metadata
    feasible: bool = True
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None
    
    def __str__(self) -> str:
        """String representation for hashing."""
        var_items = []
        for k, v in sorted(self.variables.items()):
            if v is not None and isinstance(v, (int, float)):
                var_items.append(f"{k}:{float(v):.4f}")
            else:
                var_items.append(f"{k}:None")
        var_str = ",".join(var_items)
        
        if self.objective_value is not None and isinstance(self.objective_value, (int, float)):
            obj_str = f"{float(self.objective_value):.4f}"
        else:
            obj_str = "None"
        
        return f"MathState({var_str},obj:{obj_str})"
    
    def copy(self) -> MathematicalState:
        """Creates a copy of the state."""
        return MathematicalState(
            variables=self.variables.copy(),
            objective_value=self.objective_value,
            constraint_violations=self.constraint_violations.copy(),
            iteration=self.iteration,
            improvement_history=self.improvement_history.copy(),
            domain=self.domain,
            domain_data=self.domain_data.copy(),
            feasible=self.feasible,
            pareto_rank=self.pareto_rank,
            crowding_distance=self.crowding_distance
        )


# ==================== MATHEMATICAL OBJECTIVE FUNCTIONS ====================

class MathematicalObjective(ABC):
    """Interface para funções objetivo matemáticas."""
    
    @abstractmethod
    def evaluate(self, variables: Dict[str, float]) -> float:
        """Avalia função objetivo."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Retorna limites das variáveis."""
        pass
    
    @abstractmethod
    def is_feasible(self, variables: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Verifica se solução é viável."""
        pass
    
    def get_gradient(self, variables: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Retorna gradiente se disponível."""
        return None


# ==================== BIOMATERIALS OBJECTIVES ====================

class ScaffoldOptimizationObjective(MathematicalObjective):
    """Otimização de scaffolds biomateriais."""
    
    def __init__(
        self,
        target_porosity: float = 0.7,
        min_mechanical_strength: float = 5.0,
        target_pore_size: float = 200.0  # micrometers
    ):
        self.target_porosity = target_porosity
        self.min_mechanical_strength = min_mechanical_strength
        self.target_pore_size = target_pore_size
    
    def evaluate(self, variables: Dict[str, float]) -> float:
        """
        Otimiza scaffold considerando:
        - Porosidade próxima ao target
        - Resistência mecânica mínima
        - Tamanho de poro adequado
        """
        porosity = variables.get('porosity', 0.5)
        strength = variables.get('mechanical_strength', 1.0)
        pore_size = variables.get('pore_size', 100.0)
        
        # Penalidades por desvio dos targets
        porosity_penalty = abs(porosity - self.target_porosity) ** 2
        strength_penalty = max(0, self.min_mechanical_strength - strength) ** 2
        pore_size_penalty = abs(pore_size - self.target_pore_size) ** 2
        
        # Função objetivo (minimização de penalidades)
        objective = porosity_penalty + 0.5 * strength_penalty + 0.1 * pore_size_penalty
        
        return -objective  # Negativo para maximização de qualidade
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'porosity': (0.1, 0.95),
            'mechanical_strength': (1.0, 50.0),
            'pore_size': (50.0, 500.0)
        }
    
    def is_feasible(self, variables: Dict[str, float]) -> Tuple[bool, List[str]]:
        violations = []
        
        porosity = variables.get('porosity', 0.5)
        strength = variables.get('mechanical_strength', 1.0)
        
        if porosity > 0.9:
            violations.append("Porosity too high - structural integrity risk")
        
        if strength < self.min_mechanical_strength:
            violations.append(f"Mechanical strength below minimum {self.min_mechanical_strength}")
        
        # Trade-off constraint: very high porosity reduces strength
        if porosity > 0.8 and strength < 10.0:
            violations.append("High porosity requires higher mechanical strength")
        
        return len(violations) == 0, violations


# ==================== NEUROSCIENCE OBJECTIVES ====================

class NetworkTopologyObjective(MathematicalObjective):
    """Otimização de topologia de rede neural (small-world networks)."""
    
    def __init__(
        self,
        target_clustering: float = 0.6,
        target_path_length: float = 2.5,
        num_nodes: int = 100
    ):
        self.target_clustering = target_clustering
        self.target_path_length = target_path_length
        self.num_nodes = num_nodes
    
    def evaluate(self, variables: Dict[str, float]) -> float:
        """
        Otimiza rede para propriedades small-world:
        - Alto clustering coefficient
        - Baixo average path length
        - Balanceamento adequado
        """
        clustering = variables.get('clustering_coefficient', 0.3)
        path_length = variables.get('average_path_length', 5.0)
        connection_density = variables.get('connection_density', 0.1)
        
        # Small-world coefficient: alta clusterização, baixo caminho médio
        clustering_score = 1.0 - abs(clustering - self.target_clustering)
        path_score = 1.0 - abs(path_length - self.target_path_length) / 10.0
        
        # Penalidade por densidade muito baixa ou muito alta
        density_penalty = abs(connection_density - 0.15) ** 2
        
        # Small-world metric
        small_world_coefficient = clustering / max(path_length, 0.1)
        
        objective = 0.4 * clustering_score + 0.4 * path_score + 0.2 * small_world_coefficient - density_penalty
        
        return objective
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'clustering_coefficient': (0.05, 0.95),
            'average_path_length': (1.5, 10.0),
            'connection_density': (0.05, 0.5)
        }
    
    def is_feasible(self, variables: Dict[str, float]) -> Tuple[bool, List[str]]:
        violations = []
        
        clustering = variables.get('clustering_coefficient', 0.3)
        path_length = variables.get('average_path_length', 5.0)
        density = variables.get('connection_density', 0.1)
        
        if density < 0.05:
            violations.append("Connection density too low - network disconnection risk")
        
        if clustering < 0.1:
            violations.append("Clustering coefficient too low")
        
        if path_length > 8.0:
            violations.append("Average path length too high - poor communication")
        
        return len(violations) == 0, violations


# ==================== PHILOSOPHY OBJECTIVES ====================

class LogicalSystemObjective(MathematicalObjective):
    """Otimização de sistemas lógicos e inferência."""
    
    def __init__(self, target_consistency: float = 0.95, target_completeness: float = 0.85):
        self.target_consistency = target_consistency
        self.target_completeness = target_completeness
    
    def evaluate(self, variables: Dict[str, float]) -> float:
        """
        Otimiza sistema lógico para:
        - Alta consistência (sem contradições)
        - Boa completeness (poder de derivação)
        - Eficiência computacional
        """
        consistency = variables.get('consistency_score', 0.8)
        completeness = variables.get('completeness_score', 0.7)
        computational_efficiency = variables.get('efficiency', 0.5)
        
        consistency_score = 1.0 - abs(consistency - self.target_consistency)
        completeness_score = 1.0 - abs(completeness - self.target_completeness)
        
        # Balanceamento entre precisão e eficiência
        objective = 0.5 * consistency_score + 0.3 * completeness_score + 0.2 * computational_efficiency
        
        return objective
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'consistency_score': (0.5, 1.0),
            'completeness_score': (0.3, 1.0),
            'efficiency': (0.1, 1.0)
        }
    
    def is_feasible(self, variables: Dict[str, float]) -> Tuple[bool, List[str]]:
        violations = []
        
        consistency = variables.get('consistency_score', 0.8)
        
        if consistency < 0.7:
            violations.append("Consistency too low - logical contradictions risk")
        
        return len(violations) == 0, violations


# ==================== MATHEMATICAL STATE EVALUATOR ====================

class MathematicalStateEvaluator(StateEvaluator[MathematicalState]):
    """Evaluator especializado para estados matemáticos."""
    
    def __init__(
        self,
        objective: MathematicalObjective,
        domain: MathematicalDomain = MathematicalDomain.GENERAL
    ):
        self.objective = objective
        self.domain = domain
        self.bounds = objective.get_bounds()
        self.evaluation_count = 0
        
        # Otimização local cache
        self.local_optima_cache: List[Tuple[Dict[str, float], float]] = []
        
    async def get_available_actions(self, state: MathematicalState) -> List[Tuple[str, float]]:
        """Retorna ações matemáticas disponíveis."""
        actions = []
        
        # Variable optimization actions
        for var_name in self.bounds.keys():
            current_value = state.variables.get(var_name, 0.5)
            bounds = self.bounds[var_name]
            
            # Step sizes based on bound range
            step_size = (bounds[1] - bounds[0]) * 0.1
            
            actions.extend([
                (f"increase_{var_name}_small", 0.6),
                (f"decrease_{var_name}_small", 0.6),
                (f"increase_{var_name}_large", 0.4),
                (f"decrease_{var_name}_large", 0.4),
                (f"optimize_{var_name}_local", 0.8),
            ])
        
        # Global optimization actions
        actions.extend([
            ("gradient_step", 0.7),
            ("random_perturbation", 0.3),
            ("local_search", 0.9),
            ("explore_boundary", 0.5),
            ("refine_solution", 0.8),
            ("escape_local_minimum", 0.4),
        ])
        
        # Domain-specific actions
        if self.domain == MathematicalDomain.BIOMATERIALS:
            actions.extend([
                ("optimize_porosity_strength_tradeoff", 0.9),
                ("explore_pore_size_distribution", 0.7),
                ("balance_mechanical_properties", 0.8),
            ])
        elif self.domain == MathematicalDomain.NEUROSCIENCE:
            actions.extend([
                ("optimize_small_world_properties", 0.9),
                ("balance_clustering_path_length", 0.8),
                ("explore_network_density", 0.6),
            ])
        
        return actions
    
    async def apply_action(
        self, 
        state: MathematicalState, 
        action: str
    ) -> Tuple[MathematicalState, float, bool]:
        """Aplica ação matemática ao estado."""
        
        new_state = state.copy()
        new_state.iteration += 1
        
        # Parse action
        if "_" in action and any(var in action for var in self.bounds.keys()):
            # Variable modification action
            reward = await self._apply_variable_action(new_state, action)
        elif action.startswith("gradient"):
            reward = await self._apply_gradient_step(new_state)
        elif action.startswith("local_search"):
            reward = await self._apply_local_search(new_state)
        elif action.startswith("random"):
            reward = await self._apply_random_perturbation(new_state)
        elif "optimize" in action and "tradeoff" in action:
            reward = await self._apply_domain_specific_optimization(new_state, action)
        else:
            # Default: small random perturbation
            reward = await self._apply_random_perturbation(new_state, scale=0.05)
        
        # Evaluate new state
        new_objective = self.objective.evaluate(new_state.variables)
        new_state.objective_value = new_objective
        
        # Check feasibility
        is_feasible, violations = self.objective.is_feasible(new_state.variables)
        new_state.feasible = is_feasible
        new_state.constraint_violations = violations
        
        # Update improvement history
        if state.objective_value is not None:
            improvement = new_objective - state.objective_value
            new_state.improvement_history.append(improvement)
        
        # Terminal conditions
        is_terminal = (
            new_state.iteration > 50 or  # Max iterations
            len(new_state.improvement_history) > 10 and 
            all(abs(imp) < 1e-6 for imp in new_state.improvement_history[-10:])  # Convergence
        )
        
        # Reward includes improvement and feasibility
        final_reward = reward
        if not is_feasible:
            final_reward -= 10.0  # Heavy penalty for infeasibility
        
        self.evaluation_count += 1
        
        return new_state, final_reward, is_terminal
    
    async def _apply_variable_action(self, state: MathematicalState, action: str) -> float:
        """Aplica ação de modificação de variável."""
        
        # Parse variable name and modification type
        var_name = None
        for var in self.bounds.keys():
            if var in action:
                var_name = var
                break
        
        if not var_name:
            return 0.0
        
        current_value = state.variables.get(var_name, 0.5)
        bounds = self.bounds[var_name]
        range_size = bounds[1] - bounds[0]
        
        if "increase" in action:
            if "small" in action:
                delta = range_size * 0.05
            else:  # large
                delta = range_size * 0.2
            new_value = min(bounds[1], current_value + delta)
        elif "decrease" in action:
            if "small" in action:
                delta = range_size * 0.05
            else:  # large
                delta = range_size * 0.2
            new_value = max(bounds[0], current_value - delta)
        elif "optimize" in action:
            # Local optimization around current value
            new_value = await self._local_optimize_variable(state, var_name)
        else:
            new_value = current_value
        
        state.variables[var_name] = new_value
        
        # Reward based on expected improvement direction
        return 1.0 if new_value != current_value else 0.0
    
    async def _apply_gradient_step(self, state: MathematicalState) -> float:
        """Aplica passo de gradiente."""
        
        gradient = self.objective.get_gradient(state.variables)
        
        if gradient is None:
            # Finite differences approximation
            gradient = await self._compute_finite_differences_gradient(state.variables)
        
        # Apply gradient step
        step_size = 0.1
        for var_name, grad_value in gradient.items():
            if var_name in self.bounds:
                bounds = self.bounds[var_name]
                current_value = state.variables.get(var_name, 0.5)
                
                # Gradient ascent (maximization)
                new_value = current_value + step_size * grad_value
                new_value = max(bounds[0], min(bounds[1], new_value))
                
                state.variables[var_name] = new_value
        
        return 2.0  # Higher reward for gradient-based improvement
    
    async def _apply_local_search(self, state: MathematicalState) -> float:
        """Aplica busca local."""
        
        # Use scipy.optimize for local search
        x0 = [state.variables.get(var, 0.5) for var in self.bounds.keys()]
        bounds_list = [self.bounds[var] for var in self.bounds.keys()]
        
        def objective_func(x):
            variables = dict(zip(self.bounds.keys(), x))
            return -self.objective.evaluate(variables)  # Minimize
        
        try:
            result = minimize(objective_func, x0, bounds=bounds_list, method='L-BFGS-B')
            if result.success:
                # Update state with optimized values
                for i, var_name in enumerate(self.bounds.keys()):
                    state.variables[var_name] = result.x[i]
                return 3.0  # High reward for successful local optimization
        except Exception as e:
            logger.warning(f"Local search failed: {e}")
        
        return 0.5  # Small reward for attempt
    
    async def _apply_random_perturbation(self, state: MathematicalState, scale: float = 0.1) -> float:
        """Aplica perturbação aleatória."""
        
        for var_name in self.bounds.keys():
            bounds = self.bounds[var_name]
            current_value = state.variables.get(var_name, 0.5)
            range_size = bounds[1] - bounds[0]
            
            # Random perturbation
            perturbation = np.random.normal(0, scale * range_size)
            new_value = current_value + perturbation
            new_value = max(bounds[0], min(bounds[1], new_value))
            
            state.variables[var_name] = new_value
        
        return 0.5  # Low reward for random exploration
    
    async def _apply_domain_specific_optimization(self, state: MathematicalState, action: str) -> float:
        """Aplica otimização específica do domínio."""
        
        if self.domain == MathematicalDomain.BIOMATERIALS and "porosity_strength" in action:
            # Otimize trade-off between porosity and mechanical strength
            porosity = state.variables.get('porosity', 0.7)
            strength = state.variables.get('mechanical_strength', 5.0)
            
            # Empirical relationship: strength decreases with porosity
            optimal_strength = 50.0 * (1 - porosity) ** 2
            
            state.variables['mechanical_strength'] = min(
                self.bounds['mechanical_strength'][1],
                max(self.bounds['mechanical_strength'][0], optimal_strength)
            )
            
            return 2.5  # Good reward for domain expertise
        
        elif self.domain == MathematicalDomain.NEUROSCIENCE and "small_world" in action:
            # Optimize for small-world properties
            clustering = state.variables.get('clustering_coefficient', 0.6)
            density = state.variables.get('connection_density', 0.15)
            
            # Optimal path length based on network theory
            optimal_path_length = 2.0 + 3.0 * (1 - density)
            
            state.variables['average_path_length'] = min(
                self.bounds['average_path_length'][1],
                max(self.bounds['average_path_length'][0], optimal_path_length)
            )
            
            return 2.5
        
        return 1.0
    
    async def _local_optimize_variable(self, state: MathematicalState, var_name: str) -> float:
        """Otimização local de uma variável específica."""
        
        bounds = self.bounds[var_name]
        current_vars = state.variables.copy()
        
        def single_var_objective(x):
            vars_copy = current_vars.copy()
            vars_copy[var_name] = x[0]
            return -self.objective.evaluate(vars_copy)
        
        try:
            result = minimize(
                single_var_objective, 
                [current_vars.get(var_name, 0.5)], 
                bounds=[bounds],
                method='Brent'
            )
            if result.success:
                return result.x[0]
        except Exception:
            pass
        
        return current_vars.get(var_name, 0.5)
    
    async def _compute_finite_differences_gradient(
        self, 
        variables: Dict[str, float], 
        h: float = 1e-6
    ) -> Dict[str, float]:
        """Computa gradiente por diferenças finitas."""
        
        gradient = {}
        base_value = self.objective.evaluate(variables)
        
        for var_name in variables.keys():
            if var_name in self.bounds:
                vars_plus = variables.copy()
                vars_plus[var_name] += h
                
                value_plus = self.objective.evaluate(vars_plus)
                gradient[var_name] = (value_plus - base_value) / h
        
        return gradient
    
    async def evaluate_state(self, state: MathematicalState) -> float:
        """Avalia estado matemático."""
        
        if state.objective_value is not None:
            base_score = state.objective_value
        else:
            base_score = self.objective.evaluate(state.variables)
        
        # Bonus for feasibility
        if state.feasible:
            base_score += 1.0
        
        # Bonus for improvement trend
        if len(state.improvement_history) > 0:
            recent_improvements = state.improvement_history[-5:]
            avg_improvement = float(np.mean(recent_improvements))
            base_score += avg_improvement * 10.0  # Scale improvement bonus
        
        return float(base_score)
    
    async def simulate_rollout(self, state: MathematicalState, max_depth: int = 5) -> float:
        """Simula rollout matemático."""
        
        current_state = state
        total_value = 0.0
        
        for _ in range(max_depth):
            actions = await self.get_available_actions(current_state)
            if not actions:
                break
            
            # Select action with some intelligence (prefer optimization actions)
            optimization_actions = [a for a in actions if 'optimize' in a[0] or 'gradient' in a[0]]
            
            if optimization_actions and np.random.random() < 0.7:
                action, _ = max(optimization_actions, key=lambda x: x[1])
            else:
                action, _ = np.random.choice(actions, p=[p for _, p in actions] / np.sum([p for _, p in actions]))
            
            new_state, reward, is_terminal = await self.apply_action(current_state, action)
            
            total_value += reward
            current_state = new_state
            
            if is_terminal:
                break
        
        return total_value / max_depth


# ==================== MATHEMATICAL SEARCH FACADE ====================

class MathematicalSearch:
    """Facade principal para busca matemática especializada."""
    
    def __init__(self, engine_config: Optional[TreeSearchEngineConfig] = None):
        self.engine = TreeSearchEngine(engine_config or TreeSearchEngineConfig())
        self.objectives_registry: Dict[str, MathematicalObjective] = {}
        
        # Register default objectives
        self._register_default_objectives()
    
    def _register_default_objectives(self):
        """Registra funções objetivo padrão."""
        self.objectives_registry.update({
            'scaffold_optimization': ScaffoldOptimizationObjective(),
            'network_topology': NetworkTopologyObjective(),
            'logical_system': LogicalSystemObjective(),
        })
    
    def register_objective(self, name: str, objective: MathematicalObjective):
        """Registra nova função objetivo."""
        self.objectives_registry[name] = objective
    
    async def search(
        self,
        request: MathematicalSearchRequest,
        custom_objective: Optional[MathematicalObjective] = None
    ) -> SearchResult:
        """Executa busca matemática."""
        
        # Get objective function
        if custom_objective:
            objective = custom_objective
        elif request.problem_type in self.objectives_registry:
            objective = self.objectives_registry[request.problem_type]
        else:
            # Create generic objective
            objective = self._create_generic_objective(request)
        
        # Determine domain
        domain = MathematicalDomain(request.domain)
        
        # Create evaluator
        evaluator = MathematicalStateEvaluator(objective, domain)
        
        # Create initial state
        initial_state = MathematicalState(
            variables=self._initialize_variables(objective.get_bounds()),
            domain=domain
        )
        
        # Convert to search request
        search_request = SearchConfigRequest(
            max_budget_nodes=request.max_evaluations,
            max_depth=20,
            default_budget=min(request.max_evaluations, 1000),
            c_puct=1.414,
            expansion_threshold=3,
            simulation_rollouts=5,
            temperature=1.0,
            use_progressive_widening=True,
            alpha_widening=0.5,
            algorithm=SearchAlgorithm.PUCT,
            selection_policy=SelectionPolicy.BEST,
            expansion_policy=ExpansionPolicy.PROGRESSIVE,
            simulation_policy=SimulationPolicy.HEURISTIC,
            backpropagation_policy=BackpropagationPolicy.AVERAGE,
            convergence_threshold=max(request.convergence_tolerance, 1e-4),
            parallel_simulations=1,
            timeout_seconds=60.0,
            verbose=True,
            save_tree=False
        )
        
        # Execute search - convert initial_state to string representation
        return await self.engine.search(search_request, str(initial_state), evaluator)
    
    def _create_generic_objective(self, request: MathematicalSearchRequest) -> MathematicalObjective:
        """Cria função objetivo genérica."""
        
        class GenericObjective(MathematicalObjective):
            def __init__(self, req: MathematicalSearchRequest):
                self.request = req
            
            def evaluate(self, variables: Dict[str, float]) -> float:
                # Simple quadratic objective for demonstration
                return sum(v ** 2 for v in variables.values())
            
            def get_bounds(self) -> Dict[str, Tuple[float, float]]:
                return self.request.variable_bounds
            
            def is_feasible(self, variables: Dict[str, float]) -> Tuple[bool, List[str]]:
                # Basic bounds checking
                violations = []
                for var_name, value in variables.items():
                    if var_name in self.request.variable_bounds:
                        bounds = self.request.variable_bounds[var_name]
                        if not (bounds[0] <= value <= bounds[1]):
                            violations.append(f"{var_name} outside bounds {bounds}")
                
                return len(violations) == 0, violations
        
        return GenericObjective(request)
    
    def _initialize_variables(self, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Inicializa variáveis nos centros dos bounds."""
        return {
            var_name: (bound_tuple[0] + bound_tuple[1]) / 2
            for var_name, bound_tuple in bounds.items()
        }
    
    async def multi_objective_search(
        self,
        objectives: List[MathematicalObjective],
        request: MathematicalSearchRequest
    ) -> List[SearchResult]:
        """Executa busca multi-objetivo (Pareto)."""
        
        results = []
        
        for i, objective in enumerate(objectives):
            # Create modified request for each objective
            obj_request = MathematicalSearchRequest(
                problem_type=f"{request.problem_type}_obj_{i}",
                objective_function=request.objective_function,
                constraints=request.constraints,
                variable_bounds=request.variable_bounds,
                discrete_variables=request.discrete_variables,
                continuous_variables=request.continuous_variables,
                max_evaluations=request.max_evaluations // len(objectives),
                convergence_tolerance=request.convergence_tolerance,
                domain=request.domain,
                specialized_algorithms=request.specialized_algorithms
            )
            
            result = await self.search(obj_request, objective)
            results.append(result)
        
        return results
    
    def get_available_objectives(self) -> List[str]:
        """Retorna objetivos disponíveis."""
        return list(self.objectives_registry.keys())
    
    def cleanup(self):
        """Limpa recursos."""
        self.engine.cleanup()