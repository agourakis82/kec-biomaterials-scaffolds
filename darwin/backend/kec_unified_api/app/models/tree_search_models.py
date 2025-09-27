"""Tree Search PUCT Models - Complete MCTS Implementation

Modelos Pydantic completos para Tree Search PUCT com algoritmos MCTS avançados.
Migrados e expandidos do backend principal com funcionalidades matemáticas especializadas.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from datetime import datetime

from pydantic import BaseModel, Field, validator

# Type variables for generic nodes
S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type

# ==================== CORE ENUMS ====================

class NodeStatus(str, Enum):
    """Status do nó na árvore de busca."""
    UNVISITED = "unvisited"
    EXPANDED = "expanded" 
    TERMINAL = "terminal"
    PRUNED = "pruned"


class SearchAlgorithm(str, Enum):
    """Algoritmos de busca disponíveis."""
    PUCT = "puct"
    UCB1 = "ucb1"
    UCT = "uct"
    THOMPSON_SAMPLING = "thompson_sampling"
    PROGRESSIVE_WIDENING = "progressive_widening"


class SelectionPolicy(str, Enum):
    """Políticas de seleção de nós."""
    BEST = "best"
    TEMPERATURE = "temperature"
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"


class ExpansionPolicy(str, Enum):
    """Políticas de expansão de nós."""
    ALL_ACTIONS = "all_actions"
    PROGRESSIVE = "progressive"
    THRESHOLD_BASED = "threshold_based"
    BAYESIAN = "bayesian"


class SimulationPolicy(str, Enum):
    """Políticas de simulação."""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    NEURAL_NETWORK = "neural_network"
    DOMAIN_SPECIFIC = "domain_specific"


class BackpropagationPolicy(str, Enum):
    """Políticas de backpropagation."""
    AVERAGE = "average"
    MAX = "max"
    MINIMAX = "minimax"
    UCB_TUNED = "ucb_tuned"


# ==================== CORE MODELS ====================

class SearchConfigRequest(BaseModel):
    """Configuração de busca Tree Search PUCT."""
    
    # PUCT Algorithm Parameters
    max_budget_nodes: Optional[int] = Field(1000, ge=10, le=10000, description="Maximum number of nodes to explore")
    max_depth: Optional[int] = Field(10, ge=1, le=50, description="Maximum search depth")
    default_budget: Optional[int] = Field(500, ge=10, le=5000, description="Default search budget")
    c_puct: Optional[float] = Field(1.414, ge=0.1, le=10.0, description="PUCT exploration constant")
    
    # Expansion and Selection
    expansion_threshold: Optional[int] = Field(5, ge=1, le=100, description="Minimum visits before expansion")
    simulation_rollouts: Optional[int] = Field(5, ge=1, le=50, description="Number of simulation rollouts")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=5.0, description="Temperature for action selection")
    
    # Progressive Widening
    use_progressive_widening: Optional[bool] = Field(True, description="Enable progressive widening")
    alpha_widening: Optional[float] = Field(0.5, ge=0.1, le=1.0, description="Progressive widening parameter")
    
    # Algorithm Selection
    algorithm: Optional[SearchAlgorithm] = Field(SearchAlgorithm.PUCT, description="Search algorithm to use")
    selection_policy: Optional[SelectionPolicy] = Field(SelectionPolicy.BEST, description="Node selection policy")
    expansion_policy: Optional[ExpansionPolicy] = Field(ExpansionPolicy.PROGRESSIVE, description="Node expansion policy")
    simulation_policy: Optional[SimulationPolicy] = Field(SimulationPolicy.HEURISTIC, description="Simulation policy")
    backpropagation_policy: Optional[BackpropagationPolicy] = Field(BackpropagationPolicy.AVERAGE, description="Backpropagation policy")
    
    # Performance and Convergence
    convergence_threshold: Optional[float] = Field(0.001, ge=1e-6, le=0.1, description="Convergence threshold")
    parallel_simulations: Optional[int] = Field(1, ge=1, le=16, description="Number of parallel simulations")
    timeout_seconds: Optional[float] = Field(30.0, ge=1.0, le=300.0, description="Search timeout in seconds")
    
    # Logging and Debugging
    verbose: Optional[bool] = Field(False, description="Enable verbose logging")
    save_tree: Optional[bool] = Field(False, description="Save complete search tree")
    
    @validator('c_puct')
    def validate_c_puct(cls, v):
        if v <= 0:
            raise ValueError("c_puct must be positive")
        return v


class TreeNode(BaseModel):
    """Nó da árvore de busca Tree Search."""
    
    node_id: str = Field(..., description="Unique node identifier")
    state: Union[str, Dict[str, Any]] = Field(..., description="Node state representation")
    action_taken: Optional[str] = Field(None, description="Action that led to this node")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    depth: int = Field(0, ge=0, description="Node depth in tree")
    
    # MCTS Statistics
    visits: int = Field(0, ge=0, description="Number of visits")
    total_value: float = Field(0.0, description="Total accumulated value")
    mean_value: Optional[float] = Field(None, description="Mean value (total_value / visits)")
    variance: Optional[float] = Field(None, description="Value variance")
    
    # PUCT Specific
    puct_score: Optional[float] = Field(None, description="PUCT selection score")
    prior_probability: float = Field(1.0, ge=0.0, le=1.0, description="Prior probability")
    
    # Node Properties
    status: NodeStatus = Field(NodeStatus.UNVISITED, description="Node status")
    is_terminal: bool = Field(False, description="Is terminal node")
    children_ids: List[str] = Field(default_factory=list, description="Child node IDs")
    
    # Additional Metadata
    created_at: Optional[datetime] = Field(None, description="Node creation timestamp")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")
    
    @validator('mean_value', always=True)
    def calculate_mean_value(cls, v, values):
        if values.get('visits', 0) > 0:
            return values.get('total_value', 0.0) / values['visits']
        return 0.0


class ActionResult(BaseModel):
    """Resultado de uma ação na busca."""
    
    action: str = Field(..., description="Action taken")
    new_state: Union[str, Dict[str, Any]] = Field(..., description="Resulting state")
    reward: float = Field(..., description="Immediate reward")
    probability: float = Field(1.0, ge=0.0, le=1.0, description="Action probability")
    is_terminal: bool = Field(False, description="Is resulting state terminal")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Action metadata")


class SearchState(BaseModel):
    """Estado atual da busca."""
    
    search_id: str = Field(..., description="Unique search identifier")
    current_iteration: int = Field(0, ge=0, description="Current search iteration")
    total_iterations: int = Field(0, ge=0, description="Total planned iterations")
    nodes_explored: int = Field(0, ge=0, description="Number of nodes explored")
    current_depth: int = Field(0, ge=0, description="Current maximum depth")
    
    # Search Progress
    start_time: Optional[datetime] = Field(None, description="Search start time")
    current_time: Optional[datetime] = Field(None, description="Current time")
    elapsed_seconds: Optional[float] = Field(None, description="Elapsed search time")
    
    # Best Solution
    best_value: Optional[float] = Field(None, description="Best value found")
    best_action_sequence: List[str] = Field(default_factory=list, description="Best action sequence")
    
    # Convergence
    has_converged: bool = Field(False, description="Has search converged")
    convergence_value: Optional[float] = Field(None, description="Convergence measure")
    
    # Statistics
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Search statistics")


class SearchResult(BaseModel):
    """Resultado completo da busca Tree Search."""
    
    # Search Metadata
    search_id: str = Field(..., description="Unique search identifier")
    algorithm: SearchAlgorithm = Field(..., description="Algorithm used")
    config: SearchConfigRequest = Field(..., description="Search configuration used")
    
    # Results
    root_node: TreeNode = Field(..., description="Root node of search tree")
    best_action_sequence: List[str] = Field(..., description="Best action sequence found")
    best_value: float = Field(..., description="Best value achieved")
    
    # Search Tree (optional)
    search_tree: Optional[List[TreeNode]] = Field(None, description="Complete search tree (if saved)")
    
    # Performance Metrics
    nodes_explored: int = Field(..., ge=0, description="Total nodes explored")
    max_depth_reached: int = Field(..., ge=0, description="Maximum depth reached")
    total_iterations: int = Field(..., ge=0, description="Total iterations performed")
    
    # Timing
    start_time: datetime = Field(..., description="Search start time")
    end_time: datetime = Field(..., description="Search end time")
    total_time_seconds: float = Field(..., ge=0, description="Total search time in seconds")
    
    # Convergence
    converged: bool = Field(..., description="Did search converge")
    convergence_iteration: Optional[int] = Field(None, description="Iteration where convergence occurred")
    
    # Statistics
    search_statistics: Dict[str, Any] = Field(..., description="Detailed search statistics")
    
    # Success/Error
    success: bool = Field(True, description="Was search successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class QuickSearchResult(BaseModel):
    """Resultado de busca rápida simplificada."""
    
    query: str = Field(..., description="Original search query")
    best_path: List[str] = Field(..., description="Best path found")
    exploration_score: float = Field(..., ge=0.0, le=1.0, description="Quality of exploration")
    nodes_explored: int = Field(..., ge=0, description="Nodes explored")
    search_time_seconds: float = Field(..., ge=0, description="Search time")
    success: bool = Field(True, description="Search success")


# ==================== PUCT SPECIFIC MODELS ====================

class PUCTConfig(BaseModel):
    """Configuração específica do algoritmo PUCT."""
    
    c_puct: float = Field(1.414, ge=0.1, le=10.0, description="PUCT exploration constant")
    use_virtual_loss: bool = Field(False, description="Use virtual loss for parallel search")
    virtual_loss_value: float = Field(1.0, ge=0.0, description="Virtual loss value")
    
    # Node Expansion
    expansion_threshold: int = Field(5, ge=1, description="Minimum visits before expansion")
    max_children: Optional[int] = Field(None, description="Maximum children per node")
    
    # Progressive Widening
    use_progressive_widening: bool = Field(True, description="Enable progressive widening")
    alpha_widening: float = Field(0.5, ge=0.1, le=1.0, description="Progressive widening parameter")
    
    # Prior Knowledge
    use_prior: bool = Field(True, description="Use prior probabilities")
    prior_weight: float = Field(1.0, ge=0.0, description="Weight for prior probabilities")
    
    # Value Estimation
    value_function: str = Field("mean", description="Value function to use")
    backup_operator: str = Field("average", description="Backup operator")


class UCBNode(BaseModel):
    """Nó específico para algoritmos UCB."""
    
    node_id: str = Field(..., description="Node identifier")
    visits: int = Field(0, ge=0, description="Visit count")
    total_reward: float = Field(0.0, description="Total reward")
    ucb_value: Optional[float] = Field(None, description="UCB value")
    confidence_radius: Optional[float] = Field(None, description="Confidence radius")
    
    # UCB Specific
    last_reward: Optional[float] = Field(None, description="Last observed reward")
    reward_variance: Optional[float] = Field(None, description="Reward variance")
    
    def calculate_ucb1(self, parent_visits: int, c: float = math.sqrt(2)) -> float:
        """Calculate UCB1 value."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def calculate_puct(self, parent_visits: int, prior: float = 1.0, c_puct: float = 1.414) -> float:
        """Calculate PUCT value."""
        if self.visits == 0:
            return float('inf')
            
        Q = self.total_reward / self.visits
        U = c_puct * prior * math.sqrt(parent_visits) / (1 + self.visits)
        return Q + U


# ==================== MATHEMATICAL SEARCH MODELS ====================

class MathematicalSearchRequest(BaseModel):
    """Request para busca matemática especializada."""
    
    # Problem Definition
    problem_type: str = Field(..., description="Type of mathematical problem")
    objective_function: str = Field(..., description="Objective function to optimize")
    constraints: List[str] = Field(default_factory=list, description="Problem constraints")
    
    # Search Space
    variable_bounds: Dict[str, tuple] = Field(..., description="Variable bounds (min, max)")
    discrete_variables: List[str] = Field(default_factory=list, description="Discrete variable names")
    continuous_variables: List[str] = Field(default_factory=list, description="Continuous variable names")
    
    # Search Configuration
    max_evaluations: int = Field(1000, ge=10, le=100000, description="Maximum function evaluations")
    convergence_tolerance: float = Field(1e-6, ge=1e-12, le=1e-2, description="Convergence tolerance")
    
    # Mathematical Domain
    domain: str = Field("general", description="Mathematical domain (biomaterials, neuroscience, etc.)")
    specialized_algorithms: List[str] = Field(default_factory=list, description="Specialized algorithms to use")


class OptimizationProblem(BaseModel):
    """Definição de problema de otimização."""
    
    name: str = Field(..., description="Problem name")
    description: str = Field(..., description="Problem description")
    
    # Objective
    minimize: bool = Field(True, description="Minimize (True) or maximize (False)")
    multi_objective: bool = Field(False, description="Is multi-objective problem")
    
    # Variables
    num_variables: int = Field(..., ge=1, description="Number of variables")
    variable_names: List[str] = Field(..., description="Variable names")
    variable_types: Dict[str, str] = Field(..., description="Variable types (continuous, integer, binary)")
    
    # Bounds and Constraints
    bounds: Dict[str, tuple] = Field(..., description="Variable bounds")
    linear_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Linear constraints")
    nonlinear_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Nonlinear constraints")
    
    # Evaluation
    evaluation_function: Optional[str] = Field(None, description="Evaluation function code")
    known_optimum: Optional[float] = Field(None, description="Known optimal value (for testing)")


class ConstraintSystem(BaseModel):
    """Sistema de restrições para otimização."""
    
    linear_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Linear constraints")
    nonlinear_constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Nonlinear constraints")
    bounds: Dict[str, tuple] = Field(..., description="Variable bounds")
    
    # Constraint Handling
    penalty_method: str = Field("quadratic", description="Penalty method for constraint violations")
    penalty_weight: float = Field(1000.0, ge=0.0, description="Penalty weight")
    
    # Feasibility
    feasibility_tolerance: float = Field(1e-6, ge=1e-12, description="Feasibility tolerance")
    repair_infeasible: bool = Field(False, description="Attempt to repair infeasible solutions")


class ObjectiveFunction(BaseModel):
    """Função objetivo para otimização."""
    
    name: str = Field(..., description="Function name")
    expression: str = Field(..., description="Mathematical expression")
    variables: List[str] = Field(..., description="Variable names")
    
    # Function Properties
    is_convex: Optional[bool] = Field(None, description="Is function convex")
    is_differentiable: Optional[bool] = Field(None, description="Is function differentiable")
    has_gradient: bool = Field(False, description="Gradient available")
    has_hessian: bool = Field(False, description="Hessian available")
    
    # Evaluation
    evaluation_count: int = Field(0, ge=0, description="Number of evaluations")
    best_value: Optional[float] = Field(None, description="Best value found")
    best_solution: Optional[List[float]] = Field(None, description="Best solution found")
    
    # Metadata
    domain_specific: bool = Field(False, description="Is domain-specific function")
    computational_complexity: Optional[str] = Field(None, description="Computational complexity")