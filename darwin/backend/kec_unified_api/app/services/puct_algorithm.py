"""PUCT Algorithm Core - Complete MCTS Implementation

Implementação completa do algoritmo PUCT (Polynomial Upper Confidence Bounds for Trees)
com variantes MCTS avançadas para exploração matemática otimizada.
"""

from __future__ import annotations

import math
import random
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from ..models.tree_search_models import (
    NodeStatus,
    SearchAlgorithm,
    SelectionPolicy,
    ExpansionPolicy,
    SimulationPolicy,
    BackpropagationPolicy,
    PUCTConfig,
    UCBNode,
)

logger = logging.getLogger(__name__)

# Type variables
S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type

# ==================== CORE NODE IMPLEMENTATION ====================

@dataclass
class PUCTNode(Generic[S]):
    """Nó PUCT com todas as estatísticas necessárias para MCTS."""
    
    # Core properties
    state: S
    node_id: str
    parent: Optional[PUCTNode[S]] = None
    action_taken: Optional[str] = None
    depth: int = 0
    
    # MCTS Statistics
    visits: int = 0
    total_value: float = 0.0
    squared_values: float = 0.0  # For variance calculation
    children: Dict[str, PUCTNode[S]] = field(default_factory=dict)
    
    # PUCT Specific
    prior_probability: float = 1.0
    virtual_loss: int = 0  # For parallel search
    
    # Node status
    status: NodeStatus = NodeStatus.UNVISITED
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    # Metadata
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    @property
    def mean_value(self) -> float:
        """Valor médio Q(s,a)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    @property
    def value_variance(self) -> float:
        """Variância dos valores observados."""
        if self.visits <= 1:
            return 0.0
        mean_sq = (self.squared_values / self.visits)
        mean = self.mean_value
        return max(0.0, mean_sq - mean * mean)
    
    @property
    def confidence_radius(self) -> float:
        """Raio de confiança para UCB-Tuned."""
        if self.visits <= 1:
            return float('inf')
        
        variance = self.value_variance
        visits = self.visits
        parent_visits = self.parent.visits if self.parent else visits
        
        # UCB-Tuned confidence bound
        log_term = math.log(parent_visits) / visits
        variance_term = variance + math.sqrt(2 * log_term)
        return math.sqrt(log_term * min(0.25, variance_term))
    
    def puct_value(self, config: PUCTConfig) -> float:
        """Calcula valor PUCT para seleção de nó."""
        if not self.parent:
            return 0.0
        
        # Exploitation term Q(s,a)
        Q = self.mean_value
        
        # Exploration term
        if self.visits == 0:
            return float('inf')  # Unvisited nodes get highest priority
        
        # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        N_parent = self.parent.visits
        N_self = self.visits + self.virtual_loss
        P = self.prior_probability
        
        if N_parent == 0:
            return Q
        
        exploration = config.c_puct * P * math.sqrt(N_parent) / (1 + N_self)
        
        return Q + exploration
    
    def ucb1_value(self, c: float = math.sqrt(2)) -> float:
        """Calcula valor UCB1."""
        if not self.parent or self.visits == 0:
            return float('inf')
        
        exploitation = self.mean_value
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def ucb_tuned_value(self) -> float:
        """Calcula valor UCB-Tuned com variância."""
        if not self.parent or self.visits == 0:
            return float('inf')
        
        exploitation = self.mean_value
        exploration = self.confidence_radius
        return exploitation + exploration
    
    def add_child(self, action: str, child_state: S, prior: float = 1.0) -> PUCTNode[S]:
        """Adiciona nó filho."""
        child_id = f"{self.node_id}_{action}"
        child = PUCTNode(
            state=child_state,
            node_id=child_id,
            parent=self,
            action_taken=action,
            depth=self.depth + 1,
            prior_probability=prior,
            children={}
        )
        self.children[action] = child
        return child
    
    def update(self, value: float):
        """Atualiza estatísticas do nó."""
        self.visits += 1
        self.total_value += value
        self.squared_values += value * value
        self.last_updated = datetime.now(timezone.utc)
    
    def is_leaf(self) -> bool:
        """Verifica se é nó folha."""
        return len(self.children) == 0 or self.is_terminal
    
    def is_fully_expanded(self, available_actions: List[str]) -> bool:
        """Verifica se nó está totalmente expandido."""
        return len(self.children) >= len(available_actions)


# ==================== ABSTRACT BASE CLASSES ====================

class StateEvaluator(ABC, Generic[S]):
    """Interface para avaliação de estados."""
    
    @abstractmethod
    async def get_available_actions(self, state: S) -> List[Tuple[str, float]]:
        """Retorna ações disponíveis com probabilidades prior."""
        pass
    
    @abstractmethod
    async def apply_action(self, state: S, action: str) -> Tuple[S, float, bool]:
        """Aplica ação e retorna (new_state, immediate_reward, is_terminal)."""
        pass
    
    @abstractmethod
    async def evaluate_state(self, state: S) -> float:
        """Avalia estado e retorna valor estimado."""
        pass
    
    @abstractmethod
    async def simulate_rollout(self, state: S, max_depth: int = 10) -> float:
        """Simula rollout a partir do estado."""
        pass


class SearchPolicy(ABC):
    """Interface para políticas de busca."""
    
    @abstractmethod
    def select_child(self, node: PUCTNode, config: PUCTConfig) -> Optional[PUCTNode]:
        """Seleciona nó filho baseado na política."""
        pass


# ==================== SELECTION POLICIES ====================

class BestChildPolicy(SearchPolicy):
    """Política de seleção do melhor filho (exploitation)."""
    
    def select_child(self, node: PUCTNode, config: PUCTConfig) -> Optional[PUCTNode]:
        if not node.children:
            return None
        return max(node.children.values(), key=lambda c: c.mean_value)


class PUCTSelectionPolicy(SearchPolicy):
    """Política de seleção PUCT (balanceamento exploration/exploitation)."""
    
    def select_child(self, node: PUCTNode, config: PUCTConfig) -> Optional[PUCTNode]:
        if not node.children:
            return None
        return max(node.children.values(), key=lambda c: c.puct_value(config))


class UCB1SelectionPolicy(SearchPolicy):
    """Política de seleção UCB1."""
    
    def __init__(self, c: float = math.sqrt(2)):
        self.c = c
    
    def select_child(self, node: PUCTNode, config: PUCTConfig) -> Optional[PUCTNode]:
        if not node.children:
            return None
        return max(node.children.values(), key=lambda c: c.ucb1_value(self.c))


class TemperatureSelectionPolicy(SearchPolicy):
    """Política de seleção baseada em temperatura (exploration stochástico)."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def select_child(self, node: PUCTNode, config: PUCTConfig) -> Optional[PUCTNode]:
        if not node.children:
            return None
        
        if self.temperature == 0:
            # Deterministic - select best
            return max(node.children.values(), key=lambda c: c.mean_value)
        
        # Boltzmann distribution
        children_list = list(node.children.values())
        values = [c.mean_value for c in children_list]
        if not values:
            return None
        
        # Normalize and apply temperature
        max_val = max(values)
        exp_values = [math.exp((v - max_val) / self.temperature) for v in values]
        total = sum(exp_values)
        
        if total == 0:
            return random.choice(children_list)
        
        probs = [exp_val / total for exp_val in exp_values]
        indices = list(range(len(children_list)))
        selected_index = np.random.choice(indices, p=probs)
        return children_list[selected_index]


# ==================== CORE PUCT ALGORITHM ====================

class PUCTAlgorithm(Generic[S]):
    """Implementação completa do algoritmo PUCT/MCTS."""
    
    def __init__(
        self,
        evaluator: StateEvaluator[S],
        config: PUCTConfig,
        algorithm: SearchAlgorithm = SearchAlgorithm.PUCT
    ):
        self.evaluator = evaluator
        self.config = config
        self.algorithm = algorithm
        
        # Initialize selection policy
        self.selection_policy = self._create_selection_policy()
        
        # Search state
        self.root: Optional[PUCTNode[S]] = None
        self.nodes_explored = 0
        self.iterations_completed = 0
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.search_statistics = {}
        
        # Parallel execution
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _create_selection_policy(self) -> SearchPolicy:
        """Cria política de seleção baseada no algoritmo."""
        if self.algorithm == SearchAlgorithm.PUCT:
            return PUCTSelectionPolicy()
        elif self.algorithm == SearchAlgorithm.UCB1:
            return UCB1SelectionPolicy()
        elif self.algorithm == SearchAlgorithm.UCT:
            return UCB1SelectionPolicy(c=math.sqrt(2))
        else:
            return PUCTSelectionPolicy()  # Default to PUCT
    
    async def search(
        self,
        initial_state: S,
        max_iterations: int,
        timeout_seconds: Optional[float] = None
    ) -> PUCTNode[S]:
        """Executa busca MCTS completa."""
        
        logger.info(f"Starting {self.algorithm} search with {max_iterations} iterations")
        
        # Initialize search
        self.start_time = datetime.now(timezone.utc)
        self.root = PUCTNode(
            state=initial_state,
            node_id="root",
            depth=0,
            children={}
        )
        
        # Check if root is terminal
        actions = await self.evaluator.get_available_actions(initial_state)
        if not actions:
            self.root.is_terminal = True
            self.root.terminal_value = await self.evaluator.evaluate_state(initial_state)
            return self.root
        
        # Main MCTS loop
        iteration = 0
        timeout_time = None
        if timeout_seconds:
            timeout_time = self.start_time.timestamp() + timeout_seconds
        
        try:
            while iteration < max_iterations:
                # Check timeout
                if timeout_time and datetime.now(timezone.utc).timestamp() > timeout_time:
                    logger.info(f"Search timeout reached after {iteration} iterations")
                    break
                
                # Single MCTS iteration
                await self._mcts_iteration()
                iteration += 1
                
                # Log progress
                if iteration % 100 == 0:
                    elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                    logger.info(f"Iteration {iteration}/{max_iterations}, "
                              f"nodes: {self.nodes_explored}, "
                              f"time: {elapsed:.2f}s")
        
        except Exception as e:
            logger.error(f"Error during MCTS search: {e}")
            raise
        
        finally:
            self.iterations_completed = iteration
            self._compute_final_statistics()
        
        return self.root
    
    async def _mcts_iteration(self):
        """Single MCTS iteration: Select, Expand, Simulate, Backpropagate."""
        
        # 1. Selection - traverse tree using selection policy
        if self.root is None:
            raise RuntimeError("Root node is None - search not initialized")
        leaf_node = await self._select(self.root)
        
        # 2. Expansion - add new child if not terminal
        child_node = await self._expand(leaf_node)
        
        # 3. Simulation - rollout from leaf/child
        simulation_node = child_node if child_node else leaf_node
        value = await self._simulate(simulation_node)
        
        # 4. Backpropagation - update statistics
        await self._backpropagate(simulation_node, value)
        
        self.nodes_explored += 1
    
    async def _select(self, node: PUCTNode[S]) -> PUCTNode[S]:
        """Selection phase - traverse tree to leaf using selection policy."""
        
        current = node
        
        while not current.is_leaf() and not current.is_terminal:
            # Select child using policy
            child = self.selection_policy.select_child(current, self.config)
            
            if not child:
                break
            
            # Add virtual loss for parallel search
            if self.config.use_virtual_loss:
                child.virtual_loss += 1
            
            current = child
        
        return current
    
    async def _expand(self, node: PUCTNode[S]) -> Optional[PUCTNode[S]]:
        """Expansion phase - add new child to leaf node."""
        
        # Don't expand terminal nodes
        if node.is_terminal:
            return None
        
        # Check expansion threshold
        if node.visits < self.config.expansion_threshold:
            return None
        
        # Get available actions
        try:
            actions = await self.evaluator.get_available_actions(node.state)
        except Exception as e:
            logger.error(f"Error getting actions for state: {e}")
            return None
        
        if not actions:
            node.is_terminal = True
            return None
        
        # Find unexpanded actions
        expanded_actions = set(node.children.keys())
        available_actions = [(action, prior) for action, prior in actions 
                           if action not in expanded_actions]
        
        if not available_actions:
            # Node is fully expanded
            return None
        
        # Progressive widening
        if self.config.use_progressive_widening:
            max_children = max(1, int(node.visits ** self.config.alpha_widening))
            if len(node.children) >= max_children:
                return None
        
        # Select action to expand (highest prior or random)
        if available_actions:
            action, prior = max(available_actions, key=lambda x: x[1])
        else:
            return None
        
        # Apply action to get new state
        try:
            new_state, reward, is_terminal = await self.evaluator.apply_action(
                node.state, action
            )
        except Exception as e:
            logger.error(f"Error applying action {action}: {e}")
            return None
        
        # Create child node
        child = node.add_child(action, new_state, prior)
        child.is_terminal = is_terminal
        
        if is_terminal:
            child.terminal_value = reward
        
        # Mark node as expanded
        if node.status == NodeStatus.UNVISITED:
            node.status = NodeStatus.EXPANDED
        
        return child
    
    async def _simulate(self, node: PUCTNode[S]) -> float:
        """Simulation phase - estimate value via rollout."""
        
        # Use terminal value if available
        if node.is_terminal and node.terminal_value is not None:
            return node.terminal_value
        
        # Use cached evaluation for visited nodes above threshold
        if node.visits > 10:
            try:
                return await self.evaluator.evaluate_state(node.state)
            except Exception as e:
                logger.warning(f"Error evaluating state, falling back to rollout: {e}")
        
        # Perform rollout simulation
        try:
            return await self.evaluator.simulate_rollout(
                node.state,
                max_depth=max(1, 20 - node.depth)
            )
        except Exception as e:
            logger.warning(f"Error in rollout simulation: {e}")
            # Fallback to simple evaluation
            try:
                return await self.evaluator.evaluate_state(node.state)
            except Exception as e2:
                logger.error(f"Error in fallback evaluation: {e2}")
                return 0.0  # Neutral value as last resort
    
    async def _backpropagate(self, node: PUCTNode[S], value: float):
        """Backpropagation phase - update statistics up the tree."""
        
        current = node
        
        while current is not None:
            # Update node statistics
            current.update(value)
            
            # Remove virtual loss
            if self.config.use_virtual_loss and current.virtual_loss > 0:
                current.virtual_loss -= 1
            
            # Move to parent
            current = current.parent
    
    def get_best_action(self) -> Optional[str]:
        """Retorna melhor ação do nó raiz."""
        if not self.root or not self.root.children:
            return None
        
        # Use visitation count (robust) or value (greedy)
        best_child = max(self.root.children.values(), key=lambda c: c.visits)
        return best_child.action_taken
    
    def get_best_action_sequence(self, max_length: int = 10) -> List[str]:
        """Retorna sequência de melhores ações."""
        if not self.root:
            return []
        
        sequence = []
        current = self.root
        
        for _ in range(max_length):
            if not current.children:
                break
            
            # Select child with most visits (most robust)
            best_child = max(current.children.values(), key=lambda c: c.visits)
            
            if best_child.action_taken:
                sequence.append(best_child.action_taken)
            
            current = best_child
        
        return sequence
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[str, float]:
        """Retorna probabilidades de ações baseadas em visitas."""
        if not self.root or not self.root.children:
            return {}
        
        if temperature == 0:
            # Deterministic - best action gets probability 1
            best_action = self.get_best_action()
            return {action: 1.0 if action == best_action else 0.0 
                   for action in self.root.children.keys()}
        
        # Temperature-based probabilities
        visits = np.array([child.visits for child in self.root.children.values()])
        actions = list(self.root.children.keys())
        
        if temperature == 1.0:
            # Direct proportion to visits
            total_visits = sum(visits)
            probs = visits / total_visits if total_visits > 0 else np.ones(len(visits)) / len(visits)
        else:
            # Apply temperature
            log_visits = np.log(visits + 1e-10)  # Avoid log(0)
            exp_values = np.exp(log_visits / temperature)
            probs = exp_values / np.sum(exp_values)
        
        return dict(zip(actions, probs))
    
    def _compute_final_statistics(self):
        """Computa estatísticas finais da busca."""
        if not self.start_time:
            return
        
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - self.start_time).total_seconds()
        
        # Tree statistics
        def count_nodes(node: PUCTNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children.values())
        
        def max_depth(node: PUCTNode) -> int:
            if not node.children:
                return node.depth
            return max(max_depth(child) for child in node.children.values())
        
        total_nodes = count_nodes(self.root) if self.root else 0
        tree_depth = max_depth(self.root) if self.root else 0
        
        self.search_statistics = {
            'total_time_seconds': total_time,
            'iterations_completed': self.iterations_completed,
            'nodes_explored': self.nodes_explored,
            'total_nodes_in_tree': total_nodes,
            'max_tree_depth': tree_depth,
            'nodes_per_second': self.nodes_explored / total_time if total_time > 0 else 0,
            'iterations_per_second': self.iterations_completed / total_time if total_time > 0 else 0,
            'root_visits': self.root.visits if self.root else 0,
            'root_value': self.root.mean_value if self.root else 0,
            'root_children': len(self.root.children) if self.root else 0,
            'algorithm_used': self.algorithm.value,
            'c_puct': self.config.c_puct,
        }
        
        logger.info(f"Search completed: {self.search_statistics}")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da busca."""
        return self.search_statistics.copy()
    
    def cleanup(self):
        """Limpa recursos."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# ==================== SPECIALIZED EVALUATORS ====================

class TestStateEvaluator(StateEvaluator[str]):
    """Avaliador de teste para estados string."""
    
    async def get_available_actions(self, state: str) -> List[Tuple[str, float]]:
        """Retorna ações básicas para teste."""
        base_actions = ["expand", "refine", "explore", "optimize"]
        # Simple prior based on action name
        actions_with_priors = [(action, 1.0 - i * 0.1) for i, action in enumerate(base_actions)]
        return actions_with_priors
    
    async def apply_action(self, state: str, action: str) -> Tuple[str, float, bool]:
        """Aplica ação e retorna novo estado."""
        new_state = f"{state}->{action}"
        reward = hash(new_state) % 100 / 100.0  # Deterministic but varied
        is_terminal = len(new_state) > 50  # Terminal after certain length
        return new_state, reward, is_terminal
    
    async def evaluate_state(self, state: str) -> float:
        """Avalia estado usando hash."""
        return (hash(state) % 1000) / 1000.0
    
    async def simulate_rollout(self, state: str, max_depth: int = 10) -> float:
        """Simula rollout randômico."""
        current_state = state
        total_reward = 0.0
        
        for _ in range(max_depth):
            actions = await self.get_available_actions(current_state)
            if not actions:
                break
            
            # Random action selection
            action, _ = random.choice(actions)
            new_state, reward, is_terminal = await self.apply_action(current_state, action)
            
            total_reward += reward
            current_state = new_state
            
            if is_terminal:
                break
        
        return total_reward / max_depth