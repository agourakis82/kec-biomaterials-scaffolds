"""
Monte Carlo Tree Search (MCTS) Engine
=====================================

Implementação clássica de MCTS com variações e otimizações.
Complementa o PUCT com abordagens mais tradicionais.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, Dict, Any, Callable
import logging
import asyncio

from .puct import TreeNode, NodeStatus

logger = logging.getLogger(__name__)

S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type


@dataclass
class MCTSConfig:
    """Configuração para MCTS tradicional."""
    max_iterations: int = 1000
    max_depth: int = 10
    simulation_depth: int = 50
    c_exploration: float = math.sqrt(2)  # UCB1 constant
    use_ucb1: bool = True
    use_rapid_action_value_estimation: bool = False
    early_termination_threshold: float = 0.95
    min_simulations_per_node: int = 10
    temperature: float = 1.0


class MCTSEngine(Generic[S, A]):
    """
    Motor MCTS clássico com funcionalidades avançadas:
    - UCB1 para seleção
    - Random rollouts
    - Progressive bias
    - Early termination
    - RAVE (Rapid Action Value Estimation)
    """
    
    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self.root: Optional[TreeNode[S]] = None
        self.action_values: Dict[A, float] = {}  # Para RAVE
        self.action_counts: Dict[A, int] = {}
        
    async def search(
        self,
        initial_state: S,
        get_legal_actions: Callable[[S], List[A]],
        apply_action: Callable[[S, A], S],
        is_terminal: Callable[[S], bool],
        evaluate_state: Callable[[S], float],
        iterations: Optional[int] = None
    ) -> Tuple[A, TreeNode[S]]:
        """
        Executa busca MCTS completa.
        
        Args:
            initial_state: Estado inicial
            get_legal_actions: Função que retorna ações legais
            apply_action: Função que aplica ação ao estado
            is_terminal: Função que verifica se estado é terminal
            evaluate_state: Função que avalia estado
            iterations: Número de iterações (override config)
            
        Returns:
            Melhor ação e nó raiz
        """
        
        max_iterations = iterations or self.config.max_iterations
        
        self.root = TreeNode(
            state=initial_state,
            depth=self.config.max_depth
        )
        
        logger.info(f"Iniciando MCTS: {max_iterations} iterações")
        
        for iteration in range(max_iterations):
            # Selection + Expansion
            leaf_node = self._select_and_expand(
                self.root, 
                get_legal_actions,
                apply_action,
                is_terminal
            )
            
            # Simulation
            value = await self._simulate(
                leaf_node,
                get_legal_actions,
                apply_action, 
                is_terminal,
                evaluate_state
            )
            
            # Backpropagation
            self._backpropagate(leaf_node, value)
            
            # Early termination check
            if self._should_terminate_early():
                logger.info(f"Early termination at iteration {iteration}")
                break
                
            if (iteration + 1) % 100 == 0:
                logger.debug(f"MCTS progress: {iteration + 1}/{max_iterations}")
        
        # Select best action
        if not self.root.children:
            raise ValueError("No legal actions from root state")
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        best_action = best_child.action_taken
        
        logger.info(f"MCTS completed: best action = {best_action}")
        
        return best_action, self.root
    
    def _select_and_expand(
        self,
        node: TreeNode[S],
        get_legal_actions: Callable[[S], List[A]],
        apply_action: Callable[[S, A], S],
        is_terminal: Callable[[S], bool]
    ) -> TreeNode[S]:
        """Selection phase com expansão automática."""
        
        # Se nó não foi expandido e não é terminal, expande
        if (node.status == NodeStatus.UNVISITED and 
            not is_terminal(node.state) and 
            node.depth > 0):
            self._expand_node(node, get_legal_actions, apply_action)
            
            # Retorna filho aleatório se disponível
            if node.children:
                return random.choice(node.children)
            return node
        
        # Se não tem filhos ou é terminal, retorna
        if not node.children or node.depth <= 0:
            return node
        
        # Selection usando UCB1 ou outro critério
        if self.config.use_ucb1:
            best_child = max(node.children, key=lambda c: self._ucb1_value(c))
        else:
            # Seleção baseada em valor médio com exploration
            best_child = max(node.children, key=lambda c: c.mean_value + 
                           random.random() * self.config.c_exploration / max(1, c.visits))
        
        return self._select_and_expand(best_child, get_legal_actions, apply_action, is_terminal)
    
    def _expand_node(
        self,
        node: TreeNode[S],
        get_legal_actions: Callable[[S], List[A]],
        apply_action: Callable[[S, A], S]
    ) -> None:
        """Expande nó com todas as ações legais."""
        
        legal_actions = get_legal_actions(node.state)
        
        for action in legal_actions:
            new_state = apply_action(node.state, action)
            
            child = TreeNode(
                state=new_state,
                parent=node,
                action_taken=action,
                depth=node.depth - 1,
                status=NodeStatus.UNVISITED
            )
            
            node.children.append(child)
        
        node.status = NodeStatus.EXPANDED
        logger.debug(f"Expanded node with {len(legal_actions)} children")
    
    def _ucb1_value(self, node: TreeNode[S]) -> float:
        """Calcula valor UCB1 para seleção."""
        
        if node.visits == 0:
            return float('inf')
        
        if not node.parent:
            return node.mean_value
        
        exploration_term = self.config.c_exploration * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )
        
        # RAVE integration se habilitado
        if self.config.use_rapid_action_value_estimation and node.action_taken:
            action = node.action_taken
            if action in self.action_values and action in self.action_counts:
                rave_value = self.action_values[action] / max(1, self.action_counts[action])
                # Beta parameter for RAVE mixing
                beta = self.action_counts[action] / (self.action_counts[action] + node.visits + 4.0)
                mixed_value = (1 - beta) * node.mean_value + beta * rave_value
                return mixed_value + exploration_term
        
        return node.mean_value + exploration_term
    
    async def _simulate(
        self,
        node: TreeNode[S],
        get_legal_actions: Callable[[S], List[A]],
        apply_action: Callable[[S, A], S],
        is_terminal: Callable[[S], bool],
        evaluate_state: Callable[[S], float]
    ) -> float:
        """Simulation/rollout phase."""
        
        current_state = node.state
        actions_taken = []
        depth = 0
        
        # Random rollout até terminal ou max depth
        while (not is_terminal(current_state) and 
               depth < self.config.simulation_depth):
            
            legal_actions = get_legal_actions(current_state)
            if not legal_actions:
                break
                
            # Seleção aleatória de ação (pode ser melhorada)
            action = random.choice(legal_actions)
            actions_taken.append(action)
            
            current_state = apply_action(current_state, action)
            depth += 1
        
        # Avalia estado final
        final_value = evaluate_state(current_state)
        
        # Update RAVE statistics se habilitado
        if self.config.use_rapid_action_value_estimation:
            for action in actions_taken:
                if action not in self.action_values:
                    self.action_values[action] = 0.0
                    self.action_counts[action] = 0
                
                self.action_values[action] += final_value
                self.action_counts[action] += 1
        
        return final_value
    
    def _backpropagate(self, node: TreeNode[S], value: float) -> None:
        """Backpropagation phase."""
        
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
    
    def _should_terminate_early(self) -> bool:
        """Verifica se deve terminar busca antecipadamente."""
        
        if not self.root or not self.root.children:
            return False
        
        # Se um filho tem muito mais visitas que os outros
        visits = [child.visits for child in self.root.children]
        max_visits = max(visits)
        total_visits = sum(visits)
        
        if total_visits < 50:  # Muito cedo para decidir
            return False
        
        # Porcentagem das visitas no melhor filho
        best_ratio = max_visits / total_visits
        
        return best_ratio >= self.config.early_termination_threshold
    
    def get_action_distribution(self, temperature: float = 1.0) -> Dict[A, float]:
        """Retorna distribuição de probabilidades das ações."""
        
        if not self.root or not self.root.children:
            return {}
        
        visits = [child.visits for child in self.root.children]
        actions = [child.action_taken for child in self.root.children]
        
        if temperature == 0:
            # Greedy
            best_idx = visits.index(max(visits))
            return {actions[i]: 1.0 if i == best_idx else 0.0 
                   for i in range(len(actions))}
        
        # Temperature scaling
        if temperature != 1.0:
            visits = [v ** (1.0 / temperature) for v in visits]
        
        total = sum(visits)
        if total == 0:
            return {action: 1.0 / len(actions) for action in actions}
        
        return {actions[i]: visits[i] / total for i in range(len(actions))}
    
    def get_principal_variation(self, max_depth: int = 5) -> List[A]:
        """Retorna variação principal (sequência de melhores jogadas)."""
        
        if not self.root:
            return []
        
        pv = []
        current = self.root
        depth = 0
        
        while current.children and depth < max_depth:
            best_child = max(current.children, key=lambda c: c.visits)
            if best_child.action_taken:
                pv.append(best_child.action_taken)
            current = best_child
            depth += 1
        
        return pv
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da busca."""
        
        if not self.root:
            return {"error": "No search performed"}
        
        total_nodes = self._count_nodes(self.root)
        max_depth = self._get_max_depth(self.root)
        
        # Tree statistics
        stats = {
            "total_nodes": total_nodes,
            "max_depth_reached": max_depth,
            "root_visits": self.root.visits,
            "root_value": self.root.mean_value,
        }
        
        # Action statistics
        if self.root.children:
            action_stats = []
            for child in sorted(self.root.children, key=lambda c: c.visits, reverse=True):
                action_stats.append({
                    "action": str(child.action_taken),
                    "visits": child.visits,
                    "win_rate": child.mean_value,
                    "visit_percentage": child.visits / self.root.visits * 100
                })
            
            stats["actions"] = action_stats[:5]  # Top 5 actions
        
        # RAVE statistics se habilitado
        if self.config.use_rapid_action_value_estimation:
            stats["rave_actions"] = len(self.action_values)
            stats["rave_total_counts"] = sum(self.action_counts.values())
        
        return stats
    
    def _count_nodes(self, node: TreeNode[S]) -> int:
        """Conta total de nós na árvore."""
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _get_max_depth(self, node: TreeNode[S]) -> int:
        """Calcula profundidade máxima alcançada."""
        if not node.children:
            return 0
        return 1 + max(self._get_max_depth(child) for child in node.children)