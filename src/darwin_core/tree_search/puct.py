"""
PUCT (Polynomial Upper Confidence Trees) Implementation
======================================================

Implementação completa do algoritmo PUCT para busca em árvore,
migrado e melhorado de kec_biomat_api.services.tree_search.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)

S = TypeVar("S")  # State type


class NodeStatus(Enum):
    """Status de um nó na árvore de busca."""
    UNVISITED = "unvisited"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    ERROR = "error"


@dataclass
class PUCTConfig:
    """Configuração para algoritmo PUCT."""
    max_budget_nodes: int = 500
    max_depth: int = 8
    default_budget: int = 200
    c_puct: float = 1.414  # Exploration constant
    expansion_threshold: int = 5
    simulation_rollouts: int = 3
    temperature: float = 1.0
    use_progressive_widening: bool = True
    alpha_widening: float = 0.5
    virtual_loss: float = 0.1  # Para paralelização
    min_visits_before_expansion: int = 1


@dataclass
class TreeNode(Generic[S]):
    """Nó em árvore de busca PUCT."""
    state: S
    parent: Optional[TreeNode[S]] = None
    action_taken: Optional[str] = None
    depth: int = 0
    visits: int = 0
    total_value: float = 0.0
    squared_value: float = 0.0  # Para calcular variância
    children: List[TreeNode[S]] = field(default_factory=list)
    status: NodeStatus = NodeStatus.UNVISITED
    prior_probability: float = 1.0
    virtual_losses: int = 0  # Para busca paralela
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Verifica se nó é terminal."""
        return self.depth <= 0 or self.status == NodeStatus.TERMINAL

    @property
    def mean_value(self) -> float:
        """Valor médio do nó."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def value_variance(self) -> float:
        """Variância dos valores do nó."""
        if self.visits < 2:
            return 0.0
        mean = self.mean_value
        return (self.squared_value / self.visits) - (mean * mean)

    @property
    def confidence_interval(self) -> float:
        """Intervalo de confiança (std error)."""
        if self.visits < 2:
            return float('inf')
        return math.sqrt(self.value_variance / self.visits)

    @property
    def node_id(self) -> str:
        """ID único do nó."""
        return f"n{hash((str(self.state), self.depth)) & 0xFFFF:04x}"

    def puct_score(self, config: PUCTConfig) -> float:
        """Calcula score PUCT para seleção de nó."""
        if not self.parent:
            return 0.0
            
        # Ajustar por virtual losses (para paralelização)
        adjusted_visits = max(1, self.visits - self.virtual_losses)
        Q = self.total_value / adjusted_visits if adjusted_visits > 0 else 0.0
        
        # Parent visits
        N = self.parent.visits if self.parent.visits > 0 else 1
        
        # Prior probability (pode ser melhorada com rede neural)
        P = self.prior_probability
        
        # Exploration term
        exploration = config.c_puct * P * math.sqrt(N) / (1 + adjusted_visits)
        
        return Q + exploration

    def ucb1_score(self) -> float:
        """Score UCB1 alternativo."""
        if not self.parent or self.visits == 0:
            return float('inf')
            
        Q = self.mean_value
        N = self.parent.visits
        n = self.visits
        
        return Q + math.sqrt(2 * math.log(N) / n)

    def update_stats(self, value: float) -> None:
        """Atualiza estatísticas do nó."""
        self.visits += 1
        self.total_value += value
        self.squared_value += value * value

    def add_virtual_loss(self) -> None:
        """Adiciona virtual loss para busca paralela."""
        self.virtual_losses += 1

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss."""
        self.virtual_losses = max(0, self.virtual_losses - 1)


class PUCTSearch(Generic[S]):
    """
    Implementação completa de PUCT (Polynomial Upper Confidence Trees).
    
    Funcionalidades:
    - Busca em árvore com balance exploration/exploitation
    - Suporte a paralelização com virtual losses
    - Estatísticas avançadas (variância, intervalos de confiança)
    - Progressive widening
    - Múltiplas políticas de seleção
    """
    
    def __init__(self, evaluator, config: Optional[PUCTConfig] = None):
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.root: Optional[TreeNode[S]] = None
        self.nodes_explored = 0
        self.search_time = 0.0
        self.nodes_cache: Dict[str, TreeNode[S]] = {}
        
    async def search(
        self, 
        initial_state: S, 
        budget: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> TreeNode[S]:
        """
        Executa busca PUCT com orçamento de nós ou tempo.
        
        Args:
            initial_state: Estado inicial
            budget: Número máximo de nós a explorar
            time_limit: Tempo limite em segundos
            
        Returns:
            Nó raiz com árvore construída
        """
        start_time = time.time()
        budget = budget or self.config.default_budget
        
        self.root = TreeNode(
            state=initial_state, 
            depth=self.config.max_depth,
            prior_probability=1.0
        )
        
        self.nodes_explored = 0
        iterations = 0
        
        logger.info(f"Iniciando busca PUCT: budget={budget}, max_depth={self.config.max_depth}")
        
        while iterations < min(budget, self.config.max_budget_nodes):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                logger.info(f"Time limit reached: {time.time() - start_time:.2f}s")
                break
                
            try:
                # Select node to expand
                node = await self._select(self.root)
                
                # Add virtual loss for parallelização (se necessário)
                node.add_virtual_loss()
                
                try:
                    # Simulate from selected node
                    value = await self._simulate(node)
                    
                    # Backpropagate results
                    self._backpropagate(node, value)
                    
                finally:
                    # Remove virtual loss
                    node.remove_virtual_loss()
                
                iterations += 1
                self.nodes_explored += 1
                
                # Log progress periodically
                if iterations % 50 == 0:
                    logger.debug(f"Search progress: {iterations}/{budget} iterations")
                    
            except Exception as e:
                logger.error(f"Error in search iteration {iterations}: {e}")
                break
        
        self.search_time = time.time() - start_time
        logger.info(f"Busca concluída: {iterations} iterações em {self.search_time:.2f}s")
        
        return self.root

    async def _select(self, node: TreeNode[S]) -> TreeNode[S]:
        """Seleciona nó folha para expansão usando PUCT."""
        
        # Se nó não foi visitado, expande
        if node.status == NodeStatus.UNVISITED:
            await self._expand(node)
            return node

        # Se é folha ou terminal, retorna
        if not node.children or node.is_terminal:
            return node

        # Progressive widening - limita número de filhos considerados
        if self.config.use_progressive_widening:
            max_children = max(1, int(self.config.alpha_widening * math.sqrt(node.visits)))
            available_children = node.children[:max_children]
        else:
            available_children = node.children

        # Seleciona filho com maior score PUCT
        if available_children:
            best_child = max(available_children, key=lambda c: c.puct_score(self.config))
            return await self._select(best_child)
        
        return node

    async def _expand(self, node: TreeNode[S]) -> None:
        """Expande nó adicionando filhos possíveis."""
        
        node.status = NodeStatus.EXPANDED
        
        if node.is_terminal:
            node.status = NodeStatus.TERMINAL
            return
        
        try:
            # Get possible actions and child states from evaluator
            children_data = await self.evaluator.expand(node.state)
            
            for action, new_state, prior_prob in children_data:
                child = TreeNode(
                    state=new_state,
                    parent=node,
                    action_taken=action,
                    depth=node.depth - 1,
                    prior_probability=max(0.001, prior_prob),  # Evita prob 0
                    status=NodeStatus.UNVISITED
                )
                
                node.children.append(child)
                
                # Cache node if useful
                if len(self.nodes_cache) < 1000:  # Limit cache size
                    self.nodes_cache[child.node_id] = child
            
            logger.debug(f"Expanded node {node.node_id} with {len(node.children)} children")
            
        except Exception as e:
            logger.error(f"Error expanding node {node.node_id}: {e}")
            node.status = NodeStatus.ERROR

    async def _simulate(self, node: TreeNode[S]) -> float:
        """Simula resultado a partir do nó (rollout)."""
        
        if node.status == NodeStatus.ERROR:
            return -1.0
            
        try:
            # Multiple rollouts para reduzir variância
            values = []
            for _ in range(self.config.simulation_rollouts):
                value = await self.evaluator.rollout(node.state, max_steps=node.depth)
                values.append(value)
            
            # Retorna média dos rollouts
            final_value = sum(values) / max(1, len(values))
            
            # Apply temperature if configured
            if self.config.temperature != 1.0:
                final_value = final_value ** (1.0 / self.config.temperature)
            
            return final_value
            
        except Exception as e:
            logger.error(f"Error in simulation for node {node.node_id}: {e}")
            return 0.0

    def _backpropagate(self, node: TreeNode[S], value: float) -> None:
        """Propaga valor para cima na árvore."""
        
        current: Optional[TreeNode[S]] = node
        
        while current is not None:
            current.update_stats(value)
            current = current.parent

    def get_best_action_sequence(self, max_length: Optional[int] = None) -> List[str]:
        """Retorna sequência de melhores ações."""
        
        if not self.root:
            return []
            
        path: List[str] = []
        node = self.root
        steps = 0
        limit = max_length or self.config.max_depth
        
        while node.children and steps < limit:
            # Escolhe filho com maior número de visitas (mais robusto)
            node = max(node.children, key=lambda c: c.visits)
            
            if node.action_taken:
                path.append(node.action_taken)
            steps += 1
            
        return path

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[str, float]:
        """Retorna probabilidades das ações baseadas em visitas."""
        
        if not self.root or not self.root.children:
            return {}
        
        visits = [child.visits for child in self.root.children]
        actions = [child.action_taken or f"action_{i}" for i, child in enumerate(self.root.children)]
        
        if temperature == 0:
            # Greedy selection
            best_idx = max(range(len(visits)), key=lambda i: visits[i])
            return {actions[i]: 1.0 if i == best_idx else 0.0 for i in range(len(actions))}
        
        # Temperature scaling
        if temperature != 1.0:
            visits = [(v ** (1.0 / temperature)) for v in visits]
        
        total = sum(visits)
        if total == 0:
            return {action: 1.0 / len(actions) for action in actions}
        
        return {actions[i]: visits[i] / total for i in range(len(actions))}

    def get_search_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da busca."""
        
        def count_nodes(n: TreeNode[S]) -> int:
            return 1 + sum(count_nodes(c) for c in n.children)

        def max_depth_reached(n: TreeNode[S]) -> int:
            if not n.children:
                return self.config.max_depth - n.depth
            return max(max_depth_reached(c) for c in n.children)

        if not self.root:
            return {
                "nodes_count": 0,
                "max_depth_reached": 0,
                "search_efficiency": 0.0,
                "search_time": self.search_time,
                "nodes_explored": self.nodes_explored
            }
        
        nodes_count = count_nodes(self.root)
        depth_reached = max_depth_reached(self.root)
        efficiency = self.nodes_explored / max(1, nodes_count)
        
        # Additional statistics
        if self.root.children:
            best_child = max(self.root.children, key=lambda c: c.visits)
            best_action_visits = best_child.visits
            total_root_visits = self.root.visits
            action_concentration = best_action_visits / max(1, total_root_visits)
        else:
            action_concentration = 0.0
        
        return {
            "nodes_count": nodes_count,
            "max_depth_reached": depth_reached,
            "search_efficiency": round(efficiency, 3),
            "search_time": round(self.search_time, 3),
            "nodes_explored": self.nodes_explored,
            "root_visits": self.root.visits,
            "root_value": round(self.root.mean_value, 3),
            "root_confidence": round(self.root.confidence_interval, 3),
            "action_concentration": round(action_concentration, 3),
            "cache_size": len(self.nodes_cache)
        }