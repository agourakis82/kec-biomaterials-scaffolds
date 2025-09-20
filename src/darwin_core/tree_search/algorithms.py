"""
Search Algorithms - Algoritmos de Busca e Avaliadores
====================================================

Algoritmos complementares de busca e interfaces para avaliadores de estado.
"""

from __future__ import annotations

import abc
import math
import random
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic, Callable
import logging
import asyncio

logger = logging.getLogger(__name__)

S = TypeVar("S")  # State type


class StateEvaluator(Generic[S], abc.ABC):
    """Interface abstrata para avaliadores de estado."""
    
    @abc.abstractmethod
    async def expand(self, state: S) -> List[Tuple[str, S, float]]:
        """
        Expande estado retornando possíveis ações.
        
        Returns:
            Lista de (action_name, new_state, prior_probability)
        """
        pass
    
    @abc.abstractmethod
    async def rollout(self, state: S, max_steps: int = 10) -> float:
        """
        Executa rollout simulado a partir do estado.
        
        Returns:
            Valor estimado do estado (0.0 a 1.0)
        """
        pass
    
    async def evaluate_terminal(self, state: S) -> float:
        """Avalia estado terminal. Default: usa rollout."""
        return await self.rollout(state, max_steps=0)


class StringStateEvaluator(StateEvaluator[str]):
    """Avaliador exemplo para estados em string (para testes)."""
    
    def __init__(self, branching_factor: int = 3, max_length: int = 10):
        self.branching_factor = branching_factor
        self.max_length = max_length
        
    async def expand(self, state: str) -> List[Tuple[str, str, float]]:
        """Expande adicionando caracteres/tokens."""
        
        if len(state) >= self.max_length:
            return []  # Terminal state
            
        actions = []
        for i in range(self.branching_factor):
            token = chr(ord('A') + i)
            action_name = f"add_{token}"
            new_state = f"{state}{token}"
            prior_prob = 1.0 / self.branching_factor
            
            actions.append((action_name, new_state, prior_prob))
            
        return actions
    
    async def rollout(self, state: str, max_steps: int = 10) -> float:
        """Rollout baseado no hash da string."""
        # Simula avaliação determinística
        score = (hash(state) % 1000) / 1000.0
        
        # Penaliza strings muito longas
        length_penalty = max(0, len(state) - self.max_length / 2) * 0.1
        
        return max(0.0, min(1.0, score - length_penalty))


class NumericStateEvaluator(StateEvaluator[float]):
    """Avaliador para otimização numérica."""
    
    def __init__(self, objective_func: Callable[[float], float], 
                 bounds: Tuple[float, float] = (-10.0, 10.0),
                 step_size: float = 1.0):
        self.objective_func = objective_func
        self.bounds = bounds
        self.step_size = step_size
        
    async def expand(self, state: float) -> List[Tuple[str, float, float]]:
        """Expande com passos incrementais."""
        actions = []
        
        # Ações possíveis: aumentar, diminuir, ou pequenos ajustes
        deltas = [-self.step_size, -self.step_size/2, self.step_size/2, self.step_size]
        
        for delta in deltas:
            new_state = state + delta
            
            # Verifica bounds
            if self.bounds[0] <= new_state <= self.bounds[1]:
                action_name = f"delta_{delta:+.2f}"
                # Prior baseado na distância do centro
                center = sum(self.bounds) / 2
                distance_from_center = abs(new_state - center)
                max_distance = (self.bounds[1] - self.bounds[0]) / 2
                prior = 1.0 - (distance_from_center / max_distance)
                
                actions.append((action_name, new_state, max(0.1, prior)))
        
        return actions
    
    async def rollout(self, state: float, max_steps: int = 10) -> float:
        """Avalia usando função objetivo."""
        try:
            value = self.objective_func(state)
            # Normaliza para [0, 1]
            return max(0.0, min(1.0, (value + 10) / 20))  # Assume range [-10, 10]
        except Exception:
            return 0.0


class SearchAlgorithms:
    """Coleção de algoritmos de busca utilitários."""
    
    @staticmethod
    async def beam_search(
        initial_state: S,
        evaluator: StateEvaluator[S],
        beam_width: int = 5,
        max_depth: int = 10
    ) -> List[Tuple[S, float, List[str]]]:
        """
        Beam Search para encontrar top-k caminhos.
        
        Returns:
            Lista de (final_state, score, action_sequence)
        """
        
        # Beam atual: (state, score, action_path)
        current_beam = [(initial_state, 0.0, [])]
        
        for depth in range(max_depth):
            next_beam = []
            
            for state, current_score, action_path in current_beam:
                # Expande estado
                expansions = await evaluator.expand(state)
                
                if not expansions:  # Estado terminal
                    score = await evaluator.evaluate_terminal(state)
                    next_beam.append((state, score, action_path))
                else:
                    for action, new_state, prior in expansions:
                        # Score combinado
                        new_score = current_score + prior
                        new_path = action_path + [action]
                        
                        next_beam.append((new_state, new_score, new_path))
            
            # Mantém apenas top beam_width
            next_beam.sort(key=lambda x: x[1], reverse=True)
            current_beam = next_beam[:beam_width]
            
            if not current_beam:
                break
        
        return current_beam
    
    @staticmethod
    async def greedy_search(
        initial_state: S,
        evaluator: StateEvaluator[S],
        max_depth: int = 10
    ) -> Tuple[S, float, List[str]]:
        """
        Busca greedy - sempre escolhe melhor ação local.
        
        Returns:
            (final_state, final_score, action_sequence)
        """
        
        current_state = initial_state
        action_path = []
        total_score = 0.0
        
        for depth in range(max_depth):
            expansions = await evaluator.expand(current_state)
            
            if not expansions:  # Terminal
                final_score = await evaluator.evaluate_terminal(current_state)
                return current_state, final_score, action_path
            
            # Escolhe melhor ação (maior prior)
            best_action, best_state, best_prior = max(expansions, key=lambda x: x[2])
            
            current_state = best_state
            action_path.append(best_action)
            total_score += best_prior
        
        final_score = await evaluator.rollout(current_state)
        return current_state, final_score, action_path
    
    @staticmethod
    async def random_walk(
        initial_state: S,
        evaluator: StateEvaluator[S],
        max_steps: int = 100,
        num_walks: int = 10
    ) -> List[Tuple[S, float, List[str]]]:
        """
        Multiple random walks para exploração.
        
        Returns:
            Lista de caminhos aleatórios com scores
        """
        
        results = []
        
        for walk in range(num_walks):
            current_state = initial_state
            action_path = []
            
            for step in range(max_steps):
                expansions = await evaluator.expand(current_state)
                
                if not expansions:  # Terminal
                    break
                
                # Escolha aleatória ponderada por prior
                actions, states, priors = zip(*expansions)
                
                # Normaliza priors
                total_prior = sum(priors)
                if total_prior > 0:
                    weights = [p / total_prior for p in priors]
                else:
                    weights = [1.0 / len(priors)] * len(priors)
                
                # Amostra ação
                choice_idx = random.choices(range(len(actions)), weights=weights)[0]
                
                current_state = states[choice_idx]
                action_path.append(actions[choice_idx])
            
            # Avalia estado final
            final_score = await evaluator.rollout(current_state)
            results.append((current_state, final_score, action_path))
        
        return results
    
    @staticmethod
    def calculate_diversity(paths: List[List[str]]) -> float:
        """Calcula diversidade entre caminhos de ação."""
        if len(paths) < 2:
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                similarity = SearchAlgorithms._path_similarity(paths[i], paths[j])
                total_similarity += similarity
                comparisons += 1
        
        avg_similarity = total_similarity / max(1, comparisons)
        return 1.0 - avg_similarity  # Diversidade é o inverso da similaridade
    
    @staticmethod
    def _path_similarity(path1: List[str], path2: List[str]) -> float:
        """Calcula similaridade entre dois caminhos."""
        if not path1 and not path2:
            return 1.0
        
        if not path1 or not path2:
            return 0.0
        
        # Jaccard similarity
        set1 = set(path1)
        set2 = set(path2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / max(1, union)
    
    @staticmethod
    async def adaptive_search(
        initial_state: S,
        evaluator: StateEvaluator[S],
        time_budget: float = 5.0,
        exploration_rate: float = 0.3
    ) -> Tuple[S, float, List[str]]:
        """
        Busca adaptativa que ajusta estratégia baseada no tempo.
        
        Args:
            time_budget: Tempo total em segundos
            exploration_rate: Taxa de exploração vs exploitation
        """
        
        import time
        start_time = time.time()
        
        # Fase 1: Exploração rápida (30% do tempo)
        exploration_time = time_budget * exploration_rate
        
        logger.info("Iniciando fase de exploração")
        beam_results = await SearchAlgorithms.beam_search(
            initial_state, evaluator, beam_width=10, max_depth=5
        )
        
        elapsed = time.time() - start_time
        remaining_time = time_budget - elapsed
        
        if remaining_time <= 0 or not beam_results:
            # Retorna melhor resultado do beam search
            best_result = max(beam_results, key=lambda x: x[1]) if beam_results else (initial_state, 0.0, [])
            return best_result
        
        # Fase 2: Exploitation dos melhores estados
        logger.info("Iniciando fase de exploitation")
        best_states = [result[0] for result in beam_results[:3]]  # Top 3
        
        final_results = []
        
        for state in best_states:
            if time.time() - start_time >= time_budget:
                break
                
            # Busca greedy a partir deste estado
            result = await SearchAlgorithms.greedy_search(state, evaluator, max_depth=8)
            final_results.append(result)
        
        # Retorna melhor resultado final
        if final_results:
            return max(final_results, key=lambda x: x[1])
        else:
            return max(beam_results, key=lambda x: x[1])


# Factory functions para casos comuns

async def optimize_function(
    func: Callable[[float], float],
    bounds: Tuple[float, float] = (-10.0, 10.0),
    method: str = "puct",
    budget: int = 500
) -> Tuple[float, float]:
    """
    Otimiza função usando busca em árvore.
    
    Args:
        func: Função objetivo a maximizar
        bounds: Limites do domínio
        method: "puct", "mcts", "beam", "greedy"
        budget: Budget de busca
        
    Returns:
        (best_x, best_value)
    """
    
    evaluator = NumericStateEvaluator(func, bounds)
    initial_state = sum(bounds) / 2  # Começa no centro
    
    if method == "beam":
        results = await SearchAlgorithms.beam_search(
            initial_state, evaluator, beam_width=10, max_depth=budget//10
        )
        best_state, best_score, _ = max(results, key=lambda x: x[1])
        return best_state, func(best_state)
        
    elif method == "greedy":
        best_state, best_score, _ = await SearchAlgorithms.greedy_search(
            initial_state, evaluator, max_depth=budget//5
        )
        return best_state, func(best_state)
        
    else:
        # Default: adaptive search
        best_state, best_score, _ = await SearchAlgorithms.adaptive_search(
            initial_state, evaluator, time_budget=2.0
        )
        return best_state, func(best_state)