"""
Iterative Search - Busca Iterativa e Refinamento
================================================

Implementação de busca iterativa com refinamento automático de queries
e raciocínio step-by-step para RAG++.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SearchStep:
    """Um passo no processo de busca iterativa."""
    step_id: int
    thought: str
    action: str
    query: str
    results: List[Dict[str, Any]]
    score: float
    reasoning: str


@dataclass
class IterativeConfig:
    """Configuração para busca iterativa."""
    max_iterations: int = 5
    min_score_threshold: float = 0.7
    convergence_threshold: float = 0.1
    enable_query_expansion: bool = True
    enable_result_filtering: bool = True


class IterativeSearch:
    """
    Motor de busca iterativa que:
    - Refina queries baseado em resultados anteriores
    - Usa raciocínio step-by-step
    - Avalia qualidade dos resultados
    - Converge para resposta optimal
    """
    
    def __init__(self, config: IterativeConfig, rag_engine=None):
        self.config = config
        self.rag_engine = rag_engine
        self.search_history: List[SearchStep] = []
        
    async def search_iteratively(
        self, 
        initial_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executa busca iterativa com refinamento automático.
        
        Args:
            initial_query: Query inicial
            context: Contexto adicional para refinamento
            
        Returns:
            Resultado final com histórico de steps
        """
        self.search_history = []
        current_query = initial_query
        best_results = []
        best_score = 0.0
        
        logger.info(f"Iniciando busca iterativa para: '{initial_query}'")
        
        for iteration in range(self.config.max_iterations):
            step = await self._execute_search_step(
                iteration + 1,
                current_query,
                context,
                previous_results=best_results
            )
            
            self.search_history.append(step)
            
            # Avalia se este step teve resultados melhores
            if step.score > best_score:
                best_results = step.results
                best_score = step.score
                
            # Verifica convergência
            if self._has_converged(step):
                logger.info(f"Converged após {iteration + 1} iterações")
                break
                
            # Refina query para próxima iteração
            current_query = self._refine_query(current_query, step, context)
            
        # Compila resultado final
        return {
            "answer": self._generate_final_answer(best_results),
            "sources": best_results,
            "method": "iterative_search",
            "iterations": len(self.search_history),
            "final_score": best_score,
            "reasoning_steps": [
                {
                    "step": step.step_id,
                    "thought": step.thought,
                    "action": step.action,
                    "query": step.query,
                    "results_count": len(step.results),
                    "score": step.score
                }
                for step in self.search_history
            ],
            "converged": best_score >= self.config.min_score_threshold
        }
    
    async def _execute_search_step(
        self,
        step_id: int,
        query: str,
        context: Optional[Dict[str, Any]],
        previous_results: List[Dict[str, Any]]
    ) -> SearchStep:
        """Executa um passo individual da busca."""
        
        # Gera thought/reasoning para este step
        thought = self._generate_thought(step_id, query, previous_results)
        action = self._determine_action(step_id, query, previous_results)
        
        logger.debug(f"Step {step_id} - Thought: {thought}, Action: {action}")
        
        # Executa busca
        if self.rag_engine:
            results = await self.rag_engine.query_knowledge_base(query, top_k=10)
        else:
            # Fallback para desenvolvimento
            results = []
            
        # Filtra e ranqueia resultados se habilitado
        if self.config.enable_result_filtering:
            results = self._filter_results(results, query, previous_results)
            
        # Calcula score para este step
        score = self._calculate_step_score(results, query, previous_results)
        
        return SearchStep(
            step_id=step_id,
            thought=thought,
            action=action,
            query=query,
            results=results,
            score=score,
            reasoning=f"Executed {action} with query '{query}', found {len(results)} results"
        )
    
    def _generate_thought(
        self, 
        step_id: int, 
        query: str, 
        previous_results: List[Dict[str, Any]]
    ) -> str:
        """Gera reasoning thought para o step atual."""
        
        if step_id == 1:
            return f"Initial search for '{query}'"
        elif len(previous_results) == 0:
            return f"Previous search yielded no results, trying broader terms"
        elif len(previous_results) > 20:
            return f"Too many results ({len(previous_results)}), narrowing search"
        else:
            return f"Refining search based on {len(previous_results)} previous results"
    
    def _determine_action(
        self, 
        step_id: int, 
        query: str, 
        previous_results: List[Dict[str, Any]]
    ) -> str:
        """Determina ação para o step atual."""
        
        if step_id == 1:
            return "initial_search"
        elif len(previous_results) == 0:
            return "expand_query"
        elif len(previous_results) > 15:
            return "narrow_query"
        else:
            return "refine_query"
    
    def _refine_query(
        self, 
        current_query: str, 
        step: SearchStep, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Refina query baseado no step anterior."""
        
        if not self.config.enable_query_expansion:
            return current_query
            
        if step.action == "expand_query":
            # Adiciona termos mais gerais
            terms = current_query.split()
            if len(terms) > 1:
                return " ".join(terms[:-1])  # Remove último termo
            else:
                return current_query + " overview"
                
        elif step.action == "narrow_query":
            # Adiciona termos mais específicos
            return current_query + " specific detailed"
            
        elif step.action == "refine_query" and step.results:
            # Extrai termos relevantes dos resultados
            common_terms = self._extract_common_terms(step.results)
            if common_terms:
                return f"{current_query} {common_terms[0]}"
                
        return current_query
    
    def _filter_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        previous_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filtra e ranqueia resultados."""
        
        if not results:
            return results
            
        # Remove duplicatas baseado em previous_results
        seen_ids = {r.get("id") for r in previous_results}
        filtered = [r for r in results if r.get("id") not in seen_ids]
        
        # Ranqueia por score (se disponível) ou por relevância simples
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Retorna top resultados
        return filtered[:8]
    
    def _calculate_step_score(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        previous_results: List[Dict[str, Any]]
    ) -> float:
        """Calcula score de qualidade para o step."""
        
        if not results:
            return 0.0
            
        # Score baseado em quantidade e qualidade dos resultados
        quantity_score = min(len(results) / 5.0, 1.0)  # Ideal: 5+ results
        
        # Score baseado em scores individuais (se disponível)
        avg_score = sum(r.get("score", 0.5) for r in results) / len(results)
        
        # Penalty por duplicatas com previous_results
        previous_ids = {r.get("id") for r in previous_results}
        duplicates = sum(1 for r in results if r.get("id") in previous_ids)
        duplicate_penalty = duplicates / max(len(results), 1) * 0.5
        
        final_score = (quantity_score * 0.4 + avg_score * 0.6) - duplicate_penalty
        return max(0.0, min(1.0, final_score))
    
    def _has_converged(self, step: SearchStep) -> bool:
        """Verifica se a busca convergiu."""
        
        # Converge se score é alto o suficiente
        if step.score >= self.config.min_score_threshold:
            return True
            
        # Converge se últimos 2 steps têm scores similares
        if len(self.search_history) >= 2:
            prev_score = self.search_history[-2].score
            if abs(step.score - prev_score) < self.config.convergence_threshold:
                return True
                
        return False
    
    def _extract_common_terms(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extrai termos comuns dos resultados para refinamento."""
        
        if not results:
            return []
            
        # TODO: Implementar extração real de termos
        # Por enquanto, retorna lista vazia
        return []
    
    def _generate_final_answer(self, results: List[Dict[str, Any]]) -> str:
        """Gera resposta final baseada nos melhores resultados."""
        
        if not results:
            return "Não foram encontrados resultados relevantes após busca iterativa."
            
        # TODO: Integrar com modelo de geração
        summary = f"Busca iterativa encontrou {len(results)} resultados relevantes."
        
        if results:
            top_result = results[0]
            content = top_result.get("content", "")[:200]
            summary += f" Principal resultado: {content}..."
            
        return summary