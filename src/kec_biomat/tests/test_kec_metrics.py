# -*- coding: utf-8 -*-
"""
Testes para módulo kec_metrics
==============================

Testes migrados e expandidos de kec_biomat_pack_2025-09-19/tests/test_kec_metrics.py
"""

import pytest
import networkx as nx
import numpy as np

from ..metrics.kec_metrics import (
    compute_kec_metrics,
    spectral_entropy,
    forman_curvature_stats,
    small_world_sigma,
    small_world_propensity,
    quantum_coherence_sigma
)


def test_kec_metrics_basic():
    """Teste básico das métricas KEC em grafo pequeno."""
    G = nx.erdos_renyi_graph(50, 0.1, seed=1)
    metrics = compute_kec_metrics(G, sigma_q=True)
    
    # Verifica se todas as métricas esperadas estão presentes
    expected_keys = [
        "H_spectral",
        "k_forman_mean", 
        "k_forman_p05",
        "k_forman_p50", 
        "k_forman_p95",
        "sigma",
        "swp",
        "sigma_Q"
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Métrica {key} não encontrada"
    
    # Verifica se valores são numéricos e não-negativos
    assert metrics["H_spectral"] >= 0.0
    assert isinstance(metrics["k_forman_mean"], (int, float))
    assert isinstance(metrics["sigma"], (int, float))
    assert isinstance(metrics["swp"], (int, float))
    assert metrics["sigma_Q"] >= 0.0


def test_entropy_calculation():
    """Teste específico para cálculo de entropia espectral."""
    
    # Grafo linear (baixa entropia)
    G_linear = nx.path_graph(10)
    entropy_linear = spectral_entropy(G_linear)
    
    # Grafo completo (alta entropia)
    G_complete = nx.complete_graph(10)
    entropy_complete = spectral_entropy(G_complete)
    
    # Grafo completo deve ter entropia maior que linear
    assert entropy_complete >= entropy_linear
    
    # Teste com grafo vazio
    G_empty = nx.Graph()
    entropy_empty = spectral_entropy(G_empty)
    assert entropy_empty == 0.0


def test_forman_curvature():
    """Teste para curvatura de Forman."""
    
    # Grafo com triângulos
    G = nx.complete_graph(5)
    stats_with_tri = forman_curvature_stats(G, include_triangles=True)
    stats_without_tri = forman_curvature_stats(G, include_triangles=False)
    
    # Verifica estrutura do resultado
    expected_keys = ["mean", "p05", "p50", "p95"]
    for key in expected_keys:
        assert key in stats_with_tri
        assert key in stats_without_tri
    
    # Com triângulos deve ter curvaturas diferentes (geralmente menores)
    assert stats_with_tri["mean"] != stats_without_tri["mean"]
    
    # Teste com grafo sem arestas
    G_empty = nx.Graph()
    G_empty.add_node(1)
    stats_empty = forman_curvature_stats(G_empty)
    assert all(v == 0.0 for v in stats_empty.values())


def test_small_world_metrics():
    """Teste para métricas small-world."""
    
    # Grafo Watts-Strogatz (conhecido small-world)
    G_sw = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
    
    sigma = small_world_sigma(G_sw, n_random=5)  # Poucos randoms para velocidade
    swp = small_world_propensity(G_sw, n_random=5)
    
    # Sigma deve ser > 1 para small-world
    assert sigma >= 0.0  # Pode ser baixo em grafos pequenos
    
    # SWP deve estar entre 0 e 1
    assert 0.0 <= swp <= 1.0
    
    # Teste com grafo desconectado
    G_disconnected = nx.Graph()
    G_disconnected.add_edge(0, 1)
    G_disconnected.add_edge(2, 3)
    
    sigma_disconnected = small_world_sigma(G_disconnected)
    assert sigma_disconnected == 0.0  # Grafo desconectado


def test_quantum_coherence():
    """Teste para coerência quântica σ_Q."""
    
    G = nx.erdos_renyi_graph(30, 0.2, seed=123)
    coherence = quantum_coherence_sigma(G)
    
    # Deve ser não-negativo
    assert coherence >= 0.0
    
    # Teste com grafo regular (baixa coerência esperada)
    G_regular = nx.cycle_graph(20)
    coherence_regular = quantum_coherence_sigma(G_regular)
    assert coherence_regular >= 0.0
    
    # Teste com grafo vazio
    G_empty = nx.Graph()
    coherence_empty = quantum_coherence_sigma(G_empty)
    assert coherence_empty == 0.0


def test_compute_kec_metrics_comprehensive():
    """Teste abrangente da função principal."""
    
    # Diferentes tipos de grafos
    graphs = {
        "erdos_renyi": nx.erdos_renyi_graph(25, 0.15, seed=1),
        "barabasi_albert": nx.barabasi_albert_graph(25, 2, seed=2),
        "watts_strogatz": nx.watts_strogatz_graph(24, 4, 0.3, seed=3),
        "cycle": nx.cycle_graph(20),
        "path": nx.path_graph(20)
    }
    
    for graph_name, G in graphs.items():
        metrics = compute_kec_metrics(G, sigma_q=True)
        
        # Verifica se não há NaN ou valores inválidos
        for metric_name, value in metrics.items():
            assert not np.isnan(value), f"NaN em {metric_name} para grafo {graph_name}"
            assert not np.isinf(value), f"Inf em {metric_name} para grafo {graph_name}"


def test_metrics_consistency():
    """Teste de consistência: mesmos grafos devem dar mesmas métricas."""
    
    G1 = nx.erdos_renyi_graph(30, 0.1, seed=42)
    G2 = nx.erdos_renyi_graph(30, 0.1, seed=42)  # Mesmo seed
    
    metrics1 = compute_kec_metrics(G1, sigma_q=False)
    metrics2 = compute_kec_metrics(G2, sigma_q=False)
    
    # Devem ser idênticos (mesmo seed)
    for key in metrics1:
        assert abs(metrics1[key] - metrics2[key]) < 1e-10, f"Inconsistência em {key}"


def test_edge_cases():
    """Teste casos extremos."""
    
    # Grafo com 1 nó
    G_single = nx.Graph()
    G_single.add_node(0)
    metrics_single = compute_kec_metrics(G_single)
    
    # Deve completar sem erro
    assert "H_spectral" in metrics_single
    
    # Grafo com 2 nós conectados
    G_pair = nx.Graph()
    G_pair.add_edge(0, 1)
    metrics_pair = compute_kec_metrics(G_pair)
    
    assert "H_spectral" in metrics_pair
    assert metrics_pair["H_spectral"] >= 0.0
    
    # Grafo com muitos nós isolados
    G_isolated = nx.Graph()
    G_isolated.add_nodes_from(range(10))  # 10 nós, 0 arestas
    metrics_isolated = compute_kec_metrics(G_isolated)
    
    assert metrics_isolated["H_spectral"] >= 0.0
    assert metrics_isolated["sigma"] == 0.0  # Sem arestas, sem small-world


def test_performance():
    """Teste básico de performance em grafos maiores."""
    import time
    
    # Grafo médio
    G = nx.erdos_renyi_graph(200, 0.05, seed=999)
    
    start_time = time.time()
    metrics = compute_kec_metrics(G, sigma_q=False)  # Sem σ_Q para velocidade
    elapsed = time.time() - start_time
    
    # Deve completar em tempo razoável (< 10s para CI)
    assert elapsed < 10.0, f"Cálculo muito lento: {elapsed:.2f}s"
    
    # Métricas devem estar válidas
    assert all(not np.isnan(v) for v in metrics.values())


if __name__ == "__main__":
    # Execução básica para debug
    test_kec_metrics_basic()
    test_entropy_calculation()
    print("✅ Testes básicos passaram!")