# -*- coding: utf-8 -*-
import networkx as nx
from pipeline.kec_metrics import compute_kec_metrics

def test_metrics_on_small_graph():
    G = nx.erdos_renyi_graph(50, 0.1, seed=1)
    m = compute_kec_metrics(G, sigma_q=True)
    assert "H_spectral" in m and m["H_spectral"] >= 0.0
    assert "k_forman_mean" in m
    assert "sigma" in m and "swp" in m
    assert "sigma_Q" in m
