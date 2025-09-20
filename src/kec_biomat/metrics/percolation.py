"""Percolation Analysis - migrated from existing percolation.py"""
import numpy as np
import networkx as nx

def percolation_threshold(G, samples=100):
    """Estimate percolation threshold for graph."""
    thresholds = []
    for _ in range(samples):
        # Remove random edges and check connectivity
        G_copy = G.copy()
        edges = list(G_copy.edges())
        np.random.shuffle(edges)
        
        for i, edge in enumerate(edges):
            G_copy.remove_edge(*edge)
            if not nx.is_connected(G_copy):
                threshold = i / len(edges)
                thresholds.append(threshold)
                break
    
    return np.mean(thresholds) if thresholds else 1.0

def giant_component_size(G):
    """Size of largest connected component."""
    if G.number_of_nodes() == 0:
        return 0.0
    components = list(nx.connected_components(G))
    if not components:
        return 0.0
    largest = max(components, key=len)
    return len(largest) / G.number_of_nodes()