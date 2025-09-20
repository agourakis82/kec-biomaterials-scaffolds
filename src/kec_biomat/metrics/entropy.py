"""Entropy calculations - migrated from existing entropy.py"""
import numpy as np
import networkx as nx

def shannon_entropy(probabilities):
    """Shannon entropy calculation."""
    p = np.array(probabilities)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def graph_entropy(G):
    """Graph entropy based on degree distribution.""" 
    degrees = [d for _, d in G.degree()]
    if not degrees: return 0.0
    total = sum(degrees)
    probs = [d/total for d in degrees]
    return shannon_entropy(probs)