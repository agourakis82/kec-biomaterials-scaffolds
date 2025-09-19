# -*- coding: utf-8 -*-
"""
KEC 2.0 — Métricas de rede para scaffolds porosos
-------------------------------------------------
Implementa:
- Entropia espectral (von Neumann) do Laplaciano normalizado
- Curvatura de Forman (2-complex, com triângulos)
- Small-world σ e Small-World Propensity (SWP)
- Coerência “quântica” opcional: S(diag ρ) − S(ρ)

Nota: Ollivier–Ricci aproximada (Sinkhorn/amostragem) deve ser chamada via
biblioteca GraphRicciCurvature quando disponível. Aqui mantemos *hooks*.
"""
from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Iterable

import numpy as np
import networkx as nx
from numpy.typing import ArrayLike
try:
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh
except Exception as e:
    eigsh = None  # fallback se SciPy não estiver disponível

# ---------- Helpers ----------

def _normalized_laplacian(G: nx.Graph):
    """Retorna Laplaciano normalizado como matriz esparsa (scipy) ou densa (numpy)."""
    try:
        return nx.normalized_laplacian_matrix(G)
    except Exception:
        # Fallback: construir denso (para grafos pequenos)
        A = nx.to_numpy_array(G)
        d = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0/np.sqrt(np.maximum(d, 1e-12)))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

# ---------- Entropia ----------

def spectral_entropy(G: nx.Graph, k_eigs: int = 64, tol: float = 1e-8) -> float:
    """
    Entropia espectral (von Neumann) baseada no Laplaciano normalizado.

    S(ρ) = - sum_i p_i log(p_i), onde ρ ~ L / Tr(L)  (autovalores normalizados).
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    L = _normalized_laplacian(G)
    # Autovalores (positivos, com 0 multiplicidade = nº componentes)
    try:
        if eigsh is not None and hasattr(L, "shape") and min(L.shape) > 128:
            k = min(max(2, k_eigs), n-1)
            vals = np.abs(eigsh(L, k=k, which="LM", return_eigenvectors=False, tol=tol))
            # Aproxima com top-k; normalizar
            s = np.sum(vals)
            p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
        else:
            # Pequeno: autovalores completos
            if hasattr(L, "toarray"):
                L = L.toarray()
            vals = np.linalg.eigvalsh(L)
            vals = np.abs(vals)
            s = np.sum(vals)
            p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
    except Exception:
        # Fallback: grau como proxy
        deg = np.array([d for _, d in G.degree()], dtype=float)
        p = deg / np.sum(deg) if deg.sum() > 0 else np.ones(len(deg))/len(deg)
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))

# ---------- Curvatura de Forman (2-complex) ----------

def forman_curvature_stats(G: nx.Graph, include_triangles: bool = True) -> Dict[str, float]:
    """
    Curvatura de Forman por aresta (approx). Retorna estatísticas globais.
    Fórmula simplificada: F(u,v) ≈ deg(u) + deg(v) - 2 - t(u,v), onde t=triângulos incidentes.
    """
    if G.number_of_edges() == 0:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    deg = dict(G.degree())
    # Contagem de triângulos por aresta (aprox): interseção de vizinhos
    def edge_tri(u, v):
        if not include_triangles:
            return 0
        Nu = set(G.neighbors(u)) - {v}
        Nv = set(G.neighbors(v)) - {u}
        return len(Nu & Nv)
    vals = []
    for u, v in G.edges():
        t = edge_tri(u, v)
        f = (deg[u] + deg[v] - 2) - t
        vals.append(f)
    a = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }

# ---------- Small-world σ e SWP ----------

def small_world_sigma(G: nx.Graph, n_random: int = 20) -> float:
    """Humphries & Gurney σ = (C/C_rand) / (L/L_rand)."""
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    try:
        C = nx.average_clustering(G)
        L = nx.average_shortest_path_length(max(nx.connected_components(G), key=len))
        # Rede aleatória equivalente
        Cr, Lr = 0.0, 0.0
        for _ in range(max(1, n_random)):
            Gr = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
            Cr += nx.average_clustering(Gr)
            # usar maior componente
            Lr += nx.average_shortest_path_length(max(nx.connected_components(Gr), key=len))
        Cr /= max(1, n_random)
        Lr /= max(1, n_random)
        if Cr == 0 or Lr == 0:
            return 0.0
        return float((C/Cr) / (L/Lr))
    except Exception:
        return 0.0

def small_world_propensity(G: nx.Graph, n_random: int = 20) -> float:
    """
    SWP (Muldoon et al., 2016): 0..1, corrige viés de densidade.
    Implementação simplificada (proxy) — suficiente para ranking/comparação.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    try:
        C = nx.average_clustering(G)
        # rede reg (lattice) proxy: caminho médio alto, clustering alto
        # estimativas aproximadas:
        n = G.number_of_nodes()
        k_bar = float(np.mean([d for _, d in G.degree()]))
        C_latt = min(1.0, (3*(k_bar-2)) / (4*(k_bar-1)+1e-9)) if k_bar >= 2 else 0.0
        C_rand = k_bar / (n-1) if n > 1 else 0.0
        # normalizações
        C_norm = (C - C_rand) / (C_latt - C_rand + 1e-12)
        C_norm = float(np.clip(C_norm, 0.0, 1.0))
        # path length normalizado (proxy)
        L = nx.average_shortest_path_length(max(nx.connected_components(G), key=len))
        L_latt = n / (2*max(1.0, k_bar))  # proxy de lattice
        L_rand = (np.log(n) / np.log(max(2.0, k_bar))) if k_bar > 1 else L
        L_norm = (L_rand - L) / (L_rand - L_latt + 1e-12)
        L_norm = float(np.clip(L_norm, 0.0, 1.0))
        # SWP como média simples dos dois normalizados
        return float((C_norm + L_norm)/2.0)
    except Exception:
        return 0.0

# ---------- Coerência “quântica” opcional ----------

def quantum_coherence_sigma(G: nx.Graph, k_eigs: int = 64, tol: float = 1e-8) -> float:
    """
    σ_Q = S(diag ρ) − S(ρ), onde ρ deriva do Laplaciano normalizado (NDM-like).
    diag ρ ~ distribuição de graus normalizada (proxy). S(.) é entropia de Shannon.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    # S(ρ): entropia espectral
    S_spectral = spectral_entropy(G, k_eigs=k_eigs, tol=tol)
    # S(diag ρ): graus normalizados
    deg = np.array([d for _, d in G.degree()], dtype=float)
    p = deg/deg.sum() if deg.sum() > 0 else np.ones(n)/n
    p = p[p > 1e-15]
    S_diag = float(-np.sum(p*np.log(p)))
    return float(max(0.0, S_diag - S_spectral))

# ---------- API principal ----------

def compute_kec_metrics(G: nx.Graph,
                        spectral_k: int = 64,
                        include_triangles: bool = True,
                        n_random: int = 20,
                        sigma_q: bool = False) -> Dict[str, Any]:
    """Calcula H/κ/σ/ϕ (+ σ_Q opcional) para um grafo de poros."""
    H = spectral_entropy(G, k_eigs=spectral_k)
    k_stats = forman_curvature_stats(G, include_triangles=include_triangles)
    sigma = small_world_sigma(G, n_random=n_random)
    swp = small_world_propensity(G, n_random=n_random)
    out = {
        "H_spectral": H,
        "k_forman_mean": k_stats["mean"],
        "k_forman_p05": k_stats["p05"],
        "k_forman_p50": k_stats["p50"],
        "k_forman_p95": k_stats["p95"],
        "sigma": sigma,
        "swp": swp,
    }
    if sigma_q:
        out["sigma_Q"] = quantum_coherence_sigma(G, k_eigs=spectral_k)
    return out

if __name__ == "__main__":
    # Demo rápido: grafo esparso
    G = nx.erdos_renyi_graph(300, 0.02, seed=42)
    print(compute_kec_metrics(G, sigma_q=True))
