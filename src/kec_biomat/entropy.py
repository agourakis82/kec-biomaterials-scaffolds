import numpy as np, networkx as nx
def random_walk_entropy_rate(G: nx.Graph) -> float:
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    deg = np.array(A.sum(axis=1)).ravel()
    P = A.multiply(1.0/deg[:,None])
    pi = deg/deg.sum()
    with np.errstate(divide='ignore'):
        row_ent = -np.array(P.multiply(np.log(P.A, where=P.A>0)).sum(axis=1)).ravel()
    return float((pi * row_ent).sum())
