import networkx as nx
def percolation_diameter(G: nx.Graph, attr="throat_d") -> float:
    ds = sorted({data.get(attr, 0.0) for _,_,data in G.edges(data=True)}, reverse=True)
    for d in ds:
        H = nx.Graph((u,v,data) for u,v,data in G.edges(data=True) if data.get(attr,0.0) >= d)
        # TODO: implementar checagem “lado-a-lado” de acordo com metadados geométricos
        if H.number_of_edges() and nx.number_connected_components(H) == 1:
            return d
    return 0.0
