import networkx as nx
import matplotlib.pyplot as plt
import mdtraj

def visualize_comp_DAG(G: nx.DiGraph, top: mdtraj.Topology, figsize=(10,10)):
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # jitter position of nodes to avoid overlapping edges
    #     for k in pos.keys():
    #         pos[k][1] += np.random.random(1) * 0.02
    
    fig, ax = plt.subplots(figsize=figsize)
    atom_names = [a.name for a in top.atoms]
    nx.draw_networkx(G, pos, ax=ax, node_size=200, 
                     node_color=["#ffffa1" if name in ['C', 'CA', 'N'] else "#90e0ef" for name in atom_names],
                     with_labels=False)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='k', font_size=5, 
                            labels=dict([(i, str(n).replace('-', '\n')) for i, n in enumerate(top.atoms)]))
    
    return fig, ax