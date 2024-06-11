import networkx as nx
import matplotlib.pyplot as plt

def plot_network(G):
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, edges=G.edges(), width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()
