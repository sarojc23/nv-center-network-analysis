import networkx as nx
import matplotlib.pyplot as plt

# def plot_network(G):
#     pos = nx.spring_layout(G)
#     weights = [G[u][v]['weight'] for u, v in G.edges()]
    
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos, edges=G.edges(), width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, with_labels=True)
#     plt.show()

import matplotlib.pyplot as plt
import networkx as nx

def plot_correlation_matrix(correlation_matrix):
    plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.show()

def plot_network(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='grey')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
