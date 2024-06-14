import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def load_processed_data(file_path):
    # Load processed data
    counts_array = np.load(file_path)
    return counts_array

def compute_correlation_matrix(counts_array):
    counts_array = np.array(counts_array)     # Convert counts_array to a NumPy array
    counts_2d = counts_array.reshape(counts_array.shape[0], -1)     # Reshape the counts array to a 2D array
    correlation_matrix = np.corrcoef(counts_2d)     # Compute the correlation matrix
    return correlation_matrix

# def build_network(correlation_matrix, threshold=0.5):
#     # Create a graph from the correlation matrix
#     G = nx.Graph()
#     num_nodes = correlation_matrix.shape[0]
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             if abs(correlation_matrix[i, j]) > threshold:
#                 G.add_edge(i, j, weight=correlation_matrix[i, j])
#     return G

# def plot_network_update(G, title=None, decimal_places=None):
#     # Plot the network
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='grey')
#     labels = nx.get_edge_attributes(G, 'weight')
#     if decimal_places is not None:
#         labels = {k: f"{v:.{decimal_places}f}" for k, v in labels.items()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     if title:
#         plt.title(title)
#     plt.show()

def build_network(correlation_matrix, threshold=0.5):
    """
    Create a graph from the correlation matrix.
    Edges are added if the absolute value of the correlation exceeds the threshold.
    """
    G = nx.Graph()
    num_nodes = correlation_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(correlation_matrix[i, j]) > threshold:
                # Add an edge with the actual correlation value as the weight
                G.add_edge(i, j, weight=correlation_matrix[i, j])
    return G

def plot_network(G, title=None, decimal_places=None):
    """
    Plot the network with edge weights.
    """
    pos = nx.circular_layout(G)
    labels = {i: f'NV{i}' for i in G.nodes()}
    colors = ['b' if G[u][v]['weight'] > 0 else 'r' for u, v in G.edges()]
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', node_size=700, edge_color=colors, font_weight='bold')
    if decimal_places is not None:
        edge_labels = {k: f"{v:.{decimal_places}f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
    if title:
        plt.title(title)
    plt.show()

def plot_network_with_communities(G, title=None, decimal_places=None):
    pos = nx.spring_layout(G)
    communities = community.louvain_communities(G)
    colors = ['b' if G[u][v]['weight'] > 0 else 'r' for u, v in G.edges()]
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color=colors, font_weight='bold')
    if decimal_places is not None:
        edge_labels = {k: f"{v:.{decimal_places}f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    for community in communities:
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_size=700, alpha=0.5)

    if title:
        plt.title(title)
    plt.show()

# def plot_network(G):
#     # Plot the network
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='grey')
#     labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     plt.show()
    
def plot_network_3d(G, ax):
    # 3D layout
    pos = nx.spring_layout(G, dim=3)

    # Extract node positions
    node_pos = np.array([pos[v] for v in G.nodes()])

    # Draw nodes
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=700, edge_color='grey')

    # Draw edges
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], color='grey')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_network_updated(G, title=None):
    """
    Plot the network with edge weights using a circular layout and a colorbar for edge weights.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an Axes
    pos = nx.circular_layout(G)
    labels = {i: f'NV{i}' for i in G.nodes()}

    # Node colors: first 5 nodes red, rest blue
    node_colors = ['red' if i < 5 else 'blue' for i in G.nodes()]

    # Edge colors based on weight
    edges = G.edges(data=True)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize edge weights for colormap
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.coolwarm
    edge_colors = [cmap(norm(weight)) for weight in weights]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='black', font_weight='bold')

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy empty array for the colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Correlation')

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_network_coords(G, coords, title=None):
    """
    Plot the network with edge weights using specified coordinates and a colorbar for edge weights.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an Axes

    # Create a dictionary for positions
    pos = {i: coords[i] for i in range(len(coords))}
    labels = {i: f'NV{i}' for i in G.nodes()}

    # Node colors: first 5 nodes red, rest blue
    node_colors = ['red' if i < 5 else 'blue' for i in G.nodes()]

    # Edge colors based on weight
    edges = G.edges(data=True)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize edge weights for colormap
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.coolwarm
    edge_colors = [cmap(norm(weight)) for weight in weights]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_size=12)

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy empty array for the colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Correlation Coeff.', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# Centrality Measures
def plot_centrality_measures(G_reference, G_signal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    degree_centrality_reference = nx.degree_centrality(G_reference)
    betweenness_centrality_reference = nx.betweenness_centrality(G_reference)
    closeness_centrality_reference = nx.closeness_centrality(G_reference)
    eigenvector_centrality_reference = nx.eigenvector_centrality(G_reference)
    
    degree_centrality_signal = nx.degree_centrality(G_signal)
    betweenness_centrality_signal = nx.betweenness_centrality(G_signal)
    closeness_centrality_signal = nx.closeness_centrality(G_signal)
    eigenvector_centrality_signal = nx.eigenvector_centrality(G_signal)

    def plot_centrality(ax, centrality, title, color):
        centrality_values = list(centrality.values())
        ax.hist(centrality_values, bins=10, color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Centrality')
        ax.set_ylabel('Frequency')

    plot_centrality(axs[0], degree_centrality_reference, 'Reference Degree Centrality', 'b')
    plot_centrality(axs[1], degree_centrality_signal, 'Signal Degree Centrality', 'r')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_centrality(axs[0], betweenness_centrality_reference, 'Reference Betweenness Centrality', 'b')
    plot_centrality(axs[1], betweenness_centrality_signal, 'Signal Betweenness Centrality', 'r')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_centrality(axs[0], closeness_centrality_reference, 'Reference Closeness Centrality', 'b')
    plot_centrality(axs[1], closeness_centrality_signal, 'Signal Closeness Centrality', 'r')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_centrality(axs[0], eigenvector_centrality_reference, 'Reference Eigenvector Centrality', 'b')
    plot_centrality(axs[1], eigenvector_centrality_signal, 'Signal Eigenvector Centrality', 'r')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Assuming the latest processed file is used
    processed_files = sorted([f for f in os.listdir('data/processed') if f.startswith('processed_counts_')], reverse=True)
    if processed_files:
        latest_processed_file = os.path.join('data', 'processed', processed_files[0])
    else:
        raise FileNotFoundError("No processed files found in the data/processed directory.")
    
    counts_array = load_processed_data(latest_processed_file)
    correlation_matrix = compute_correlation_matrix(counts_array)
    
    G = build_network(correlation_matrix, threshold=0.5)
    plot_network(G)
