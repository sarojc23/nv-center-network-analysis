import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import os

def load_processed_data(file_path):
    # Load processed data
    counts_array = np.load(file_path)
    return counts_array

def compute_correlation_matrix(counts_array):
    counts_array = np.array(counts_array)     # Convert counts_array to a NumPy array
    counts_2d = counts_array.reshape(counts_array.shape[0], -1)     # Reshape the counts array to a 2D array
    correlation_matrix = np.corrcoef(counts_2d)     # Compute the correlation matrix
    return correlation_matrix

def build_network(correlation_matrix, threshold=0.5):
    # Create a graph from the correlation matrix
    G = nx.Graph()
    num_nodes = correlation_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if correlation_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=correlation_matrix[i, j])
    return G

def plot_network(G):
    # Plot the network
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='grey')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def plot_degree_distribution(G_reference, G_signal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    degrees_reference = [G_reference.degree(n) for n in G_reference.nodes()]
    axs[0].hist(degrees_reference, bins=10, color='b', alpha=0.7)
    axs[0].set_title('Reference Counts Degree Distribution')
    axs[0].set_xlabel('Degree')
    axs[0].set_ylabel('Frequency')

    degrees_signal = [G_signal.degree(n) for n in G_signal.nodes()]
    axs[1].hist(degrees_signal, bins=10, color='r', alpha=0.7)
    axs[1].set_title('Signal Counts Degree Distribution')
    axs[1].set_xlabel('Degree')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_clustering_coefficient(G_reference, G_signal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    clustering_reference = nx.clustering(G_reference).values()
    axs[0].hist(clustering_reference, bins=10, color='b', alpha=0.7)
    axs[0].set_title('Reference Counts Clustering Coefficient')
    axs[0].set_xlabel('Clustering Coefficient')
    axs[0].set_ylabel('Frequency')

    clustering_signal = nx.clustering(G_signal).values()
    axs[1].hist(clustering_signal, bins=10, color='r', alpha=0.7)
    axs[1].set_title('Signal Counts Clustering Coefficient')
    axs[1].set_xlabel('Clustering Coefficient')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_shortest_path_length(G_reference, G_signal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    path_length_reference = dict(nx.shortest_path_length(G_reference))
    path_length_reference = [length for lengths in path_length_reference.values() for length in lengths.values()]
    axs[0].hist(path_length_reference, bins=10, color='b', alpha=0.7)
    axs[0].set_title('Reference Counts Shortest Path Length')
    axs[0].set_xlabel('Path Length')
    axs[0].set_ylabel('Frequency')

    path_length_signal = dict(nx.shortest_path_length(G_signal))
    path_length_signal = [length for lengths in path_length_signal.values() for length in lengths.values()]
    axs[1].hist(path_length_signal, bins=10, color='r', alpha=0.7)
    axs[1].set_title('Signal Counts Shortest Path Length')
    axs[1].set_xlabel('Path Length')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_community_detection(G_reference, G_signal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    communities_reference = greedy_modularity_communities(G_reference)
    community_map_reference = {}
    for i, community in enumerate(communities_reference):
        for node in community:
            community_map_reference[node] = i
    colors_reference = [community_map_reference[node] for node in G_reference]
    pos_reference = nx.spring_layout(G_reference)
    nx.draw(G_reference, pos_reference, node_color=colors_reference, with_labels=True, node_size=500, cmap=plt.cm.rainbow, ax=axs[0])
    axs[0].set_title('Reference Counts Community Detection')

    communities_signal = greedy_modularity_communities(G_signal)
    community_map_signal = {}
    for i, community in enumerate(communities_signal):
        for node in community:
            community_map_signal[node] = i
    colors_signal = [community_map_signal[node] for node in G_signal]
    pos_signal = nx.spring_layout(G_signal)
    nx.draw(G_signal, pos_signal, node_color=colors_signal, with_labels=True, node_size=500, cmap=plt.cm.rainbow, ax=axs[1])
    axs[1].set_title('Signal Counts Community Detection')

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
