import numpy as np
import networkx as nx

# def compute_correlations(data):
#     return np.corrcoef(data.T)

# def build_network(correlations, threshold=0.5):
#     G = nx.Graph()
#     num_nodes = correlations.shape[0]
    
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             if abs(correlations[i, j]) > threshold:
#                 G.add_edge(i, j, weight=correlations[i, j])
    
#     return G

import pandas as pd
import numpy as np

def extract_nv_data(data, key):
    return np.array([nv[key] for nv in data['nv_list']])

def calculate_distances(nv_data):
    distances = np.linalg.norm(nv_data[:, np.newaxis] - nv_data, axis=2)
    return distances
