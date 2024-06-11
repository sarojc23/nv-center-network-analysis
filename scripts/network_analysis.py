import networkx as nx
import pandas as pd
import numpy as np
from src.utils import compute_correlations, build_network
from src.visualization import plot_network

def main():
    data_path = "../data/processed/nv_center_data_processed.csv"
    data = pd.read_csv(data_path)
    
    correlations = compute_correlations(data)
    G = build_network(correlations)
    
    plot_network(G)

if __name__ == "__main__":
    main()
