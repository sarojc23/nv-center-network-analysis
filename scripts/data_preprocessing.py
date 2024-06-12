import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def load_data(file_path):
    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

def preprocess_counts_data(data):
    counts_data = data['counts']
    counts_array = np.array(counts_data)
    return counts_array

def save_processed_data(counts_array, output_dir):
    # Create a smart filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"processed_counts_{timestamp}.npy"
    output_path = os.path.join(output_dir, filename)
    # Save the processed data
    np.save(output_path, counts_array)
    return output_path

if __name__ == "__main__":
    input_path = os.path.join('data', 'raw', '2024_05_01-10_21_43-johnson-nv0_2024_03_12.txt')
    output_dir = os.path.join('data', 'processed')
    
    data = load_data(input_path)
    counts_array = preprocess_counts_data(data)
    output_path = save_processed_data(counts_array, output_dir)
    print(f"Processed data saved to: {output_path}")
